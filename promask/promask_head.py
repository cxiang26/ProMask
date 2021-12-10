# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS, HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale, ConvModule
from mmcv.runner import force_fp32
from mmcv.ops import DeformConv2d

from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean, bbox_overlaps
import numpy as np

INF = 1e8
EPS = 1e-8
PRO_DIM = 2
EXT_DIM = 2


class FeatureAlign(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4,
                 flag_norm=True):
        super(FeatureAlign, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(4,
                                     deformable_groups * offset_channels,
                                     1,
                                     bias=False)
        self.conv_adaption = DeformConv2d(in_channels,
                                          out_channels,
                                          kernel_size=kernel_size,
                                          padding=(kernel_size - 1) // 2,
                                          deform_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.GroupNorm(32, in_channels)
        self.flag_norm = flag_norm

    def init_weights(self, bias_value=0):
        torch.nn.init.normal_(self.conv_offset.weight, std=0.01)
        torch.nn.init.normal_(self.conv_adaption.weight, std=0.01)

    def forward(self, x, shape):
        offset = self.conv_offset(shape.detach())
        if self.flag_norm:
            x = self.relu(self.norm(self.conv_adaption(x, offset)))
        else:
            x = self.relu(self.conv_adaption(x, offset))
        return x

def crop_split(mask_pred, boxes, masksG=None):
    h, w, n = mask_pred.size()
    rows = torch.arange(w, device=mask_pred.device, dtype=boxes.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=mask_pred.device, dtype=boxes.dtype).view(-1, 1, 1).expand(h, w, n)

    x1, x2 = boxes[:, 0], boxes[:, 2]
    y1, y2 = boxes[:, 1], boxes[:, 3]
    x1 = torch.clamp(x1, min=0, max=w - 1)
    y1 = torch.clamp(y1, min=0, max=h - 1)
    x2 = torch.clamp(x2, min=0, max=w - 1)
    y2 = torch.clamp(y2, min=0, max=h - 1)

    ##x1,y1,x2,y2
    crop_mask = (rows >= x1.view(1, 1, -1)) & (rows < x2.view(1, 1, -1)) & (cols >= y1.view(1, 1, -1)) & (cols < y2.view(1, 1, -1))
    crop_mask = crop_mask.float().detach()
    masks = mask_pred * crop_mask
    if masksG is not None:
        masksG = masksG * crop_mask
        return masks, masksG
    return masks


@DETECTORS.register_module()
class PROMaskHead(AnchorFreeHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.feat_dim = 32
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.conv_subspace = nn.Conv2d(self.feat_channels, self.feat_dim * PRO_DIM, 3, padding=1)
        self.conv_ext = nn.Conv2d(self.feat_channels, EXT_DIM, 3, padding=1)
        self.feat_align = FeatureAlign(self.feat_channels, self.feat_channels, 3, flag_norm=self.norm_cfg is not None)
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(inplace=True)

        ## mask ##
        self.promask_lat = nn.Conv2d(512, self.feat_dim, 3, padding=1)
        self.promask_lat0 = nn.Conv2d(768, 512, 1, padding=0)

        self.dy_t = nn.Parameter(torch.tensor(0.1, dtype=torch.float))


    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        centernesses = []
        subspace_preds = []
        ext_preds = []
        feat_masks = []
        count = 0
        for x, scale, stride in zip(feats, self.scales, self.strides):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs:
                cls_feat = cls_layer(cls_feat)

            for reg_layer in self.reg_convs:
                reg_feat = reg_layer(reg_feat)

            bbox_pred = self.conv_reg(reg_feat)

            cls_feat = self.feat_align(cls_feat, bbox_pred)
            cls_score = self.conv_cls(cls_feat)
            subspace_pred = self.conv_subspace(cls_feat)
            ext_pred = self.conv_ext(cls_feat)

            if self.centerness_on_reg:
                centerness = self.conv_centerness(reg_feat)
            else:
                centerness = self.conv_centerness(cls_feat)
            # scale the bbox_pred of different level
            # float to avoid overflow when enabling FP16
            bbox_pred = scale(bbox_pred).float()
            if self.norm_on_bbox:
                bbox_pred = F.relu(bbox_pred)
                if not self.training:
                    bbox_pred *= stride
            else:
                bbox_pred = bbox_pred.exp()

            if count < 3:
                if count == 0:
                    feat_masks.append(reg_feat)
                else:
                    feat_up = F.interpolate(reg_feat, scale_factor=(2 ** count), mode='bilinear', align_corners=False)
                    feat_masks.append(feat_up)
            count = count + 1

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            centernesses.append(centerness)
            subspace_preds.append(subspace_pred)
            ext_preds.append(ext_pred)

        feat_masks = torch.cat(feat_masks, dim=1)
        feat_masks = self.promask_lat(self.relu(self.promask_lat0(feat_masks)))
        # feat_masks = F.interpolate(feat_masks, scale_factor=4, mode='bilinear', align_corners=False)
        return cls_scores, bbox_preds, centernesses, subspace_preds, ext_preds, feat_masks

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             subspace_preds,
             ext_preds,
             feat_masks,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             gt_masks_list=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets, label_list, bbox_targets_list, gt_inds = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores[0].size(0)
        num_levels = len(bbox_preds)
        det_bboxes = []
        for img_id in range(num_imgs):
            bbox_pred_list = [bbox_preds[i][img_id].permute(1, 2, 0).reshape(-1, 4).detach() for i in range(num_levels)]

            bboxes = []
            for i in range(len(bbox_pred_list)):
                if self.norm_on_bbox:
                    bbox_pred = bbox_pred_list[i] * self.strides[i]
                else:
                    bbox_pred = bbox_pred_list[i]
                points = all_level_points[i]
                bboxes.append(distance2bbox(points, bbox_pred))
            bboxes = torch.cat(bboxes, dim=0)
            det_bboxes.append(bboxes)

        gt_masks = []
        for i in range(len(gt_labels)):
            gt_label = gt_labels[i]
            gt_masks.append(torch.from_numpy(np.array(gt_masks_list[i].masks[:gt_label.shape[0]], dtype=np.float32)).to(gt_label.device))

        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        # ========mask========== #
        # ====================== #
        flatten_cls_scores1 = [cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels) for cls_score in cls_scores]
        flatten_cls_scores1 = torch.cat(flatten_cls_scores1, dim=1)

        flatten_subspace_preds = [subspace_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.feat_dim * PRO_DIM) for subspace_pred in subspace_preds]
        flatten_subspace_preds = torch.cat(flatten_subspace_preds, dim=1)

        flatten_ext_preds = [ext_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, EXT_DIM) for ext_pred in ext_preds]
        flatten_ext_preds = torch.cat(flatten_ext_preds, dim=1)

        loss_mask = 0
        for i in range(num_imgs):
            l_labels = torch.cat([labels_level.flatten() for labels_level in label_list[i]])
            bbox_dt = det_bboxes[i] / 2
            bbox_dt = bbox_dt.detach()
            pos_inds = ((l_labels >= 0) & (l_labels < bg_class_ind)).nonzero().reshape(-1)
            subspace_pred = flatten_subspace_preds[i][pos_inds]
            ext_pred = flatten_ext_preds[i][pos_inds]
            img_mask = feat_masks[i]
            mask_h = img_mask.shape[1]
            mask_w = img_mask.shape[2]
            idx_gt = gt_inds[i]
            bbox_dt = bbox_dt[pos_inds, :4]

            area = (bbox_dt[:, 2] - bbox_dt[:, 0]) * (bbox_dt[:, 3] - bbox_dt[:, 1])
            bbox_dt = bbox_dt[area > 1.0, :]
            if bbox_dt.shape[0] == 0:
                loss_mask += flatten_subspace_preds[i].sum() * img_mask.sum() * 0
                continue

            idx_gt = idx_gt[area > 1.0]
            subspace_pred = subspace_pred[area > 1.0]
            cls_score = flatten_cls_scores1[i, pos_inds, l_labels[pos_inds]].sigmoid().detach()
            cls_score = cls_score[area > 1.0]
            ext_pred = ext_pred[area > 1.0]
            bbox_gt = gt_bboxes[i]
            # bbox_gt = bbox_gt[idx_gt] / 2
            # ious = bbox_overlaps(bbox_gt[idx_gt] / 2, bbox_dt, is_aligned=True)
            # with torch.no_grad():
            #     weighting = cls_score * ious
            #     weighting = weighting / (torch.sum(weighting) + 0.0001) * len(weighting)

            gt_mask = F.interpolate(gt_masks[i].unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
            shape = np.minimum((feat_masks[i].shape[0], mask_h * 4, mask_w * 4), gt_mask.shape)
            gt_mask_new = gt_mask.new_zeros(gt_mask.shape[0], mask_h * 4, mask_w * 4)
            gt_mask_new[:gt_mask.shape[0], :shape[1], :shape[2]] = gt_mask[:gt_mask.shape[0], :shape[1], :shape[2]]
            gt_mask_new = gt_mask_new.gt(0.5).float()
            gt_mask_new = torch.index_select(gt_mask_new, 0, idx_gt).permute(1, 2, 0).contiguous()

            # =====projection==== #
            img_mask1 = img_mask.permute(1, 2, 0)

            # # baseline
            # mask_pred = img_mask1.unsqueeze(-2).relu() @ subspace_pred.tanh().t()
            # mask_pred = mask_pred.permute(3, 2, 0, 1)
            # mask_pred = F.interpolate(mask_pred, scale_factor=4, mode='bilinear', align_corners=False)
            # mask_pred = mask_pred.squeeze(dim=1).permute(1, 2, 0)
            # mask_pred = torch.sigmoid(mask_pred)

            # projection
            _, _, img_c = img_mask1.shape
            basis_v = subspace_pred.view(subspace_pred.shape[0], self.feat_dim, PRO_DIM)
            W = torch.matmul(basis_v.permute(0, 2, 1), basis_v)
            # approximate iteration method - Newton-Schulz iterations
            sigma, left_v = batch_sqrtm(W)

            # svd is extremely slow in GPU
            # eye = torch.eye(PRO_DIM, device=W.device).expand_as(W)
            # sigma = torch.inverse(W + EPS * eye)
            # u, s, _ = torch.svd(sigma)
            # s = s.sqrt().diag_embed()
            # left_v = u @ s
            left_pro = basis_v @ left_v
            pro_feat = img_mask1.view(1, mask_h * mask_w, 1, img_c)
            pro_mask = pro_feat @ left_pro.unsqueeze(dim=1)
            mask_pred = torch.cosine_similarity(pro_mask.squeeze(-2), ext_pred.unsqueeze(dim=-2), dim=-1)
            # mask_pred = pro_mask.norm(dim=-1).squeeze() - torch.relu(self.dy_t) + torch.tanh(ext_pred)

            mask_pred = mask_pred.view(mask_pred.shape[0], mask_h, mask_w)
            mask_pred = F.interpolate(mask_pred.unsqueeze(dim=1), scale_factor=4, mode='bilinear', align_corners=False)
            mask_pred = mask_pred.squeeze(dim=1).permute(1, 2, 0)
            mask_pred = torch.sigmoid(5 * mask_pred) # norm->1, cosine->5
            mask_pred, gt_mask_crop = crop_split(mask_pred, bbox_dt, gt_mask_new)

            bbox_ious = bbox_overlaps(bbox_gt[idx_gt] / 2, bbox_dt, is_aligned=True)
            # mask_ious = gt_mask_crop.sum(dim=(0,1)) / (gt_mask_new.sum(dim=(0,1)) + 0.1)
            with torch.no_grad():
                weighting = cls_score * bbox_ious
                weighting = weighting / (torch.sum(weighting) + 0.0001) * len(weighting)

            pre_loss = F.binary_cross_entropy(mask_pred, gt_mask_crop, reduction='none')
            wh = bbox_dt[:, 2:] - bbox_dt[:, :2]
            dt_box_width = wh[:, 0]
            dt_box_height = wh[:, 1]
            pre_loss = pre_loss.sum(dim=(0, 1)) / dt_box_width / dt_box_height / bbox_dt.shape[0]
            loss_mask += torch.sum(pre_loss * weighting.detach())

        loss_mask = loss_mask / num_imgs

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_mask=loss_mask)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   subspace_preds,
                   ext_preds,
                   feat_masks,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        centerness_pred_list = [centernesses[i].detach() for i in range(num_levels)]
        subspace_pred_list = [subspace_preds[i].detach() for i in range(num_levels)]
        ext_pred_list = [ext_preds[i].detach() for i in range(num_levels)]
        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        ori_shape = [ img_metas[i]['ori_shape'] for i in range(cls_scores[0].shape[0])]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
                                       centerness_pred_list, subspace_pred_list, ext_pred_list, feat_masks, mlvl_points,
                                       img_shapes, ori_shape, scale_factors, cfg, rescale,
                                       with_nms)
        return result_list

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    centernesses,
                    subspace_preds,
                    ext_preds,
                    feat_masks,
                    mlvl_points,
                    img_shapes,
                    ori_shape,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        mlvl_subspaces = []
        mlvl_exts = []
        for cls_score, bbox_pred, centerness, subspace_pred, ext_pred, points in zip(
                cls_scores, bbox_preds, centernesses, subspace_preds, ext_preds, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(0, 2, 3, 1).reshape(batch_size, -1).sigmoid()

            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            subspace_pred = subspace_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, self.feat_dim * PRO_DIM)
            ext_pred = ext_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, EXT_DIM)
            points = points.expand(batch_size, -1, 2)
            # Get top-k prediction
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                max_scores, _ = (scores * centerness[..., None]).max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                if torch.onnx.is_in_onnx_export():
                    transformed_inds = bbox_pred.shape[
                                           1] * batch_inds + topk_inds
                    points = points.reshape(-1,
                                            2)[transformed_inds, :].reshape(
                        batch_size, -1, 2)
                    bbox_pred = bbox_pred.reshape(
                        -1, 4)[transformed_inds, :].reshape(batch_size, -1, 4)
                    scores = scores.reshape(
                        -1, self.num_classes)[transformed_inds, :].reshape(
                        batch_size, -1, self.num_classes)
                    centerness = centerness.reshape(
                        -1, 1)[transformed_inds].reshape(batch_size, -1)
                else:
                    points = points[batch_inds, topk_inds, :]
                    bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                    scores = scores[batch_inds, topk_inds, :]
                    centerness = centerness[batch_inds, topk_inds]
                    subspace_pred = subspace_pred[batch_inds, topk_inds, :]
                    ext_pred = ext_pred[batch_inds, topk_inds, :]

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_subspaces.append(subspace_pred)
            mlvl_exts.append(ext_pred)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)
        batch_mlvl_subspaces = torch.cat(mlvl_subspaces, dim=1)
        batch_mlvl_exts = torch.cat(mlvl_exts, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export() and with_nms:
            from mmdet.core.export import add_dummy_nms_for_onnx
            batch_mlvl_scores = batch_mlvl_scores * (
                batch_mlvl_centerness.unsqueeze(2))
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size, batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores,
                 mlvl_centerness) in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                         batch_mlvl_centerness):
                det_bbox, det_label, inds = multiclass_nms(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=mlvl_centerness,
                    return_inds=True)

                # ====================================
                batch_mlvl_subspaces = batch_mlvl_subspaces.reshape(-1, self.feat_dim * PRO_DIM)
                batch_mlvl_subspaces = batch_mlvl_subspaces[:, None, :].expand(batch_mlvl_subspaces.shape[0], self.num_classes, self.feat_dim * PRO_DIM)
                batch_mlvl_subspaces = batch_mlvl_subspaces.reshape(-1, self.feat_dim * PRO_DIM)

                batch_mlvl_exts = batch_mlvl_exts.reshape(-1, EXT_DIM)
                batch_mlvl_exts = batch_mlvl_exts[:, None, :].expand(batch_mlvl_exts.shape[0], self.num_classes, EXT_DIM)
                batch_mlvl_exts = batch_mlvl_exts.reshape(-1, EXT_DIM)
                det_subspaces = batch_mlvl_subspaces[inds, :]
                det_exts = batch_mlvl_exts[inds, :]

                img_mask1 = feat_masks[0].permute(1, 2, 0)

                # # baseline
                # mask_pred = img_mask1.unsqueeze(dim=-2).relu() @ det_subspaces.t().tanh()
                # mask_pred = mask_pred.permute(3, 2, 0, 1)
                # mask_pred = F.interpolate(mask_pred, scale_factor=(8/scale_factors[0][0], 8/scale_factors[0][1]), mode='bilinear', align_corners=False)
                # mask_pred = mask_pred.squeeze(dim=1).permute(1, 2, 0)
                # mask_pred = torch.sigmoid(mask_pred)

                # projection
                mask_h, mask_w, _ = img_mask1.shape
                basis_v = det_subspaces.view(det_subspaces.shape[0], self.feat_dim, PRO_DIM)
                W = torch.matmul(basis_v.permute(0, 2, 1), basis_v)

                # approximate iteration method - Newton-Schulz iterations
                sigma, left_v = batch_sqrtm(W)

                # svd is extremely slow in GPU
                # eye = torch.eye(PRO_DIM, device=W.device).expand_as(W)
                # sigma = torch.inverse(W + EPS * eye)
                # u, s, _ = torch.svd(sigma)
                # s = s.sqrt().diag_embed()
                # left_v = u @ s
                left_pro = basis_v @ left_v
                pro_feat = img_mask1.view(1, mask_h * mask_w, 1, self.feat_dim)
                pro_mask = pro_feat @ left_pro.unsqueeze(dim=1)
                mask_pred = torch.cosine_similarity(pro_mask.squeeze(-2), det_exts.unsqueeze(dim=-2), dim=-1)
                # mask_pred = pro_mask.squeeze(2).norm(dim=-1) - torch.relu(self.dy_t) + torch.tanh(det_exts)

                # left_w = basis_v @ sigma
                # pro_feat = img_mask1.view(1, -1, 1, self.feat_dim)
                # pro_feat_r = basis_v.permute(0, 2, 1).unsqueeze(dim=1) @ pro_feat.permute(0, 1, 3, 2)
                # pro_feat = pro_feat @ left_w.unsqueeze(dim=1)
                # pro_mask = pro_feat @ pro_feat_r
                # mask_pred = pro_mask.squeeze() - torch.relu(det_exts)

                mask_pred = mask_pred.view(mask_pred.shape[0], mask_h, mask_w)
                mask_pred = F.interpolate(mask_pred.unsqueeze(dim=1), scale_factor=(8/scale_factors[0][1], 8/scale_factors[0][0]), mode='bilinear',
                                          align_corners=False)
                mask_pred = mask_pred.squeeze(dim=1).permute(1, 2, 0)
                mask_pred = torch.sigmoid(5 * mask_pred)  # norm->1 cosine->5
                mask_pred = crop_split(mask_pred, det_bbox, masksG=None) > 0.4


                cls_segms = [[] for _ in range(self.num_classes)]
                for i in range(det_bbox.shape[0]):
                    label = det_label[i]
                    mask = mask_pred[:,:,i].cpu().numpy().astype(np.uint8)
                    im_mask = np.zeros((ori_shape[0][0], ori_shape[0][1]), dtype=np.uint8)
                    shape = np.minimum(mask.shape, ori_shape[0][0:2])
                    im_mask[:shape[0], :shape[1]] = mask[:shape[0], :shape[1]]
                    cls_segms[label].append(im_mask)
                det_results.append(tuple([det_bbox, det_label, cls_segms]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                   batch_mlvl_centerness)
            ]
        return det_results

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, gt_inds = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets, labels_list, bbox_targets_list, gt_inds

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_labels.new_empty(0)

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
                (max_regress_distance >= regress_ranges[..., 0])
                & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        gt_inds = min_area_inds[labels < self.num_classes]

        return labels, bbox_targets, gt_inds

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                                         left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                                         top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

def batch_sqrtm(A, numIters=10, reg=2.0):
    """
    Batch matrix root via Newton-Schulz iterations
    """
    # batchSize = A.shape[0]
    # dim = A.shape[1]
    # Renormalize so that the each matrix has a norm lesser than 1/reg, but only normalize when necessary

    normA = reg * torch.sqrt(1e-6 + torch.sum(torch.square(A), dim=(1, 2), keepdim=True))
    # print('I am here-1',normA.get_shape())
    ones = torch.ones_like(normA)
    renorm_factor = torch.where(normA > ones, normA, ones)
    # print('I am here0',renorm_factor.get_shape())
    Y = A / renorm_factor
    # print('I am here1',Y.get_shape())
    I = torch.eye(PRO_DIM, device=A.device).expand_as(A)

    # I=I.reshape([1,dim,dim])
    # I= T.repeat(I,bs,axis=0)
    Z = torch.eye(PRO_DIM, device=A.device).expand_as(A)  # I.copy()
    for i in range(numIters):
        t = 0.5 * (3.0 * I - torch.matmul(Z, Y))
        # t=0.5*(3.0*I-th.tensor.batched_dot(Z,Y))
        # Y=th.tensor.batched_dot(Y,t)
        Y = torch.matmul(Y, t)
        Z = torch.matmul(t, Z)
        # Z=th.tensor.batched_dot(t,Z)
    sA = Y * torch.sqrt(renorm_factor)
    sAinv = Z / torch.sqrt(renorm_factor)
    return sA, sAinv
