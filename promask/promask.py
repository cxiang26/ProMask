# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.core import bbox2result

@DETECTORS.register_module()
class PROMask(SingleStageDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PROMask, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      gt_masks=None,
                      **kwargs):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, gt_masks_list=gt_masks)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def simple_test(self, img, img_metas, rescale=False):
        feat = self.extract_feat(img)
        outs = self.bbox_head(feat)
        inputs = outs + (img_metas,)
        results_list = self.bbox_head.get_bboxes(*inputs, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels, aa in results_list
        ]
        segm_results = [
            aa for _, _, aa in results_list
        ]
        return list(zip(bbox_results, segm_results))