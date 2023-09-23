from deformable_detr import DeformableDETR
from util.misc import NestedTensor

class DeformableDETRADV(DeformableDETR):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, aux_loss=True, with_box_refine=False, two_stage=False):
        super().__init__(backbone, transformer, num_classes, num_queries, num_feature_levels, aux_loss, with_box_refine, two_stage)

    def forward(self, samples: NestedTensor):
        out = super().forward(samples)
        # only return pred_logits
        return out["pred_logits"]