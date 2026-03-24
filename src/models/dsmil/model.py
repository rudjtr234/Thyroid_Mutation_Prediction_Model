import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import create_mlp
from ..mil_template import MIL


class DSMILModel(MIL):
    """
    DSMIL (Dual-Stream MIL) style model for bag-level classification.
    """

    def __init__(
        self,
        in_dim: int = 1536,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        dropout: float = 0.25,
        temperature: float = 1.0,
        num_classes: int = 2,
        **kwargs,
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)

        self.temperature = max(1e-6, float(temperature))

        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False,
        )
        self.instance_classifier = nn.Linear(embed_dim, num_classes)
        self.bag_classifier = nn.Linear(embed_dim, num_classes)

        self.initialize_weights()

    def forward_attention(self, h: torch.Tensor, attn_mask=None, attn_only: bool = True):
        h = self.patch_embed(h)  # [B, N, D]
        instance_logits = self.instance_classifier(h)  # [B, N, C]

        if self.num_classes > 1:
            critical_scores = instance_logits[..., 1]
        else:
            critical_scores = instance_logits[..., 0]

        critical_idx = torch.argmax(critical_scores, dim=1)  # [B]
        batch_indices = torch.arange(h.size(0), device=h.device)
        critical_feat = h[batch_indices, critical_idx]  # [B, D]

        attn_logits = torch.bmm(h, critical_feat.unsqueeze(-1)).squeeze(-1)
        attn_logits = attn_logits / (math.sqrt(h.size(-1)) * self.temperature)

        if attn_mask is not None:
            attn_logits = attn_logits + (1 - attn_mask) * torch.finfo(attn_logits.dtype).min

        attn_logits = attn_logits.unsqueeze(1)  # [B, 1, N]

        if attn_only:
            return attn_logits
        return h, instance_logits, attn_logits

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_attention: bool = True):
        h, instance_logits, attn_logits = self.forward_attention(
            h,
            attn_mask=attn_mask,
            attn_only=False,
        )
        A = F.softmax(attn_logits, dim=-1)
        bag_feats = torch.bmm(A, h).squeeze(dim=1)

        log_dict = {
            "attention": attn_logits if return_attention else None,
            "A": A if return_attention else None,
            "instance_logits": instance_logits,
            "max_instance_logits": instance_logits.max(dim=1).values,
        }
        return bag_feats, log_dict

    def forward_head(self, h: torch.Tensor):
        return self.bag_classifier(h)

    def forward(
        self,
        h: torch.Tensor,
        loss_fn: nn.Module = None,
        label: torch.LongTensor = None,
        attn_mask=None,
        return_attention: bool = False,
        return_slide_feats: bool = False,
        return_extra: bool = False,
    ):
        bag_feats, log_dict = self.forward_features(
            h,
            attn_mask=attn_mask,
            return_attention=return_attention,
        )
        bag_logits = self.forward_head(bag_feats)
        logits = 0.5 * (bag_logits + log_dict["max_instance_logits"])

        cls_loss = MIL.compute_loss(loss_fn, logits, label)

        if return_extra:
            return {
                "logits": logits,
                "loss": cls_loss,
                "attention": log_dict.get("attention"),
                "A": log_dict.get("A"),
                "instance_logits": log_dict.get("instance_logits"),
                "slide_feats": bag_feats if return_slide_feats else None,
            }

        results_dict = {"logits": logits, "loss": cls_loss}
        log_dict["loss"] = cls_loss.item() if cls_loss is not None else -1
        if return_slide_feats:
            log_dict["slide_feats"] = bag_feats
        return results_dict, log_dict
