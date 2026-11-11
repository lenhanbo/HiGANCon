"""HiGAN training extensions for writer-disjoint IAM experiments."""

from .losses import kl_loss, supervised_contrastive_loss

__all__ = ["kl_loss", "supervised_contrastive_loss"]
