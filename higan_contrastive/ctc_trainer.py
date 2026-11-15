from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from networks.utils import set_requires_grad

from .checkpoint import git_commit
from .ctc_matching import (CTCAlignment, ctc_forward_backward, interpolate_alignment,
                           soft_patch_cosine)
from .ctc_preview import save_alignment_heatmap, save_ctc_preview
from .losses import (kl_loss, optional_contrastive_loss, scheduled_weight_at_step)
from .trainer import HiGANTrainer


def paired_indices(batch_size: int, device: torch.device | None = None) -> torch.Tensor:
    if batch_size <= 0 or batch_size % 2:
        raise ValueError("CTC-guided matching requires a positive even batch size")
    return torch.arange(batch_size, device=device) ^ 1


def validate_bidirectional_pairs(batch: dict) -> None:
    count = len(batch["writer_ids"])
    if count % 2:
        raise ValueError("Writer-pair batch must contain an even number of samples")
    for first in range(0, count, 2):
        second = first + 1
        if batch["writer_ids"][first] != batch["writer_ids"][second]:
            raise ValueError(f"Pair {first}/{second} does not have the same writer")
        if batch["word_ids"][first] == batch["word_ids"][second]:
            raise ValueError(f"Pair {first}/{second} contains the same sample twice")


def _pad_last(tensor: torch.Tensor, width: int, value: float = 0.0) -> torch.Tensor:
    return F.pad(tensor, (0, width - tensor.shape[-1]), value=value)


class CTCSoftTrainer(HiGANTrainer):
    """HiGAN training with opt-out SupCon and CTC-guided soft patch matching."""

    def __init__(self, *args, enable_supcon: bool = True, enable_ctc_guided: bool = True,
                 **kwargs):
        self.enable_supcon = bool(enable_supcon)
        self.enable_ctc_guided = bool(enable_ctc_guided)
        cfg = args[0] if args else kwargs["cfg"]
        self.match_start_step = int(cfg.ctc_match.start_step)
        self.match_ramp_steps = int(cfg.ctc_match.ramp_steps)
        super().__init__(*args, enable_supcon=self.enable_supcon, **kwargs)
        if self.match_start_step < 1 or self.match_ramp_steps < 0:
            raise ValueError("ctc_match start_step must be >=1 and ramp_steps must be >=0")

    @property
    def active_losses(self) -> list[str]:
        result = ["higan"]
        if self.enable_supcon:
            result.append("supcon")
        if self.enable_ctc_guided:
            result.append("ctc-guided")
        return result

    def _supcon_weight(self, step: int) -> float:
        if not self.enable_supcon:
            return 0.0
        return scheduled_weight_at_step(
            step, float(self.cfg.contrastive.lambda_con), self.contrastive_start_step,
            self.contrastive_ramp_steps)

    def _match_weight(self, step: int) -> float:
        if not self.enable_ctc_guided:
            return 0.0
        return scheduled_weight_at_step(
            step, float(self.cfg.ctc_match.lambda_match), self.match_start_step,
            self.match_ramp_steps)

    def _auxiliary_state(self, step: int | None = None) -> dict:
        step = max(1, self.global_step if step is None else step)
        return {
            "active_losses": self.active_losses,
            "contrastive": {
                "enabled": self.enable_supcon,
                "target_weight": float(self.cfg.contrastive.lambda_con),
                "start_step": self.contrastive_start_step,
                "ramp_steps": self.contrastive_ramp_steps,
                "effective_weight": self._supcon_weight(step),
            },
            "ctc_match": {
                "enabled": self.enable_ctc_guided,
                "target_weight": float(self.cfg.ctc_match.lambda_match),
                "start_step": self.match_start_step,
                "ramp_steps": self.match_ramp_steps,
                "effective_weight": self._match_weight(step),
            },
        }

    def _write_run_metadata(self) -> None:
        # During the base constructor the CTC schedule is read directly from cfg.
        if not hasattr(self, "match_start_step"):
            self.match_start_step = int(self.cfg.ctc_match.start_step)
            self.match_ramp_steps = int(self.cfg.ctc_match.ramp_steps)
        metadata = {
            "git_commit": git_commit(),
            "config_hash": self.config_hash, "split_hash": self.split_hash,
            "initial_state_hash": self.initial_state_hash, "device": str(self.device),
            "train_images": len(self.dataset),
            "train_writers": len(self.dataset.writer_to_index),
            "dataset_format": self.dataset_format,
            "max_steps_this_session": self.max_steps,
            **self._auxiliary_state(max(1, self.global_step)),
        }
        (self.output / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @contextmanager
    def _eval_models(self, *models):
        modes = [model.training for model in models]
        try:
            for model in models:
                model.eval()
            yield
        finally:
            for model, mode in zip(models, modes):
                model.train(mode)

    def _alignment(self, images: torch.Tensor, image_lengths: torch.Tensor,
                   labels: torch.Tensor, label_lengths: torch.Tensor) -> CTCAlignment:
        recognizer = self.models["R"]
        with self._eval_models(recognizer), torch.no_grad():
            logits = recognizer(images)
            return ctc_forward_backward(
                logits, labels, image_lengths // recognizer.len_scale, label_lengths,
                blank=0, eps=float(self.cfg.ctc_match.eps))

    def _match_loss(self, real_images: torch.Tensor, real_lengths: torch.Tensor,
                    generated: torch.Tensor, generated_lengths: torch.Tensor,
                    labels: torch.Tensor, label_lengths: torch.Tensor
                    ) -> tuple[torch.Tensor, dict, CTCAlignment, CTCAlignment, torch.Tensor]:
        real_alignment = self._alignment(real_images, real_lengths, labels, label_lengths)
        fake_alignment = self._alignment(generated, generated_lengths, labels, label_lengths)
        writer = self.models["W"]
        with self._eval_models(writer.cnn_backbone):
            with torch.no_grad():
                real_features = writer.cnn_backbone(real_images)
            fake_features = writer.cnn_backbone(generated)
        real_weights, _ = interpolate_alignment(
            real_alignment.weights, real_lengths // self.models["R"].len_scale,
            real_features.shape[-1], real_alignment.weights.shape[-1],
            float(self.cfg.ctc_match.eps))
        fake_weights, _ = interpolate_alignment(
            fake_alignment.weights, generated_lengths // self.models["R"].len_scale,
            fake_features.shape[-1], fake_alignment.weights.shape[-1],
            float(self.cfg.ctc_match.eps))
        valid = real_alignment.valid & fake_alignment.valid
        loss, cosine, mask = soft_patch_cosine(
            real_features, fake_features, real_weights, fake_weights,
            real_alignment.occurrence_mask & fake_alignment.occurrence_mask, valid,
            int(self.cfg.ctc_match.patch_width), float(self.cfg.ctc_match.eps))
        valid_count = int(valid.sum().item())
        metrics = {
            "match_valid_ratio": float(valid.float().mean().item()),
            "match_valid_count": float(valid_count),
            "match_posterior_entropy": float(
                ((real_alignment.entropy + fake_alignment.entropy) * mask).sum().item()
                / max(1, 2 * int(mask.sum().item()))),
            "match_cosine": float((cosine * mask).sum().item() / max(1, int(mask.sum().item()))),
        }
        return loss, metrics, real_alignment, fake_alignment, cosine.detach()

    def _g_step(self, batch: dict) -> dict[str, float]:
        if not self.enable_ctc_guided:
            return super()._g_step(batch)
        validate_bidirectional_pairs(batch)
        G, D, E, R, W = (self.models[key] for key in ("G", "D", "E", "R", "W"))
        set_requires_grad([D, R, W], False)
        set_requires_grad([G, E], True)
        self.optimizers["G"].zero_grad(set_to_none=True)
        images = batch["images"].to(self.device, non_blocking=True)
        image_lens = batch["image_lens"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        label_lens = batch["label_lens"].to(self.device, non_blocking=True)
        writers = batch["writers"].to(self.device, non_blocking=True)
        partners = paired_indices(len(images), self.device)
        random_labels, random_label_lens, _ = self._texts(len(images))
        random_lens = random_label_lens * self.char_width
        paired_lens = label_lens * self.char_width
        con_weight = self._supcon_weight(self.global_step + 1)
        match_weight = self._match_weight(self.global_step + 1)
        match_metrics = {"match_valid_ratio": 0.0, "match_valid_count": 0.0,
                         "match_posterior_entropy": 0.0, "match_cosine": 0.0}
        with self._autocast():
            random_z = torch.randn(len(images), int(self.cfg.GenModel.style_dim), device=self.device)
            random_images = G(random_z, random_labels, random_label_lens)
            encoded_style, mu, logvar = self._encode_style(
                images[partners], image_lens[partners], vae=True)
            noise = torch.randn(len(images), self.noise_dim, device=self.device)
            paired_images = G(torch.cat((noise, encoded_style), dim=1), labels, label_lens)
            max_image_width = max(random_images.shape[-1], paired_images.shape[-1])
            all_images = torch.cat((_pad_last(random_images, max_image_width, -1.0),
                                    _pad_last(paired_images, max_image_width, -1.0)), dim=0)
            max_label_width = max(random_labels.shape[-1], labels.shape[-1])
            all_labels = torch.cat((_pad_last(random_labels, max_label_width),
                                    _pad_last(labels, max_label_width)), dim=0)
            all_label_lens = torch.cat((random_label_lens, label_lens), dim=0)
            all_image_lens = torch.cat((random_lens, paired_lens), dim=0)
            adversarial = -D(all_images, all_image_lens, all_label_lens).mean()
            fake_ctc_logits = R(all_images)
            fake_ctc = self.ctc_loss(fake_ctc_logits, all_labels,
                                     all_image_lens // R.len_scale, all_label_lens)
            reconstructed = self._encode_style(random_images, random_lens, vae=False)
            style_reconstruction = (reconstructed - random_z[:, -self.style_dim:].detach()).abs().mean()
            wid_logits = W(paired_images, paired_lens)
            fake_wid = self.writer_loss(wid_logits, writers)
            divergence = kl_loss(mu, logvar)
            grad_adv = torch.autograd.grad(adversarial, all_images, create_graph=True,
                                           retain_graph=True)[0]
            grad_ctc = torch.autograd.grad(fake_ctc, fake_ctc_logits, create_graph=True,
                                           retain_graph=True)[0]
            grad_style = torch.autograd.grad(style_reconstruction, random_images,
                                             create_graph=True, retain_graph=True)[0]
            grad_wid = torch.autograd.grad(fake_wid, wid_logits, create_graph=True,
                                           retain_graph=True)[0]
            adv_std = grad_adv.float().std()
            g_ctc = (adv_std / (grad_ctc.float().std() + 1e-8)).detach().add(1).clamp_max(100)
            g_style = (adv_std / (grad_style.float().std() + 1e-8)).detach().add(1)
            g_wid = (adv_std / (grad_wid.float().std() + 1e-8)).detach().add(1).clamp_max(10)
            contrastive = optional_contrastive_loss(
                con_weight, adversarial,
                lambda: self._supcon_branch(images, image_lens, writers, random_labels,
                                            random_label_lens))
            if match_weight > 0:
                match, match_metrics, _, _, _ = self._match_loss(
                    images, image_lens, paired_images, paired_lens, labels, label_lens)
            else:
                match = adversarial.new_zeros(())
            total = (2 * adversarial + g_ctc * fake_ctc + g_style * style_reconstruction
                     + g_wid * fake_wid + float(self.cfg.training.lambda_kl) * divergence
                     + con_weight * contrastive + match_weight * match)
        self.scaler.scale(total).backward()
        self.scaler.step(self.optimizers["G"])
        self.scaler.update()
        return {"g_total": total.item(), "g_adv": adversarial.item(),
                "ctc_fake": fake_ctc.item(), "style_reconstruction": style_reconstruction.item(),
                "wid_fake": fake_wid.item(), "kl": divergence.item(),
                "supcon": contrastive.item(), "lambda_con": con_weight,
                "ctc_match": match.item(), "lambda_match": match_weight,
                "g_ctc": g_ctc.item(), "g_style": g_style.item(), "g_wid": g_wid.item(),
                **match_metrics}

    @torch.inference_mode()
    def save_training_preview(self, batch: dict) -> Path:
        validate_bidirectional_pairs(batch)
        modes = {name: model.training for name, model in self.models.items()}
        for model in self.models.values():
            model.eval()
        try:
            images = batch["images"].to(self.device)
            image_lens = batch["image_lens"].to(self.device)
            labels = batch["labels"].to(self.device)
            label_lens = batch["label_lens"].to(self.device)
            partners = paired_indices(len(images), self.device)
            generator = torch.Generator(device=self.device).manual_seed(int(self.cfg.seed) + self.global_step)
            reference_mu = self._encode_style(images[partners], image_lens[partners], vae=False)
            noise = torch.randn(len(images), self.noise_dim, device=self.device, generator=generator)
            paired_images = self.models["G"](torch.cat((noise, reference_mu), dim=1), labels, label_lens)
            random_z = torch.randn(len(images), int(self.cfg.GenModel.style_dim),
                                   device=self.device, generator=generator)
            random_images = self.models["G"](random_z, labels, label_lens)
            generated_lens = label_lens * self.char_width
            supcon_images = supcon_lens = None
            supcon_text = None
            if self.enable_supcon:
                shared_labels = labels[:1].expand(len(images), -1).contiguous()
                shared_lens = label_lens[:1].expand(len(images)).contiguous()
                shared_noise = torch.randn(1, self.noise_dim, device=self.device,
                                           generator=generator).expand(len(images), -1)
                supcon_images = self.models["G"](
                    torch.cat((shared_noise, reference_mu), dim=1), shared_labels, shared_lens)
                supcon_lens, supcon_text = shared_lens * self.char_width, batch["texts"][0]
            metadata = {"global_step": self.global_step,
                        **self._auxiliary_state(max(1, self.global_step)), "items": []}
            real_alignment = fake_alignment = None
            cosine = None
            if self.enable_ctc_guided:
                real_alignment = self._alignment(images, image_lens, labels, label_lens)
                fake_alignment = self._alignment(paired_images, generated_lens, labels, label_lens)
                _, match_info, _, _, cosine = self._match_loss(
                    images, image_lens, paired_images, generated_lens, labels, label_lens)
                metadata["alignment_summary"] = match_info
            count = min(int(self.cfg.training.preview_items), len(images))
            for index in range(count):
                item = {"sample_id": batch["word_ids"][index],
                        "writer_id": batch["writer_ids"][index],
                        "style_reference_id": batch["word_ids"][int(partners[index])],
                        "reference_text": batch["texts"][int(partners[index])],
                        "query_text": batch["texts"][index],
                        "ctc_input_length_real": int(image_lens[index] // self.models["R"].len_scale),
                        "ctc_input_length_generated": int(generated_lens[index] // self.models["R"].len_scale),
                        "target_length": int(label_lens[index])}
                if real_alignment is not None and fake_alignment is not None:
                    length = int(label_lens[index])
                    item.update({
                        "alignment_valid": bool(real_alignment.valid[index] and fake_alignment.valid[index]),
                        "real_log_probability": float(real_alignment.log_probability[index]),
                        "generated_log_probability": float(fake_alignment.log_probability[index]),
                        "real_normalized_weights": real_alignment.weights[index, :length].cpu().tolist(),
                        "generated_normalized_weights": fake_alignment.weights[index, :length].cpu().tolist(),
                        "real_peaks": real_alignment.peaks[index, :length].cpu().tolist(),
                        "generated_peaks": fake_alignment.peaks[index, :length].cpu().tolist(),
                        "real_entropy": real_alignment.entropy[index, :length].cpu().tolist(),
                        "generated_entropy": fake_alignment.entropy[index, :length].cpu().tolist(),
                        "cosine_per_character": cosine[index, :length].cpu().tolist(),
                    })
                    if bool(self.cfg.ctc_match.preview_heatmaps):
                        heatmap = self.output / "previews" / "alignments"
                        stem = f"step_{self.global_step:08d}_item_{index:02d}"
                        save_alignment_heatmap(
                            real_alignment.weights[index, :length], batch["texts"][index],
                            heatmap / f"{stem}.png", real_alignment.entropy[index, :length],
                            real_alignment.peaks[index, :length], "REAL CTC ALIGNMENT")
                        save_alignment_heatmap(
                            fake_alignment.weights[index, :length], batch["texts"][index],
                            heatmap / f"{stem}_generated.png",
                            fake_alignment.entropy[index, :length],
                            fake_alignment.peaks[index, :length], "GENERATED CTC ALIGNMENT")
                metadata["items"].append(item)
            if not bool(self.cfg.ctc_match.preview_json):
                preview_metadata = None
            else:
                preview_metadata = metadata
            return save_ctc_preview(
                images[partners], image_lens[partners], images, image_lens,
                paired_images, generated_lens, random_images, generated_lens,
                [batch["texts"][int(i)] for i in partners.cpu()], batch["texts"],
                self.output / "previews" / f"step_{self.global_step:08d}.png",
                int(self.cfg.training.preview_items), batch["writer_ids"],
                supcon_images, supcon_lens, supcon_text, preview_metadata)
        finally:
            for name, model in self.models.items():
                model.train(modes[name])

    def checkpoint_payload(self) -> dict:
        payload = super().checkpoint_payload()
        payload["schema_version"] = 2
        payload["auxiliary_objectives"] = self._auxiliary_state(max(1, self.global_step))
        return payload

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        expected = self._auxiliary_state(max(1, int(checkpoint.get("global_step", 0))))
        saved = checkpoint.get("auxiliary_objectives")
        if saved is None:
            raise ValueError("Checkpoint does not contain CTC-soft auxiliary objective metadata")
        # Effective weights are derived; compare the loss set and immutable schedules/targets.
        for section in ("active_losses", "contrastive", "ctc_match"):
            if saved.get(section) != expected.get(section):
                raise ValueError("Checkpoint active losses or auxiliary schedule does not match this run")
        super().load_checkpoint(path)
