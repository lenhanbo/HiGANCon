from __future__ import annotations

import json
import random
import time
from contextlib import nullcontext
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CTCLoss, CrossEntropyLoss
from torch.utils.data import DataLoader

from lib.alphabet import Alphabets, get_lexicon, strLabelConverter
from networks.BigGAN_networks import Discriminator, Generator
from networks.module import Recognizer, StyleEncoder, WriterIdentifier
from networks.utils import get_scheduler, rand_clip, set_requires_grad

from .checkpoint import atomic_torch_save, file_hash, git_commit, restore_rng_state, rng_state, state_hash
from .data import StatefulWriterPairBatchSampler, collate_words, load_experiment_dataset
from .losses import (contrastive_weight_at_step, kl_loss, optional_contrastive_loss,
                     supervised_contrastive_loss)
from .preview import save_preview


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class HiGANTrainer:
    def __init__(self, cfg, config_path: str, dataset_root: str, manifest_root: str,
                 output: str, resume: str | None = None, max_hours: float | None = None,
                 max_steps: int | None = None,
                 contrastive_start_step: int | None = None,
                 contrastive_ramp_steps: int | None = None,
                 dataset_format: str = "iam_raw",
                 preview_every_steps: int | None = None):
        self.cfg = cfg
        self.config_path = Path(config_path).resolve()
        self.manifest_root = Path(manifest_root).resolve()
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.dataset_format = dataset_format
        (self.output / "checkpoints").mkdir(exist_ok=True)
        self.max_hours = max_hours
        self.max_steps = int(max_steps) if max_steps is not None and int(max_steps) > 0 else None
        self.preview_every_steps = int(
            cfg.training.preview_every_steps
            if preview_every_steps is None else preview_every_steps)
        if self.preview_every_steps < 0:
            raise ValueError("preview_every_steps must be >= 0")
        self.contrastive_start_step = int(
            contrastive_start_step if contrastive_start_step is not None
            else cfg.contrastive.start_step)
        self.contrastive_ramp_steps = int(
            contrastive_ramp_steps if contrastive_ramp_steps is not None
            else cfg.contrastive.ramp_steps)
        if self.contrastive_start_step < 1 or self.contrastive_ramp_steps < 0:
            raise ValueError("contrastive start_step must be >=1 and ramp_steps must be >=0")
        self.started_at = time.monotonic()
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        seed_everything(int(cfg.seed))

        metadata_path = self.manifest_root / "split_metadata.json"
        self.split_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_format = self.split_metadata.get("dataset_format", "iam_raw")
        if metadata_format != self.dataset_format:
            raise ValueError(
                f"Dataset format mismatch: CLI={self.dataset_format}, metadata={metadata_format}")
        self.split_hash = self.split_metadata["split_hash"]
        self.config_hash = file_hash(self.config_path)
        self.dataset = load_experiment_dataset(
            self.dataset_format, dataset_root, self.manifest_root, "train")
        writers_per_batch = int(cfg.training.writers_per_batch)
        if int(cfg.training.batch_size) != 2 * writers_per_batch:
            raise ValueError("training.batch_size must equal 2 * training.writers_per_batch")
        self.sampler = StatefulWriterPairBatchSampler(self.dataset.records, writers_per_batch, int(cfg.seed))
        self.loader_generator = torch.Generator().manual_seed(int(cfg.seed) + 17)
        self.loader = DataLoader(
            self.dataset, batch_sampler=self.sampler, collate_fn=collate_words,
            num_workers=int(cfg.training.num_workers), pin_memory=self.device.type == "cuda",
            persistent_workers=int(cfg.training.num_workers) > 0, generator=self.loader_generator)

        cfg.WidModel.n_writer = len(self.dataset.writer_to_index)
        self.models = {
            "G": Generator(**cfg.GenModel).to(self.device),
            "D": Discriminator(**cfg.DiscModel).to(self.device),
            "E": StyleEncoder(**cfg.EncModel).to(self.device),
            "R": Recognizer(**cfg.OcrModel).to(self.device),
            "W": WriterIdentifier(**cfg.WidModel).to(self.device),
        }
        self.initial_state_hash = state_hash(self.models)
        self.optimizers = {
            "G": torch.optim.Adam(chain(self.models["G"].parameters(), self.models["E"].parameters()),
                                  lr=cfg.training.lr, betas=(cfg.training.adam_b1, cfg.training.adam_b2)),
            "D": torch.optim.Adam(chain(self.models["D"].parameters(), self.models["R"].parameters(),
                                       self.models["W"].parameters()), lr=cfg.training.lr,
                                  betas=(cfg.training.adam_b1, cfg.training.adam_b2)),
        }
        self.schedulers = {name: get_scheduler(optimizer, cfg.training)
                           for name, optimizer in self.optimizers.items()}
        amp_enabled = bool(cfg.training.amp) and self.device.type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        self.amp_enabled = amp_enabled
        self.converter = strLabelConverter("all")
        self.ctc_loss = CTCLoss(zero_infinity=True, reduction="mean")
        self.writer_loss = CrossEntropyLoss()
        self.style_dim = int(cfg.EncModel.style_dim)
        self.noise_dim = int(cfg.GenModel.style_dim) - self.style_dim
        self.char_width = int(cfg.char_width)
        self.lexicon = get_lexicon(cfg.training.lexicon, Alphabets["all"],
                                   max_length=int(cfg.training.max_word_len))
        if not self.lexicon:
            raise ValueError("The configured lexicon contains no usable words")
        self.epoch = 0
        self.global_step = 0
        self.metrics_path = self.output / "metrics.jsonl"
        self.log_window_sums: dict[str, float] = {}
        self.log_window_counts: dict[str, int] = {}
        if resume:
            self.load_checkpoint(resume)
        self._write_run_metadata()

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16) if self.amp_enabled else nullcontext()

    def _write_run_metadata(self) -> None:
        metadata = {
            "git_commit": git_commit(), "config_hash": self.config_hash,
            "split_hash": self.split_hash, "initial_state_hash": self.initial_state_hash,
            "device": str(self.device), "train_images": len(self.dataset),
            "train_writers": len(self.dataset.writer_to_index),
            "dataset_format": self.dataset_format,
            "contrastive_start_step": self.contrastive_start_step,
            "contrastive_ramp_steps": self.contrastive_ramp_steps,
            "max_steps_this_session": self.max_steps,
        }
        (self.output / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _texts(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        words = [random.choice(self.lexicon) for _ in range(batch_size)]
        labels, lengths = self.converter.encode(words)
        return labels.to(self.device), lengths.to(self.device), words

    def _encode_style(self, images, lengths, vae: bool = False):
        return self.models["E"](images, lengths, self.models["W"].cnn_backbone, vae_mode=vae)

    def _sample_fakes(self, real_images, real_lengths, labels, label_lengths):
        batch_size = real_images.shape[0]
        random_z = torch.randn(batch_size, int(self.cfg.GenModel.style_dim), device=self.device)
        random_images = self.models["G"](random_z, labels, label_lengths)
        encoded_style, mu, logvar = self._encode_style(real_images, real_lengths, vae=True)
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        reference_images = self.models["G"](torch.cat((noise, encoded_style), dim=1), labels, label_lengths)
        return random_z, random_images, reference_images, mu, logvar

    def _d_step(self, batch: dict) -> dict[str, float]:
        G, D, E, R, W = (self.models[key] for key in ("G", "D", "E", "R", "W"))
        set_requires_grad([G, E], False)
        set_requires_grad([D, R, W], True)
        self.optimizers["D"].zero_grad(set_to_none=True)
        images = batch["images"].to(self.device, non_blocking=True)
        image_lens = batch["image_lens"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        label_lens = batch["label_lens"].to(self.device, non_blocking=True)
        writers = batch["writers"].to(self.device, non_blocking=True)
        with self._autocast():
            real_ctc_loss = self.ctc_loss(R(images), labels, image_lens // 8, label_lens)
            clipped, clipped_lens = rand_clip(images, image_lens)
            real_wid_loss = self.writer_loss(W(clipped, clipped_lens), writers)
            fake_labels, fake_label_lens, _ = self._texts(images.shape[0])
            with torch.no_grad():
                _, random_images, reference_images, _, _ = self._sample_fakes(
                    images, image_lens, fake_labels, fake_label_lens)
            fake_images = torch.cat((random_images, reference_images), dim=0)
            fake_lens = fake_label_lens.repeat(2) * self.char_width
            fake_text_lens = fake_label_lens.repeat(2)
            fake_disc_loss = F.relu(1 + D(fake_images, fake_lens, fake_text_lens)).mean()
            real_disc_loss = F.relu(1 - D(images, image_lens, label_lens)).mean()
            total = real_ctc_loss + real_wid_loss + fake_disc_loss + real_disc_loss
        self.scaler.scale(total).backward()
        self.scaler.step(self.optimizers["D"])
        self.scaler.update()
        return {"d_total": total.item(), "ctc_real": real_ctc_loss.item(),
                "wid_real": real_wid_loss.item(), "d_real": real_disc_loss.item(),
                "d_fake": fake_disc_loss.item()}

    def _supcon_branch(self, images, image_lens, writers, labels, label_lens) -> torch.Tensor:
        ref_mu = self._encode_style(images, image_lens, vae=False)
        shared_labels = labels[:1].expand(images.shape[0], -1).contiguous()
        shared_label_lens = label_lens[:1].expand(images.shape[0]).contiguous()
        shared_noise = torch.randn(1, self.noise_dim, device=self.device).expand(images.shape[0], -1)
        generated = self.models["G"](torch.cat((shared_noise, ref_mu), dim=1),
                                     shared_labels, shared_label_lens)
        generated_mu = self._encode_style(generated, shared_label_lens * self.char_width, vae=False)
        return supervised_contrastive_loss(generated_mu, writers, float(self.cfg.contrastive.temperature))

    def _g_step(self, batch: dict) -> dict[str, float]:
        G, D, E, R, W = (self.models[key] for key in ("G", "D", "E", "R", "W"))
        set_requires_grad([D, R, W], False)
        set_requires_grad([G, E], True)
        self.optimizers["G"].zero_grad(set_to_none=True)
        images = batch["images"].to(self.device, non_blocking=True)
        image_lens = batch["image_lens"].to(self.device, non_blocking=True)
        writers = batch["writers"].to(self.device, non_blocking=True)
        fake_labels, fake_label_lens, _ = self._texts(images.shape[0])
        fake_image_lens = fake_label_lens * self.char_width
        con_weight = contrastive_weight_at_step(
            self.global_step + 1, float(self.cfg.contrastive.lambda_con),
            self.contrastive_start_step, self.contrastive_ramp_steps)
        with self._autocast():
            random_z, random_images, reference_images, mu, logvar = self._sample_fakes(
                images, image_lens, fake_labels, fake_label_lens)
            all_images = torch.cat((random_images, reference_images), dim=0)
            all_labels = fake_labels.repeat(2, 1)
            all_label_lens = fake_label_lens.repeat(2)
            all_image_lens = all_label_lens * self.char_width
            adversarial = -D(all_images, all_image_lens, all_label_lens).mean()
            fake_ctc_logits = R(all_images)
            fake_ctc = self.ctc_loss(fake_ctc_logits, all_labels, all_image_lens // 8, all_label_lens)
            reconstructed = self._encode_style(random_images, fake_image_lens, vae=False)
            style_reconstruction = (reconstructed - random_z[:, -self.style_dim:].detach()).abs().mean()
            wid_logits = W(reference_images, fake_image_lens)
            fake_wid = self.writer_loss(wid_logits, writers)
            divergence = kl_loss(mu, logvar)

            grad_adv = torch.autograd.grad(adversarial, all_images, create_graph=True, retain_graph=True)[0]
            grad_ctc = torch.autograd.grad(fake_ctc, fake_ctc_logits, create_graph=True, retain_graph=True)[0]
            grad_style = torch.autograd.grad(style_reconstruction, random_images,
                                             create_graph=True, retain_graph=True)[0]
            grad_wid = torch.autograd.grad(fake_wid, wid_logits, create_graph=True, retain_graph=True)[0]
            adv_std = grad_adv.float().std()
            g_ctc = (adv_std / (grad_ctc.float().std() + 1e-8)).detach().add(1).clamp_max(100)
            g_style = (adv_std / (grad_style.float().std() + 1e-8)).detach().add(1)
            g_wid = (adv_std / (grad_wid.float().std() + 1e-8)).detach().add(1).clamp_max(10)
            contrastive = optional_contrastive_loss(
                con_weight, adversarial,
                lambda: self._supcon_branch(images, image_lens, writers, fake_labels, fake_label_lens))
            total = (2 * adversarial + g_ctc * fake_ctc + g_style * style_reconstruction
                     + g_wid * fake_wid + float(self.cfg.training.lambda_kl) * divergence
                     + con_weight * contrastive)
        self.scaler.scale(total).backward()
        self.scaler.step(self.optimizers["G"])
        self.scaler.update()
        return {"g_total": total.item(), "g_adv": adversarial.item(), "ctc_fake": fake_ctc.item(),
                "style_reconstruction": style_reconstruction.item(), "wid_fake": fake_wid.item(),
                "kl": divergence.item(), "supcon": contrastive.item(), "lambda_con": con_weight,
                "g_ctc": g_ctc.item(), "g_style": g_style.item(), "g_wid": g_wid.item()}

    def train_batch(self, batch: dict) -> dict[str, float]:
        for model in self.models.values():
            model.train()
        metrics = self._d_step(batch)
        if self.global_step % int(self.cfg.training.num_critic_train) == 0:
            metrics.update(self._g_step(batch))
        return metrics

    def _update_log_window(self, metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            self.log_window_sums[name] = self.log_window_sums.get(name, 0.0) + float(value)
            self.log_window_counts[name] = self.log_window_counts.get(name, 0) + 1

    def _pop_log_window(self) -> dict[str, float]:
        averages = {name: self.log_window_sums[name] / self.log_window_counts[name]
                    for name in sorted(self.log_window_sums)}
        self.log_window_sums.clear()
        self.log_window_counts.clear()
        return averages

    @torch.inference_mode()
    def save_training_preview(self, batch: dict) -> Path:
        modes = {name: model.training for name, model in self.models.items()}
        for model in self.models.values():
            model.eval()
        try:
            images = batch["images"].to(self.device)
            image_lens = batch["image_lens"].to(self.device)
            labels = batch["labels"].to(self.device)
            label_lens = batch["label_lens"].to(self.device)
            generator = torch.Generator(device=self.device)
            preview_seed = int(self.cfg.seed) + self.global_step
            generator.manual_seed(preview_seed)
            random_z_single = torch.randn(
                1, int(self.cfg.GenModel.style_dim), device=self.device, generator=generator)
            random_z = random_z_single.expand(len(images), -1)
            reference_mu = self._encode_style(images, image_lens, vae=False)
            noise = torch.randn(
                len(images), self.noise_dim, device=self.device, generator=generator)
            random_images = self.models["G"](random_z, labels, label_lens)
            reference_images = self.models["G"](
                torch.cat((noise, reference_mu), dim=1), labels, label_lens)
            con_weight = contrastive_weight_at_step(
                max(1, self.global_step), float(self.cfg.contrastive.lambda_con),
                self.contrastive_start_step, self.contrastive_ramp_steps)
            contrastive_images = None
            contrastive_lengths = None
            contrastive_text = None
            if float(self.cfg.contrastive.lambda_con) > 0:
                shared_labels = labels[:1].expand(len(images), -1).contiguous()
                shared_label_lens = label_lens[:1].expand(len(images)).contiguous()
                shared_noise = torch.randn(
                    1, self.noise_dim, device=self.device, generator=generator).expand(len(images), -1)
                contrastive_images = self.models["G"](
                    torch.cat((shared_noise, reference_mu), dim=1),
                    shared_labels, shared_label_lens)
                contrastive_lengths = shared_label_lens * self.char_width
                contrastive_text = batch["texts"][0]
            metadata = {
                "global_step": self.global_step,
                "random_synthetic_style": {
                    "definition": "One z sampled from N(0,I); it has no real writer identity and is shared across preview rows.",
                    "seed": preview_seed,
                    "latent_dim": int(self.cfg.GenModel.style_dim),
                    "auxiliary_noise_dims": self.noise_dim,
                    "style_code_dims": self.style_dim,
                    "z": random_z_single[0].detach().float().cpu().tolist(),
                },
                "reference_guided_style": {
                    "definition": "Per-row deterministic E_mu(real) plus per-row auxiliary noise.",
                },
                "supcon_view": {
                    "enabled_in_objective": con_weight > 0,
                    "lambda_con": con_weight,
                    "start_step": self.contrastive_start_step,
                    "ramp_steps": self.contrastive_ramp_steps,
                    "definition": "E_mu(real) with one shared auxiliary noise and one shared text for every row; adjacent rows are same-writer pairs.",
                    "shared_text": contrastive_text,
                },
            }
            return save_preview(
                images, image_lens, random_images, reference_images,
                label_lens * self.char_width, batch["texts"],
                self.output / "previews" / f"step_{self.global_step:08d}.png",
                int(self.cfg.training.preview_items), writer_ids=batch["writer_ids"],
                random_title=f"RANDOM z~N(0,I) seed={preview_seed}",
                contrastive_style=contrastive_images,
                contrastive_lengths=contrastive_lengths,
                contrastive_text=contrastive_text,
                metadata=metadata)
        finally:
            for name, model in self.models.items():
                model.train(modes[name])

    def checkpoint_payload(self) -> dict:
        return {
            "schema_version": 1, "models": {n: m.state_dict() for n, m in self.models.items()},
            "optimizers": {n: o.state_dict() for n, o in self.optimizers.items()},
            "schedulers": {n: s.state_dict() for n, s in self.schedulers.items()},
            "scaler": self.scaler.state_dict(), "epoch": self.epoch, "global_step": self.global_step,
            "sampler": self.sampler.state_dict(), "loader_generator": self.loader_generator.get_state(),
            "rng": rng_state(), "git_commit": git_commit(),
            "config_hash": self.config_hash, "split_hash": self.split_hash,
            "dataset_format": self.dataset_format,
            "initial_state_hash": self.initial_state_hash,
            "contrastive_schedule": {
                "start_step": self.contrastive_start_step,
                "ramp_steps": self.contrastive_ramp_steps,
            },
        }

    def save_checkpoint(self, name: str = "last.pt") -> Path:
        payload = self.checkpoint_payload()
        path = self.output / "checkpoints" / name
        atomic_torch_save(payload, path)
        if name != "last.pt":
            atomic_torch_save(payload, self.output / "checkpoints" / "last.pt")
        return path

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint["config_hash"] != self.config_hash:
            raise ValueError("Checkpoint config hash does not match this run")
        if checkpoint["split_hash"] != self.split_hash:
            raise ValueError("Checkpoint split hash does not match manifests")
        if checkpoint.get("dataset_format", "iam_raw") != self.dataset_format:
            raise ValueError("Checkpoint dataset format does not match this run")
        if checkpoint["initial_state_hash"] != self.initial_state_hash:
            raise ValueError("Checkpoint was created from a different initialization")
        expected_schedule = {"start_step": self.contrastive_start_step,
                             "ramp_steps": self.contrastive_ramp_steps}
        if checkpoint.get("contrastive_schedule") != expected_schedule:
            raise ValueError("Checkpoint contrastive step schedule does not match this run")
        for name, state in checkpoint["models"].items(): self.models[name].load_state_dict(state)
        for name, state in checkpoint["optimizers"].items(): self.optimizers[name].load_state_dict(state)
        for name, state in checkpoint["schedulers"].items(): self.schedulers[name].load_state_dict(state)
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.epoch, self.global_step = int(checkpoint["epoch"]), int(checkpoint["global_step"])
        self.sampler.load_state_dict(checkpoint["sampler"])
        self.loader_generator.set_state(checkpoint["loader_generator"])
        restore_rng_state(checkpoint["rng"])

    def _time_expired(self) -> bool:
        return self.max_hours is not None and (time.monotonic() - self.started_at) >= self.max_hours * 3600

    def _step_limit_reached(self) -> bool:
        return self.max_steps is not None and self.global_step >= self.max_steps

    def train(self, smoke_test: bool = False) -> None:
        total_epochs = int(self.cfg.training.epochs)
        if self._step_limit_reached():
            self.save_checkpoint()
            print(f"Already at --max-steps={self.max_steps}; saved last.pt.", flush=True)
            return
        while self.epoch < total_epochs:
            self.sampler.set_position(self.epoch, self.sampler.start_batch)
            for batch_index, batch in enumerate(self.loader, start=self.sampler.start_batch):
                if self._time_expired():
                    self.save_checkpoint()
                    print("Reached --max-hours; saved a resumable checkpoint.", flush=True)
                    return
                metrics = self.train_batch(batch)
                self._update_log_window(metrics)
                self.global_step += 1
                self.sampler.set_position(self.epoch, batch_index + 1)
                record = {"epoch": self.epoch + 1, "batch": batch_index + 1,
                          "global_step": self.global_step, **metrics}
                with self.metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                if self.global_step % int(self.cfg.training.print_every_steps) == 0:
                    log_record = {
                        "epoch": self.epoch + 1,
                        "batch": batch_index + 1,
                        "global_step": self.global_step,
                        "window_steps": int(self.cfg.training.print_every_steps),
                        **self._pop_log_window(),
                    }
                    print(json.dumps(log_record, sort_keys=True), flush=True)
                if self.global_step % int(self.cfg.training.save_every_steps) == 0:
                    self.save_checkpoint(f"step_{self.global_step:08d}.pt")
                preview_every = self.preview_every_steps
                if preview_every > 0 and self.global_step % preview_every == 0:
                    preview_path = self.save_training_preview(batch)
                    print(f"Saved preview: {preview_path}", flush=True)
                if smoke_test:
                    preview_path = self.save_training_preview(batch)
                    print(f"Saved smoke preview: {preview_path}", flush=True)
                    print("Smoke test passed: one full D-side and G/E-side update.", flush=True)
                    return
                if self._step_limit_reached():
                    self.save_checkpoint()
                    print(f"Reached --max-steps={self.max_steps}; saved last.pt.", flush=True)
                    return
            self.epoch += 1
            self.sampler.set_position(self.epoch, 0)
            for scheduler in self.schedulers.values(): scheduler.step()
            self.save_checkpoint(f"epoch_{self.epoch:03d}.pt")
