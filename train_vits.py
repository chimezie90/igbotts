#!/usr/bin/env python3
"""
Train VITS for Igbo TTS.

Adapted from vits_repo/train.py with:
  - Igbo text processing (pre-computed integer phoneme sequences)
  - Single-GPU and multi-GPU DDP support
  - Smoke test mode (5k steps, quick validation)
  - Resampling support for non-22050Hz audio

Usage:
    # Single GPU
    python train_vits.py -c igbo_vits/configs/igbo_base.json -m output_vits_igbo

    # Smoke test (5k steps)
    python train_vits.py -c igbo_vits/configs/igbo_base.json -m output_vits_igbo --smoke

    # Multi-GPU (2x A100)
    python train_vits.py -c igbo_vits/configs/igbo_base.json -m output_vits_igbo --n-gpus 2

    # Resume from checkpoint
    python train_vits.py -c igbo_vits/configs/igbo_base.json -m output_vits_igbo --resume
"""

import os
import sys
import json
import argparse
import itertools
import math
import time
import logging

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

# Add vits_repo to path
VITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vits_repo")
sys.path.insert(0, VITS_DIR)

# CRITICAL: Patch VITS text module BEFORE importing data_utils,
# because data_utils does `from text import cleaned_text_to_sequence`
# which copies the function reference at import time.
import igbo_vits.text_processing as igbo_text
import text as vits_text_module
vits_text_module.cleaned_text_to_sequence = igbo_text.cleaned_text_to_sequence
vits_text_module.text_to_sequence = lambda t, c: igbo_text.text_to_sequence(t)

import commons
import utils
from data_utils import TextAudioCollate, DistributedBucketSampler
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch

# Also patch data_utils directly since it already imported the old functions
import data_utils as _data_utils_module
_data_utils_module.cleaned_text_to_sequence = igbo_text.cleaned_text_to_sequence
_data_utils_module.text_to_sequence = lambda t, c: igbo_text.text_to_sequence(t)

# Import symbols for vocab size
from igbo_vits.symbols import symbols

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    parser = argparse.ArgumentParser(description="Train Igbo VITS")
    parser.add_argument("-c", "--config", required=True, help="Config JSON path")
    parser.add_argument("-m", "--model", required=True, help="Model output directory")
    parser.add_argument("--n-gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (5k steps)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required for VITS training"

    os.makedirs(args.model, exist_ok=True)

    # Load and save config
    with open(args.config, "r") as f:
        config = json.load(f)

    # Override for smoke test
    if args.smoke:
        config["train"]["epochs"] = 10000  # will stop at 5k steps
        config["train"]["eval_interval"] = 500
        config["train"]["log_interval"] = 50
        logger.info("SMOKE TEST MODE — will stop at 5000 steps")

    config_path = os.path.join(args.model, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    hps = utils.HParams(**config)
    hps.model_dir = args.model

    n_gpus = args.n_gpus if args.n_gpus > 0 else torch.cuda.device_count()

    if n_gpus > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps, args.smoke, args.resume))
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        run(0, 1, hps, args.smoke, args.resume)


def run(rank, n_gpus, hps, smoke_test=False, resume=False):
    global global_step

    if rank == 0:
        log = utils.get_logger(hps.model_dir)
        log.info(hps)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    if n_gpus > 1:
        dist.init_process_group(backend="nccl", init_method="env://",
                                world_size=n_gpus, rank=rank)

    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    # Dataset
    train_dataset = _make_dataset(hps.data.training_files, hps.data)

    if n_gpus > 1:
        train_sampler = DistributedBucketSampler(
            train_dataset, hps.train.batch_size,
            [32, 300, 400, 500, 600, 700, 800, 900, 1000],
            num_replicas=n_gpus, rank=rank, shuffle=True)
    else:
        train_sampler = DistributedBucketSampler(
            train_dataset, hps.train.batch_size,
            [32, 300, 400, 500, 600, 700, 800, 900, 1000],
            num_replicas=1, rank=0, shuffle=True)

    collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset, num_workers=4, shuffle=False, pin_memory=True,
        collate_fn=collate_fn, batch_sampler=train_sampler)

    if rank == 0:
        eval_dataset = _make_dataset(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset, num_workers=2, shuffle=False,
            batch_size=hps.train.batch_size, pin_memory=True,
            drop_last=False, collate_fn=collate_fn)

    # Model
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

    if n_gpus > 1:
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])

    optim_g = torch.optim.AdamW(
        net_g.parameters(), hps.train.learning_rate,
        betas=hps.train.betas, eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(), hps.train.learning_rate,
        betas=hps.train.betas, eps=hps.train.eps)

    # Resume
    epoch_start = 0
    if resume:
        try:
            _, _, _, epoch_start = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
            _, _, _, _ = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
            global_step = (epoch_start + 1) * len(train_loader)
            if rank == 0:
                logger.info(f"Resumed from epoch {epoch_start}, step {global_step}")
        except Exception as e:
            if rank == 0:
                logger.warning(f"No checkpoint to resume from: {e}")

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_start - 1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_start - 1)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    if rank == 0:
        total_params_g = sum(p.numel() for p in net_g.parameters())
        total_params_d = sum(p.numel() for p in net_d.parameters())
        logger.info(f"Generator: {total_params_g:,} params ({total_params_g*4/1e6:.1f} MB)")
        logger.info(f"Discriminator: {total_params_d:,} params ({total_params_d*4/1e6:.1f} MB)")
        logger.info(f"Vocab size: {len(symbols)}")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Batch size: {hps.train.batch_size} x {n_gpus} GPU(s)")

    # Training loop
    for epoch in range(epoch_start, hps.train.epochs):
        if n_gpus > 1:
            train_sampler.set_epoch(epoch)

        stopped = train_and_evaluate(
            rank, epoch, hps, [net_g, net_d], [optim_g, optim_d],
            [scheduler_g, scheduler_d], scaler,
            [train_loader, eval_loader if rank == 0 else None],
            [writer if rank == 0 else None, writer_eval if rank == 0 else None],
            smoke_test)

        scheduler_g.step()
        scheduler_d.step()

        if stopped:
            break

    if rank == 0:
        logger.info(f"Training complete at step {global_step}")

    if n_gpus > 1:
        dist.destroy_process_group()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler,
                       loaders, writers, smoke_test=False):
    global global_step
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    net_g.train()
    net_d.train()

    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        spec = spec.cuda(rank, non_blocking=True)
        spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)

        # Forward generator
        with autocast(enabled=hps.train.fp16_run):
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
                (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                    x, x_lengths, spec, spec_lengths)

            mel = spec_to_mel_torch(
                spec, hps.data.filter_length, hps.data.n_mel_channels,
                hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1), hps.data.filter_length,
                hps.data.n_mel_channels, hps.data.sampling_rate,
                hps.data.hop_length, hps.data.win_length,
                hps.data.mel_fmin, hps.data.mel_fmax)

            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size)

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        torch.nn.utils.clip_grad_norm_(net_d.parameters(), max_norm=1.0)
        scaler.step(optim_d)

        # Generator
        with autocast(enabled=hps.train.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_hat_mel, y_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=1.0)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                logger.info(
                    f"Step {global_step} | "
                    f"loss_g={loss_gen_all.item():.3f} "
                    f"mel={loss_mel.item():.3f} "
                    f"kl={loss_kl.item():.3f} "
                    f"dur={loss_dur.item():.3f} "
                    f"disc={loss_disc_all.item():.3f} "
                    f"grad={grad_norm_g:.2f} "
                    f"lr={lr:.6f}"
                )
                writer.add_scalar("train/loss_gen_all", loss_gen_all.item(), global_step)
                writer.add_scalar("train/loss_mel", loss_mel.item(), global_step)
                writer.add_scalar("train/loss_kl", loss_kl.item(), global_step)
                writer.add_scalar("train/loss_dur", loss_dur.item(), global_step)
                writer.add_scalar("train/loss_disc", loss_disc_all.item(), global_step)
                writer.add_scalar("train/grad_norm_g", grad_norm_g, global_step)
                writer.add_scalar("train/lr", lr, global_step)

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval, global_step)

                # Save checkpoint
                utils.save_checkpoint(
                    net_g, optim_g, hps.train.learning_rate, epoch,
                    os.path.join(hps.model_dir, f"G_{global_step}.pth"))
                utils.save_checkpoint(
                    net_d, optim_d, hps.train.learning_rate, epoch,
                    os.path.join(hps.model_dir, f"D_{global_step}.pth"))

                # Keep only last 3 checkpoints
                _cleanup_checkpoints(hps.model_dir, keep=3)

            # Smoke test: stop at 5k steps
            if smoke_test and global_step >= 5000:
                logger.info(f"SMOKE TEST COMPLETE at step {global_step}")
                logger.info(f"  Gen loss: {loss_gen_all.item():.3f}")
                logger.info(f"  Mel loss: {loss_mel.item():.3f}")
                logger.info(f"  Check {hps.model_dir}/eval for TensorBoard alignment plots")
                return True  # signal to stop

        global_step += 1

    return False


def evaluate(hps, generator, eval_loader, writer_eval, step):
    generator.eval()
    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(eval_loader):
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)

            # Get the raw model if wrapped in DDP
            model = generator.module if hasattr(generator, "module") else generator

            # Inference (no teacher forcing)
            y_hat, attn, mask, *_ = model.infer(x, x_lengths, max_len=1000)
            y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

            # Mel for logging
            mel = spec_to_mel_torch(
                spec, hps.data.filter_length, hps.data.n_mel_channels,
                hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)

            if y_hat.shape[-1] > 0:
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(), hps.data.filter_length,
                    hps.data.n_mel_channels, hps.data.sampling_rate,
                    hps.data.hop_length, hps.data.win_length,
                    hps.data.mel_fmin, hps.data.mel_fmax)

            if batch_idx == 0:
                # Log audio and spectrograms
                if y_hat.shape[-1] > 0:
                    writer_eval.add_audio(
                        f"gen/audio_{batch_idx}",
                        y_hat[0, :, :y_hat_lengths[0]],
                        step, hps.data.sampling_rate)
                writer_eval.add_audio(
                    f"gt/audio_{batch_idx}",
                    y[0, :, :y_lengths[0]],
                    step, hps.data.sampling_rate)

                # Log alignment attention
                if attn is not None and attn.shape[-1] > 0 and attn.shape[-2] > 0:
                    writer_eval.add_image(
                        f"alignment/{batch_idx}",
                        utils.plot_alignment_to_numpy(
                            attn[0, 0].cpu().numpy()),
                        step, dataformats="HWC")

            if batch_idx >= 3:
                break

    generator.train()


def _cleanup_checkpoints(model_dir, keep=3):
    """Keep only the N most recent checkpoints."""
    import glob
    for prefix in ["G_", "D_"]:
        ckpts = sorted(glob.glob(os.path.join(model_dir, f"{prefix}*.pth")))
        for ckpt in ckpts[:-keep]:
            os.remove(ckpt)


def _make_dataset(filelist_path, data_config):
    """Create TextAudioLoader with our Igbo text processing patched in."""
    from data_utils import TextAudioLoader
    return TextAudioLoader(filelist_path, data_config)


if __name__ == "__main__":
    main()
