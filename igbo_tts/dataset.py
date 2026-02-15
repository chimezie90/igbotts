#!/usr/bin/env python3
"""
WAXAL Igbo Dataset Loader

Adapted from kokoro_training/data/ljspeech_dataset.py for Igbo TTS.
Loads audio, Igbo phoneme sequences, and MFA duration alignments.
"""

import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
import logging
import random
from tqdm import tqdm

try:
    import soundfile
except ImportError:
    raise ImportError("soundfile is required. Install with: pip install soundfile")

logger = logging.getLogger(__name__)


class IgboDataset(Dataset):
    """
    Dataset for WAXAL Igbo corpus with MFA alignments.

    Expects the directory layout produced by setup_data.py:
        data_dir/
            metadata.csv           # id|text|normalized
            wavs/                  # individual WAV files
            TextGrid/wavs/         # MFA TextGrid alignments (optional subdir)
    """

    def __init__(self, data_dir: str, config):
        from .g2p import IgboG2P

        self.data_dir = Path(data_dir)
        self.config = config
        self.phoneme_processor = IgboG2P()

        # TextGrid files from MFA — check both common layouts
        self.alignment_dir = self.data_dir / "TextGrid" / "wavs"
        if not self.alignment_dir.exists():
            self.alignment_dir = self.data_dir / "TextGrid"
        self.has_alignments = self.alignment_dir.exists() and any(
            self.alignment_dir.rglob("*.TextGrid")
        )

        if self.has_alignments:
            logger.info(f"Found MFA alignments at: {self.alignment_dir}")
        else:
            logger.warning(
                f"No MFA alignments found at {self.data_dir / 'TextGrid'}. "
                "Durations will be estimated (uniform). For better quality, "
                "run: python -m igbo_tts.setup_data --align"
            )

        # Validate mel params
        if self.config.win_length > self.config.n_fft:
            raise ValueError(
                f"win_length ({self.config.win_length}) > n_fft ({self.config.n_fft})"
            )

        # Pre-create mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            power=2.0,
            normalized=False,
            window_fn=torch.hann_window,
        )

        self.samples = self._load_samples()
        self.skipped_samples = 0
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")

    def _load_samples(self) -> List[Dict]:
        """Load samples from metadata.csv (LJSpeech format)."""
        samples = []
        metadata_file = self.data_dir / "metadata.csv"

        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file}\n"
                f"Run: python -m igbo_tts.setup_data -o {self.data_dir}"
            )

        with open(metadata_file, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=total_lines, desc="Loading metadata"):
                parts = line.strip().split("|")
                if len(parts) < 2:
                    continue

                audio_file_stem = parts[0]
                text = parts[2] if len(parts) >= 3 else parts[1]
                audio_path = self.data_dir / "wavs" / f"{audio_file_stem}.wav"

                if not audio_path.exists():
                    continue

                sample = {
                    "audio_path": str(audio_path),
                    "text": text,
                    "audio_file": audio_file_stem,
                    "alignment_path": None,
                }

                if self.has_alignments:
                    tg_path = self.alignment_dir / f"{audio_file_stem}.TextGrid"
                    if tg_path.exists():
                        sample["alignment_path"] = str(tg_path)

                samples.append(sample)

        return samples

    def _load_mfa_durations(
        self,
        alignment_path: Optional[str],
        phoneme_count: int,
        mel_frame_count: int,
    ) -> torch.Tensor:
        """Load phoneme durations from MFA TextGrid file."""
        if alignment_path is None:
            raise ValueError("MFA alignment path is None")

        if not Path(alignment_path).exists():
            raise FileNotFoundError(f"MFA alignment not found: {alignment_path}")

        import textgrid

        tg = textgrid.TextGrid.fromFile(alignment_path)

        phones_tier = None
        for tier in tg.tiers:
            if tier.name.lower() in ("phones", "phone", "phonemes", "phoneme"):
                phones_tier = tier
                break

        if phones_tier is None:
            raise ValueError(
                f"No phones tier in {alignment_path}. "
                f"Available: {[t.name for t in tg.tiers]}"
            )

        durations_sec = []
        for interval in phones_tier:
            if (
                interval.mark
                and interval.mark.strip()
                and interval.mark.lower() not in ("sil", "sp", "")
            ):
                durations_sec.append(interval.maxTime - interval.minTime)

        durations_frames = [
            max(1, int(d * self.config.sample_rate / self.config.hop_length))
            for d in durations_sec
        ]

        # Validate alignment — allow generous mismatch since MFA and G2P
        # may segment phones differently (e.g. MFA merges geminate consonants)
        if phoneme_count > 0:
            mismatch = abs(len(durations_frames) - phoneme_count) / phoneme_count
            if mismatch > 0.70:
                raise ValueError(
                    f"Phoneme count mismatch in {alignment_path}: "
                    f"MFA={len(durations_frames)}, G2P={phoneme_count} "
                    f"({mismatch*100:.0f}%)"
                )

        # Adjust to match phoneme count
        if len(durations_frames) != phoneme_count:
            durations_frames = self._adjust_duration_count(
                durations_frames, phoneme_count, mel_frame_count
            )

        return torch.tensor(durations_frames, dtype=torch.long)

    def _adjust_duration_count(
        self, durations: List[int], target: int, total_frames: int
    ) -> List[int]:
        """Adjust duration list length by splitting/merging."""
        while len(durations) < target:
            max_idx = durations.index(max(durations))
            val = durations[max_idx]
            durations[max_idx] = val // 2
            durations.insert(max_idx + 1, val - val // 2)

        while len(durations) > target:
            min_sum = float("inf")
            min_idx = 0
            for i in range(len(durations) - 1):
                s = durations[i] + durations[i + 1]
                if s < min_sum:
                    min_sum = s
                    min_idx = i
            durations[min_idx] += durations[min_idx + 1]
            durations.pop(min_idx + 1)

        return durations

    def _estimate_durations(self, phoneme_count: int, mel_frames: int) -> torch.Tensor:
        """Estimate uniform durations when MFA alignments are unavailable."""
        if phoneme_count == 0:
            return torch.ones(1, dtype=torch.long)
        base = mel_frames // phoneme_count
        remainder = mel_frames % phoneme_count
        durations = [base] * phoneme_count
        for i in range(remainder):
            durations[i] += 1
        return torch.tensor(durations, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        try:
            import soundfile as sf

            audio_data, sr = sf.read(sample["audio_path"])
            waveform = torch.from_numpy(audio_data).float().unsqueeze(0)

            if sr != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                waveform = resampler(waveform)

            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Minimum length for STFT
            if waveform.shape[1] < self.config.win_length:
                waveform = torch.nn.functional.pad(
                    waveform, (0, self.config.win_length - waveform.shape[1])
                )

            # Mel spectrogram
            mel_spec = self.mel_transform(waveform).squeeze(0).T  # [time, n_mels]
            mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
            mel_spec = torch.clamp(mel_spec, min=-11.5, max=0.0)

            # Clip to max sequence length
            if mel_spec.shape[0] > self.config.max_seq_length:
                mel_spec = mel_spec[: self.config.max_seq_length, :]

            # Phonemes
            phoneme_indices = self.phoneme_processor.text_to_indices(sample["text"])
            phoneme_tensor = torch.tensor(phoneme_indices, dtype=torch.long)

            # Durations — from MFA alignment or estimated
            if sample.get("alignment_path") and Path(sample["alignment_path"]).exists():
                phoneme_durations = self._load_mfa_durations(
                    sample["alignment_path"],
                    len(phoneme_indices),
                    mel_spec.shape[0],
                )
            else:
                phoneme_durations = self._estimate_durations(
                    len(phoneme_indices), mel_spec.shape[0]
                )

            # Stop token targets
            stop_targets = torch.zeros(mel_spec.shape[0], dtype=torch.float32)
            if mel_spec.shape[0] > 0:
                stop_targets[-1] = 1.0

            return {
                "phoneme_indices": phoneme_tensor,
                "mel_spec": mel_spec,
                "phoneme_durations": phoneme_durations,
                "stop_token_targets": stop_targets,
                "audio_file": sample["audio_file"],
                "text": sample["text"],
            }

        except Exception as e:
            self.skipped_samples += 1
            logger.debug(f"Skipping {sample['audio_file']}: {e}")
            return {
                "phoneme_indices": torch.tensor([0], dtype=torch.long),
                "mel_spec": torch.zeros((1, self.config.n_mels), dtype=torch.float32),
                "phoneme_durations": torch.tensor([1], dtype=torch.long),
                "stop_token_targets": torch.tensor([1.0], dtype=torch.float32),
                "audio_file": sample["audio_file"],
                "text": "",
            }


def collate_fn(batch: List[Dict]) -> Dict:
    """Pad and batch samples."""
    phoneme_indices_list = [item["phoneme_indices"] for item in batch]
    mel_specs_list = [item["mel_spec"] for item in batch]
    phoneme_durations_list = [item["phoneme_durations"] for item in batch]
    stop_targets_list = [item["stop_token_targets"] for item in batch]

    phoneme_lengths = torch.tensor(
        [len(p) for p in phoneme_indices_list], dtype=torch.long
    )
    mel_lengths = torch.tensor(
        [m.shape[0] for m in mel_specs_list], dtype=torch.long
    )

    phoneme_indices_padded = pad_sequence(
        phoneme_indices_list, batch_first=True, padding_value=0
    )
    phoneme_durations_padded = pad_sequence(
        phoneme_durations_list, batch_first=True, padding_value=0
    )

    max_mel_len = max(m.shape[0] for m in mel_specs_list)
    mel_dim = mel_specs_list[0].shape[1]

    mel_specs_padded = torch.zeros(len(batch), max_mel_len, mel_dim)
    stop_targets_padded = torch.zeros(len(batch), max_mel_len)

    for i, (mel, stop) in enumerate(zip(mel_specs_list, stop_targets_list)):
        mel_len = mel.shape[0]
        mel_specs_padded[i, :mel_len, :] = mel
        stop_targets_padded[i, :mel_len] = stop

    return {
        "phoneme_indices": phoneme_indices_padded,
        "mel_specs": mel_specs_padded,
        "phoneme_durations": phoneme_durations_padded,
        "stop_token_targets": stop_targets_padded,
        "phoneme_lengths": phoneme_lengths,
        "mel_lengths": mel_lengths,
        "audio_files": [item["audio_file"] for item in batch],
        "texts": [item["text"] for item in batch],
    }


class LengthBasedBatchSampler(Sampler):
    """Groups samples by similar length for efficient GPU utilization."""

    def __init__(
        self, dataset: Dataset, batch_size: int, drop_last: bool = True, shuffle: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.lengths = self._get_lengths()

    def _get_lengths(self) -> List[int]:
        from torch.utils.data import Subset

        if isinstance(self.dataset, Subset):
            underlying = self.dataset.dataset
            return [
                len(underlying.samples[i]["text"])
                for i in tqdm(self.dataset.indices, desc="Calculating lengths")
            ]
        return [
            len(s["text"])
            for s in tqdm(self.dataset.samples, desc="Calculating lengths")
        ]

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        pairs = sorted(zip(self.lengths, indices))

        batches = []
        for i in range(0, len(pairs), self.batch_size):
            batch = [idx for _, idx in pairs[i : i + self.batch_size]]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
