# Lessons Learned

Hard-won knowledge from building Igbo TTS. Reference this to avoid repeating mistakes.

---

## Data Pipeline

### WAXAL dataset has outlier-length clips
- **Problem**: Some clips are 17-29 minutes long (igbo_000240 = 1,771 seconds). These cause CUDA OOM even on 80GB GPUs and trigger gradient explosions.
- **Fix**: Cap `max_seq_length=1300` in config (~15 seconds of audio). This eliminates OOM risk and allows batch_size=48 on A100.
- **Lesson**: Always profile your dataset for sequence length distribution before training. We created `sequence_lengths.csv` on RunPod to audit this.

### WAXAL transcriptions contain newlines
- **Problem**: Some WAXAL transcription fields have embedded `\n` characters that break metadata.csv (LJSpeech format uses `|` delimiters, one line per sample).
- **Fix**: `" ".join(text.split())` during data prep.

### WAXAL config name is `ibo_tts`, not `ig_tts`
- The HuggingFace dataset config uses ISO 639-3 (`ibo`), not ISO 639-1 (`ig`).

### torchcodec doesn't work with FFmpeg 8
- WAXAL audio decoding needs soundfile, not torchcodec. Manual resampling with torchaudio to 22050 Hz.

### MFA alignment is essential for quality
- Without MFA TextGrid durations, the model uses uniform duration estimates → unintelligible output.
- With MFA, even 5 epochs produce structured (though not yet intelligible) audio.
- We pre-computed 1,485 TextGrids and committed them as `textgrids.tar.gz` (4.4 MB) so cloud training doesn't need to re-run MFA.

---

## Training Stability

### Gradient explosion from long sequences + high learning rate
- **What happened**: Epoch 3, batch 14 — grad_norm spiked to 16.05, cascaded to NaN for all subsequent batches. Model weights corrupted permanently.
- **Root cause**: 29-minute audio clip + learning_rate=1e-3.
- **Fix**: Lower lr to 5e-4 AND cap max_seq_length to 1300. Both were needed.
- **Lesson**: If grad_norm exceeds ~10, the model is at risk. Once weights go NaN, they never recover — you must restart from scratch or a clean checkpoint.

### Batch size vs gradient checkpointing tradeoff
- **Initial**: batch_size=48 + gradient_checkpointing=True → CUDA OOM (because of 29-min clips).
- **Intermediate**: batch_size=24 + gradient_checkpointing=True → worked but slow (~5 hours for 300 epochs).
- **Final**: batch_size=48 + gradient_checkpointing=False + max_seq_length=1300 → fast (~3.5 hours) and stable. The OOM was caused by outlier clips, not batch size.

### Validation always shows `total=nan, mel_refined=nan`
- This is a BatchNorm1d issue in the PostNet when running with bfloat16 mixed precision.
- The mel_coarse, duration, and stop losses in validation ARE valid.
- `save_best_only` must be False (it would fail on NaN val loss).
- Not a training blocker — the model trains fine, just validation total/refined metrics are NaN.

### NaN batch norm in inference (MPS and CUDA)
- PostNet uses BatchNorm1d which accumulates running stats that go NaN under bfloat16.
- Fix in inference.py: reset running_mean/running_var to 0/1 and set momentum=0.01 before inference.

---

## Kokoro Codebase Quirks

### Submodule has hardcoded "russian" filename
- `kokoro_training/training/checkpoint_manager.py` saves final model as `kokoro_russian_final.pth`.
- Must apply sed fix after every `git pull`: `sed -i 's/kokoro_russian_final/kokoro_igbo_final/g' kokoro_training/training/checkpoint_manager.py`
- This is because kokoro_training is a git submodule pointing to upstream.

### Monkeypatching requires patching the trainer module's namespace
- Can't just replace `IgboDataset` at the module level — `EnglishTrainer` copies the import reference at import time.
- Must patch `trainer_module.IgboDataset` (or whatever the trainer's namespace uses).

### MPS positional encoding device mismatch
- PE extension creates CPU tensors by default. Must pass `device=self.pe.device` when extending.
- PE size from checkpoint may exceed config's `max_decoder_seq_len` — detect from checkpoint and override.

---

## Cloud Training (RunPod)

### SSH command substitution can fail
- `$()` inside double-quoted SSH commands sometimes returns exit code 255.
- Use `killall python3` instead of `kill $(pgrep ...)` for reliability.
- Or use single quotes for the remote command.

### RunPod SSH needs direct TCP, not proxy
- `ssh -T -p PORT -i KEY root@IP` works. The `-T` flag is important (no PTY).

### WAXAL download is slow on RunPod
- ~40 files/min for 1,552 files. Takes ~40 minutes. Plan for this.

### Always check GPU memory after config changes
- `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader` is the quick check.
- 47 GB / 80 GB with batch_size=48 = plenty of headroom.

---

## Igbo Linguistics

### Igbo orthography is highly phonemic
- Nearly 1:1 letter-to-sound mapping. This makes G2P straightforward (rule-based, no ML needed).
- Digraphs must be processed before singles: ch, sh, gb, kp, gh, gw, kw, nw, ny.

### Tones are essential for meaning
- Igbo is a tone language with HIGH and LOW tones (plus downstep).
- Minimal pairs: akwa with different tones = "bed" vs "cloth" vs "cry" vs "egg".
- Our G2P encodes tones as explicit tokens (˥, ˩) — this is better than implicit handling.
- But: most written Igbo text lacks tone marks. A tone prediction model is needed for production use.

### Dotted vowels are phonemically distinct
- i/ị, o/ọ, u/ụ are different phonemes (not variants).
- Our G2P maps: ị→ɪ, ọ→ɔ, ụ→ʊ.

### espeak-ng does NOT support Igbo
- No existing G2P tool works for Igbo. We built our own.

---

## Inference Issues (Post-Training)

### Autoregressive decoder collapse
- **Problem**: Model produces near-silence mel (std=0.05) during autoregressive inference, but perfect mel (std=3.85) with teacher forcing. The decoder learned to rely on real mel input and collapses when fed its own predictions.
- **Root cause**: Scheduled sampling wasn't aggressive enough. Settings were: warmup_batches=5000, max_prob=0.5, zero_input_ratio=0.1. The model saw its own predictions too rarely during training.
- **Evidence**: Teacher-forced audio is recognizable Igbo speech. Free-running inference is silence/buzz.
- **Fix for next training**: Much more aggressive scheduled sampling — reduce warmup, increase max_prob to 0.8+, increase zero_input_ratio.

### PostNet BatchNorm completely broken
- **Problem**: PostNet's mel_refined output is 100% NaN. Not just validation — the PostNet is fundamentally broken from bfloat16 training.
- **Impact**: mel_coarse (pre-PostNet) is the only usable output. mel_refined (post-PostNet) is all NaN.
- **Fix in inference**: Bypass PostNet entirely — replace its forward() with identity (return zeros_like).
- **Fix for next training**: Either train in float32, or disable PostNet entirely and use single-path (coarse only) architecture.

### Duration predictor scale mismatch
- **Problem**: MFA durations only cover voiced phoneme segments, not silence. Training durations for a 435-frame mel might sum to only 93 frames. The model learns this compressed scale, but inference needs full-length output.
- **Fix in inference**: Scale predicted durations by ~5x: `log_dur + log(5.0)`.
- **Better fix**: Adjust dataset to include silence segments in durations so they sum to the total mel frame count.

### Griffin-Lim sounds rough/robotic
- This is expected — Griffin-Lim is a basic phase reconstruction algorithm. Speech will sound "demonic" but intelligible.
- HiFi-GAN will dramatically improve quality. This is the #1 audio quality upgrade.

### Audio clicks at start/end
- Caused by abrupt mel spectrogram edges (silence-to-speech transition).
- Fix: 10ms fade-in/fade-out on the final audio waveform.

---

## Workflow

### Commit textgrids.tar.gz to the repo
- MFA alignment takes 1-2 hours on CPU. Pre-computing and committing the result (4.4 MB) saves re-running on every cloud instance.

### Delete old checkpoints aggressively
- Checkpoints are large. `keep_last_n_checkpoints=2` or 5 is fine.
- Always clean old checkpoints before restarting training: `rm -f output_models_igbo/checkpoint_epoch_*.pth`

### Log sequence lengths for debugging
- When OOM or gradient issues hit, generate a CSV of all audio durations sorted by length. The outliers are usually the cause.
