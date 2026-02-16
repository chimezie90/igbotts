# Igbo TTS/ASR Project

## Purpose

Build the first high-quality neural text-to-speech (TTS) and automatic speech recognition (ASR) system for the Igbo language, with explicit tone modeling and a self-improving feedback loop between the two models.

Igbo is spoken by 27+ million people but has virtually no production-quality speech technology. UNESCO projects it could become endangered. This project is both a research contribution and a product — the models will power real applications.

## Architecture

We adapt the [Kokoro training codebase](https://github.com/jonirajala/kokoro_training) (a simplified StyleTTS2 encoder-decoder TTS) for Igbo. Key adaptations:

- **Custom Igbo G2P**: Rule-based grapheme-to-phoneme converter (50-token IPA vocab including HIGH/LOW tone markers)
- **WAXAL dataset**: Google's CC-BY-4.0 Igbo TTS dataset — 1,552 clips, ~180 hours, single-speaker, studio quality
- **MFA alignment**: Montreal Forced Aligner provides phoneme-level duration targets
- **Same model architecture as Kokoro**: Transformer encoder-decoder with duration predictor

## Project Structure

```
igbotts/
├── igbo_tts/              # Our custom Igbo modules
│   ├── g2p.py             # Igbo grapheme-to-phoneme (rule-based, IPA output)
│   ├── phonemes.py        # 50-token Igbo phoneme inventory
│   ├── dataset.py         # WAXAL dataset loader (mel + phonemes + durations)
│   ├── setup_data.py      # Download & prepare WAXAL data
│   ├── config.py          # Training configs (small/medium/default/cloud/large)
│   └── inference.py       # TTS inference (G2P → model → vocoder → WAV)
├── kokoro_training/       # Git submodule — upstream Kokoro trainer
├── data/waxal_igbo/       # Downloaded audio + TextGrid alignments (gitignored)
├── train.py               # Main training entry point
├── scripts/cloud_setup.sh # RunPod setup script
└── textgrids.tar.gz       # Pre-computed MFA alignments (committed)
```

## Key Technical Decisions

- **Phoneme inventory**: 8 vowels + 28 consonants + 2 tones + special tokens = 50 tokens
- **Tone as explicit tokens**: HIGH (˥) and LOW (˩) are separate tokens in the phoneme sequence, not implicit
- **MFA for alignment**: 1,485 TextGrid files with phoneme boundaries — critical for audio quality
- **Medium model for training**: 25-30M params (384d, 4L enc/dec, 8H) — optimal for single-speaker data
- **max_seq_length=1300**: Caps audio at ~15 seconds to prevent gradient explosions from outlier clips

## Training

- **Cloud**: RunPod A100 SXM 80GB ($1.49/hr)
- **Config**: batch_size=48, lr=5e-4, 300 epochs, bfloat16 mixed precision
- **Time**: ~3.5 hours for full training

## Roadmap

1. TTS model (current — training now)
2. HiFi-GAN vocoder (replace Griffin-Lim for better audio quality)
3. ASR model (fine-tune Whisper on same WAXAL data)
4. TTS-ASR feedback loop (self-improving cycle using unlabeled text and audio)
5. Tone prediction model (auto-add tone marks to undiacriticized Igbo text)
6. Multi-dialect expansion (leverage IgboAPI's 33-dialect dataset)

## GitHub

https://github.com/chimezie90/igbotts
