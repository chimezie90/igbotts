# Research Notes: Igbo TTS & ASR

Working document for our research paper and product thinking.

---

## Thesis

We can build production-quality Igbo TTS by combining:
1. A large, high-quality single-speaker dataset (WAXAL, 180 hrs)
2. Explicit tone modeling in the phoneme representation
3. Rule-based G2P (viable because Igbo orthography is phonemic)
4. A TTS-ASR feedback loop to self-improve beyond the initial training data

This is the first neural Igbo TTS with explicit tone tokens and MFA-aligned phoneme durations.

---

## Related Work

### OkwuGbé (Dossou & Emezue, 2021)
- **Paper**: [arxiv.org/abs/2103.07762](https://arxiv.org/abs/2103.07762)
- First end-to-end ASR for Igbo (and Fon)
- Architecture: ResidualCNN (5 layers) + BiGRU (3 layers, 512d) + CTC loss
- Used Common Voice data — very limited Igbo audio at the time
- Tones handled implicitly through character set (no explicit tone tokens)
- Key finding: including diacritical marks in transcriptions improved accuracy
- They called Igbo results a "benchmark" (not SOTA) — data scarcity was the bottleneck
- **Gap we address**: They did ASR with scarce data; we do TTS with 180 hrs + explicit tones

### IgboAPI Dataset (Emezue, Okoh, Mbonu et al., 2024)
- **Paper**: [arxiv.org/abs/2405.00997](https://arxiv.org/abs/2405.00997)
- Multi-dialectal Igbo dictionary: 5,095 words, 33 dialects, 17,979 dialectal variations
- 278,899 parallel Igbo-English sentences (including synthetic dialectal augmentation)
- Fine-tuned M2M100 for MT: 16.87 → 71.95 BLEU (standard), 67.91 BLEU (unseen dialects)
- Dialectal augmentation gave +4.61 BLEU on unseen dialects
- UNESCO endangerment framing — tech as language preservation
- **Relevance to us**:
  - IgboAPI word list = good evaluation vocabulary for TTS
  - Dialectal data could expand TTS to multiple dialects in future
  - Same first author (Emezue) as OkwuGbé — he's the key researcher in this space

### WAXAL Dataset (Google, 2026)
- **Source**: [huggingface.co/datasets/google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP), config `ibo_tts`
- 1,552 Igbo TTS audio clips, studio quality, single speaker
- CC-BY-4.0 license
- Released February 2026 — very recent, likely not yet used in published research
- **This is our core data advantage**: 180 hrs of paired audio-text when prior work had maybe 5-10 hrs

### Kokoro / StyleTTS2
- Kokoro is a simplified reimplementation of StyleTTS2 for training
- Encoder-decoder architecture with duration predictor
- We use the medium config (~25-30M params) — right-sized for single-speaker data

---

## Our Contributions (for the paper)

### 1. First neural Igbo TTS with explicit tone modeling
- Prior work (OkwuGbé) was ASR, not TTS, and handled tones implicitly
- We encode HIGH (˥) and LOW (˩) as dedicated tokens in the phoneme sequence
- Hypothesis: explicit tone tokens produce better tonal accuracy than implicit character-level modeling
- **Evaluation**: tone minimal pairs (akwa variants) — does the model produce distinct F0 contours?

### 2. Rule-based Igbo G2P that actually works
- espeak-ng has no Igbo support
- We built a 50-token IPA phoneme inventory with full coverage of Igbo sounds
- Digraph-first processing handles ch, sh, gb, kp, gh, gw, kw, nw, ny correctly
- No ML needed — Igbo orthography is sufficiently phonemic for rule-based G2P
- **Contribution**: open-source Igbo G2P usable by other researchers

### 3. WAXAL as TTS training data
- First published use of WAXAL for Igbo TTS training (dataset only released Feb 2026)
- Document data prep pipeline: download, resample, MFA alignment, TextGrid extraction
- Characterize dataset: clip length distribution, outliers (29-min clips), quality issues

### 4. TTS-ASR feedback loop for low-resource languages
- Train TTS on WAXAL → generate synthetic audio from unlabeled text → train ASR
- Train ASR on WAXAL → transcribe unlabeled audio → expand TTS training data
- Round-trip evaluation: text → TTS → ASR → compare (automatic quality metric)
- Especially powerful for Igbo because unlabeled text is abundant (Wikipedia, IgboAPI, Bible translations) but paired data is scarce

### 5. Rigorous evaluation methodology for entangled TTS-ASR systems
- Round-trip WER (text → TTS → ASR → compare) conflates TTS and ASR errors
- We propose a disentanglement protocol using WAXAL ground truth as an anchor (see Evaluation Plan)
- This methodology is generalizable to any low-resource language feedback loop

---

## Evaluation Plan

### The Entanglement Problem

Round-trip evaluation (text → TTS → ASR → compare) is appealing because it requires no human raters, but it has a fundamental flaw: **errors from TTS and ASR are entangled**. A high WER could mean bad TTS, bad ASR, or both. Worse, the models can "collude" — if TTS consistently mispronounces a tone and ASR consistently mishears it the same way, round-trip WER looks fine but the speech is actually wrong.

This is especially dangerous for **tone**, where both models might learn the same systematic bias from the same WAXAL training data.

### Disentanglement Protocol

We use WAXAL's paired (audio, text) ground truth as an anchor to isolate each model's errors.

**Step 1: Measure ASR in isolation (ASR baseline)**
```
WAXAL held-out audio (real human speech, known text)
  → ASR → transcription
  → compare to WAXAL ground truth
  → ASR baseline WER (e.g., 15%)
```
This is the ASR's error floor on clean, studio-quality speech. Any round-trip WER above this is at least partially TTS's fault.

**Step 2: Measure TTS in isolation (acoustic comparison)**
```
WAXAL text → TTS → generated mel spectrogram
WAXAL text → (look up) → real mel spectrogram
  → MCD between generated and real mel
  → F0 RMSE between generated and real pitch contour
```
No ASR in the loop. Pure acoustic similarity to the human recording.

**Step 3: Attribute round-trip errors**
```
Round-trip WER = 35%
ASR baseline WER = 15%
TTS-attributable error ≈ 35% - 15% = 20%
```
Not perfectly additive (errors can compound or cancel), but gives a rough decomposition.

**Step 4: Cross-validate with reverse direction**
```
Forward:  Text → TTS → ASR → Text'     (TTS-weighted: ASR judges TTS)
Reverse:  Audio → ASR → TTS → Audio'   (ASR-weighted: TTS judges ASR)
```
If forward is bad but reverse is decent → TTS is the weak link.
If both are bad → ASR is the bottleneck.

**Step 5: Per-phoneme error analysis**
Run round-trip on controlled word lists, group errors by phoneme/tone. This reveals which specific sounds each model struggles with, enabling targeted improvement.

### Objective Metrics

| Metric | What it measures | How | Isolates |
|--------|-----------------|-----|----------|
| MCD (Mel Cepstral Distortion) | Spectral similarity to ground truth | Compare generated mel to WAXAL reference | TTS only |
| F0 RMSE | Pitch/tone accuracy | Extract F0 contours, compare to reference | TTS only |
| F0 direction accuracy | Tone token correctness | For each tone token, does F0 go up (H) or down (L)? | TTS only |
| Duration RMSE | Phoneme timing accuracy | Compare predicted vs MFA durations | TTS only |
| ASR baseline WER | ASR quality on real speech | ASR on held-out WAXAL audio | ASR only |
| Round-trip WER | Combined intelligibility | text → TTS → ASR → WER against original | Both (entangled) |
| Adjusted round-trip WER | TTS intelligibility estimate | Round-trip WER minus ASR baseline WER | TTS (approximate) |
| Reverse round-trip MCD | Combined acoustic fidelity | audio → ASR → TTS → compare spectrograms | Both (ASR-weighted) |
| Tone minimal pair accuracy | Tone discrimination | Generate minimal pairs, check F0 difference | TTS only |

### Subjective Metrics (if we can get raters)
| Metric | What it measures |
|--------|-----------------|
| MOS (Mean Opinion Score) | Overall naturalness (1-5 scale) |
| Tone intelligibility | Can listeners distinguish tonal minimal pairs? |
| Dialect acceptability | Does it sound like real Igbo to native speakers? |
| ABX tone test | Play two audio clips of tone minimal pairs, ask which is which |

### Tone Minimal Pairs for Testing
| Pair | Meaning 1 | Meaning 2 |
|------|-----------|-----------|
| àkwà / ákwá | bed / cloth |
| ọ́kụ́ / ọ̀kụ̀ | fire / leg |
| égó / ègò | money / masquerade |
| àzụ̀ / ázụ́ | back / fish |
| ụ́lọ̀ / ùlô | house / (variant) |

### Limitations We Acknowledge

1. **Systematic tone bias**: If WAXAL's speaker has idiosyncratic tone patterns, both TTS and ASR learn the same bias. Round-trip evaluation cannot detect this — only native speaker evaluation can.
2. **Single-speaker ceiling**: All acoustic metrics are measured against one speaker. A TTS that sounds different but equally natural would score poorly on MCD/F0 RMSE.
3. **ASR baseline is optimistic**: ASR baseline WER is measured on studio audio. Real-world TTS audio has artifacts (vocoder noise, etc.) that may cause additional ASR errors beyond what the baseline predicts.
4. **Tone marks in ground truth**: If WAXAL transcriptions lack tone marks, our ASR can't learn to output them, which limits tone-specific round-trip evaluation.

---

## TTS-ASR Feedback Loop: Detailed Design

### How TTS Improves ASR

ASR needs paired (audio, text) data. Unpaired Igbo text is abundant:
- Igbo Wikipedia (~7,000 articles)
- IgboAPI (278,899 sentences across 33 dialects)
- Igbo Bible translations
- BBC Igbo news articles
- Nollywood subtitle files

**Pipeline:**
1. Collect 100K+ unpaired Igbo sentences
2. TTS generates audio for each → synthetic paired data
3. Mix synthetic + real WAXAL audio (70/30 ratio) to prevent ASR overfitting to robot speech
4. Train ASR on combined dataset → vocabulary and coverage jump massively

### How ASR Improves TTS

TTS needs more audio diversity. Unpaired Igbo audio exists:
- YouTube (sermons, vlogs, news in Igbo)
- Radio archives (Igbo-language stations)
- Podcast archives
- Church recordings

**Pipeline:**
1. Scrape Igbo audio from YouTube/radio/podcasts
2. ASR transcribes each clip → new paired data
3. Filter: keep only high-confidence transcriptions (confidence > 0.9)
4. Run MFA alignment on new audio → duration targets
5. Retrain TTS on WAXAL + new data → multi-speaker, more diverse

### Cycle Dynamics

```
Cycle 0: Both models trained on 180 hrs WAXAL only
Cycle 1: TTS augments ASR with 100K synthetic utterances
         ASR augments TTS with ~500 hrs YouTube/radio transcriptions
         Round-trip WER: ~40% → ~25%
Cycle 2: Better TTS → higher quality synthetic audio → better ASR
         Better ASR → more accurate transcriptions → better TTS
         Round-trip WER: ~25% → ~15%
Cycle 3+: Diminishing returns, converges around 5-10% round-trip WER
```

### Overcoming Feedback Loop Limitations

**Problem 1: Error amplification — bad model outputs become bad training data**

Models can reinforce each other's mistakes. If TTS mispronounces a word and ASR learns from that mispronunciation, both models converge on the wrong answer.

*Solutions:*
- **Confidence filtering**: Only use synthetic data where the generating model is highly confident. For TTS→ASR: filter by ASR confidence score. For ASR→TTS: filter by ASR confidence + language model perplexity.
- **Real data anchoring**: Always keep WAXAL real data as a fixed proportion (≥30%) of every training batch. This prevents drift from ground truth.
- **Periodic re-evaluation against held-out WAXAL**: If metrics on the WAXAL test set degrade, the loop is amplifying errors. Roll back.
- **Data mixing curriculum**: Start each cycle with 90% real / 10% synthetic, gradually increase synthetic ratio only as quality improves.

**Problem 2: Systematic tone bias — both models learn the same wrong tones**

If WAXAL's speaker has non-standard tone patterns, both models inherit the bias and round-trip evaluation can't detect it.

*Solutions:*
- **External tone oracle**: Use linguistic rules (Igbo tone sandhi rules are well-documented) as an independent check. Compare model F0 contours against rule-predicted tone patterns.
- **Multi-source audio in ASR→TTS direction**: When we scrape YouTube/radio audio, we get different speakers with different tone realizations. This diversifies the tone signal and washes out single-speaker bias.
- **ABX tone discrimination test with native speakers**: Even a small panel (5-10 speakers) testing 50 minimal pairs gives high-signal evaluation data.
- **F0 analysis on known words**: For words with unambiguous tones (from dictionaries), check that TTS F0 contour matches the expected HIGH/LOW pattern. This is automatable, no ASR needed.

**Problem 3: Domain mismatch — synthetic audio sounds different from real audio**

TTS audio has vocoder artifacts, uniform speaking style, and no background noise. ASR trained on synthetic audio may not generalize to real-world audio.

*Solutions:*
- **Audio augmentation on synthetic data**: Add noise, reverb, speed perturbation to TTS outputs before feeding to ASR. This bridges the acoustic domain gap.
- **Style token conditioning**: If we add a noise/style token to TTS, we can generate "noisy" synthetic audio that better matches real-world conditions.
- **Progressive domain adaptation**: Train ASR first on clean synthetic, then fine-tune on noisy real audio. The synthetic pre-training provides vocabulary coverage, the real fine-tuning provides acoustic robustness.

**Problem 4: Convergence to mediocrity — models plateau at "good enough"**

After a few cycles, the models stop improving because they're only learning from each other.

*Solutions:*
- **Inject external data each cycle**: Don't just recycle the same data. Each cycle, scrape new text sources and new audio sources. Fresh data prevents stagnation.
- **Active learning**: Use round-trip disagreements to find the hardest examples. Specifically target sentences where round-trip WER is highest for the next cycle's training data.
- **Human-in-the-loop on hard cases**: For the 5% of utterances where models persistently disagree, get a native speaker to provide the ground truth. Small amounts of expert annotation on the hardest cases have outsized impact.

---

## Tone Predictor: Detailed Design

### The Problem
Most written Igbo lacks tone marks. A user typing `Akwa di na ulo` could mean several different things. The TTS needs tone-marked input to pronounce correctly.

### Architecture Options

**Option A: BiLSTM + CRF (baseline)**
```
Character embeddings (64d) → BiLSTM (2 layers, 256d) → CRF → per-vowel tone labels
```
- CRF enforces valid tone sequences (e.g., downstep only after high tone)
- Fast, small, runs on CPU. Good starting point.

**Option B: Fine-tuned transformer (best quality)**
```
Igbo text → AfroLM or XLM-RoBERTa → token classification head → tone labels
```
- Captures sentence-level context (grammatical tone, phrasal tone rules)
- AfroLM is pre-trained on African languages including Igbo
- Larger model, needs GPU, but highest accuracy

**Option C: TTS+ASR ensemble (no separate model needed)**
```
For ambiguous word → generate all tone variants via TTS
→ ASR scores each → pick highest confidence
→ or: language model scores each → pick lowest perplexity
```
- Creative zero-shot approach. Slow but requires no tone-labeled training data.
- Good for batch processing, bad for real-time.

**Option D: Lexicon lookup + context rules (hybrid)**
```
Known words → look up tones in dictionary (IgboAPI, Kay Williamson)
Unknown words → fall back to BiLSTM/transformer predictor
Grammar rules → override where applicable (e.g., verb tone in past tense)
```
- Most practical for production. Dictionary covers common words perfectly.
- ML model only needed for rare/novel words.

### Training Data Sources
| Source | Size | Tone quality |
|--------|------|-------------|
| WAXAL transcriptions | 1,552 sentences | Need to verify |
| Igbo Bible (tone-marked editions) | ~31,000 verses | High |
| Kay Williamson's dictionary | ~10,000 entries | High |
| IgboAPI dataset | 5,095 words | Partial |
| Linguistic textbooks/papers | Variable | High |
| F0-derived pseudo-labels from WAXAL audio | 1,552 sentences | Medium (automatic) |

### Novel idea: Derive tone labels from audio F0

Even if WAXAL transcriptions lack tone marks, the **audio contains the tone information**. We can extract it:
```
1. MFA gives us phoneme boundaries (we already have this)
2. Extract F0 (pitch) contour from audio using CREPE or WORLD
3. For each vowel segment, measure mean F0
4. Cluster into HIGH vs LOW based on relative F0 within utterance
5. → pseudo-tone-labels for every vowel in every WAXAL utterance
```
This gives us 1,552 sentences of automatically tone-labeled text — enough to bootstrap a tone predictor without any manually tone-marked data. Quality won't be perfect (boundary cases, downstep), but it's a strong starting signal.

---

## Product Vision

### Product 1: Igbo TTS API
- REST API: send Igbo text → receive audio WAV/MP3
- Use cases: audiobooks, language learning apps, accessibility, voice assistants
- Challenge: most input text won't have tone marks → need tone prediction model

### Product 2: Igbo ASR API
- REST API: send audio → receive Igbo transcription (with tone marks)
- Use cases: transcription services, voice search, dictation, subtitle generation
- Bootstrapped from same WAXAL data, improved via feedback loop

### Product 3: Igbo Tone Predictor
- Input: undiacriticized Igbo text → output: text with tone marks
- Essential for TTS (most written Igbo lacks tone marks)
- Could be trained on: WAXAL transcriptions (if tone-marked), Igbo dictionaries, linguistic resources
- This is a hard NLP problem but high-impact

### Data Flywheel
```
TTS API users send text → we collect (anonymized) text samples → more training data
ASR API users send audio → we collect (anonymized) audio samples → more training data
Both models improve → more users → more data → better models
```

---

## Open Questions

1. **Is WAXAL tone-marked?** We need to check if the transcriptions include diacritical tone marks or just dotted vowels. This determines whether our model can learn tones from the training data at all. If not, we can still derive pseudo-tone-labels from F0 contours (see Tone Predictor section).

2. **What dialect is WAXAL?** Likely Standard/Central Igbo but we should confirm. The IgboAPI paper shows 33 dialects — our model will only speak whichever dialect WAXAL contains.

3. **How good is Griffin-Lim really?** We're using Griffin-Lim vocoder for now. HiFi-GAN would be a major quality upgrade but needs setup. How much quality are we leaving on the table?

4. **Can we get native speaker evaluators?** MOS testing requires Igbo speakers. The IgboAPI team (Emezue et al.) might be interested in collaboration — they're the most active Igbo NLP research group.

5. **Is 300 epochs enough?** Loss is still dropping at epoch 195. Might benefit from 500+ epochs, but diminishing returns vs. RunPod cost.

6. **How does our model compare to commercial APIs?** Google TTS and Amazon Polly don't support Igbo. Microsoft has limited Igbo TTS. We should benchmark against whatever exists.

7. **How do we prevent the feedback loop from amplifying tone errors?** Our proposed solutions: external tone oracle from linguistic rules, multi-speaker audio diversity, ABX tests with native speakers, and F0 analysis on dictionary words with known tones. But we need to test which of these actually catches errors in practice.

8. **What's the minimum viable tone predictor?** Dictionary lookup (IgboAPI + Kay Williamson) might cover 80%+ of running text. The ML model is only needed for the long tail. Need to measure dictionary coverage on realistic Igbo text to decide how much ML investment is needed.

9. **Can F0-derived pseudo-labels bootstrap a tone predictor?** We hypothesize that extracting pitch from WAXAL audio + MFA boundaries can produce usable tone labels without any manually annotated data. This needs validation against linguist-annotated ground truth.

---

## Timeline

| Phase | Task | Status |
|-------|------|--------|
| 1 | TTS model training (300 epochs) | DONE - model learned Igbo (teacher-forced audio proves it) |
| 1b | Fix autoregressive inference collapse | NEXT - need aggressive scheduled sampling retrain |
| 2 | Inference evaluation + tone minimal pairs | Blocked on 1b |
| 3 | HiFi-GAN vocoder integration | Planned (will replace rough Griffin-Lim audio) |
| 4 | Fine-tune Whisper for Igbo ASR | Planned |
| 5 | TTS-ASR feedback loop (first iteration) | Planned |
| 6 | Tone prediction model | Planned |
| 7 | Write research paper | Planned |
| 8 | API productionization | Planned |

## Training Run 1 Results (Feb 16, 2026)

**Cost**: ~$6 (A100 SXM 80GB @ $1.49/hr x ~4 hours including setup)

**What we got**:
- 300-epoch trained model (kokoro_igbo_final.pth, 106 MB)
- Model successfully learned Igbo speech representations (proven via teacher-forced inference)
- Teacher-forced audio is recognizable Igbo speech
- Full training logs, checkpoints at epochs 260-300

**What went wrong**:
1. **Autoregressive decoder collapsed**: Model outputs near-silence (mel std=0.05) when generating freely, but produces excellent mel (std=3.85, matching training target std=3.82) with teacher forcing. The decoder learned to depend on ground-truth input and can't bootstrap from its own predictions.
2. **PostNet completely broken**: All NaN output from bfloat16 BatchNorm corruption. mel_refined was NaN throughout all 300 epochs of training — we saw this in validation logs ("total=nan, mel_refined=nan") but didn't realize it meant the PostNet path was dead.
3. **Duration scale mismatch**: Predicted durations are 5x too short because MFA durations exclude silence segments.

**Root cause analysis**:
- Scheduled sampling settings were too conservative: warmup_batches=5000 (167 epochs!), max_prob=0.5. The model spent the first 167 epochs in pure teacher forcing, then only gradually saw its own predictions for the remaining 133 epochs — not enough exposure to self-generated input.
- The mel_refined NaN should have been a red flag. The "good" loss numbers (mel_coarse=0.44) masked the fact that the PostNet output path was dead.

**Fixes for Training Run 2**:
- `scheduled_sampling_warmup_batches`: 5000 → 500 (start self-exposure by epoch 17)
- `scheduled_sampling_max_prob`: 0.5 → 0.8 (use own predictions 80% of the time by mid-training)
- `scheduled_sampling_zero_input_ratio`: 0.1 → 0.3 (30% of decoder inputs are zeros — forces generation)
- Disable PostNet entirely OR train in float32 for PostNet layers
- Fix duration dataset to include silence segments so durations sum to total mel frames

**What we keep from Run 1**:
- All data pipeline code (G2P, dataset, MFA alignments) — proven working
- Config and cloud setup scripts — proven working
- The knowledge that the model architecture CAN learn Igbo speech
- Understanding of what hyperparameters matter most
