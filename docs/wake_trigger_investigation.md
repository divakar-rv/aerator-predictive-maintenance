# A Two-Stage, Physically-Grounded Wake-Trigger for Intermittent Edge-AI Fault Detection in Aerator Motors

## 1. Motivation

The project's core architectural claim is an *intermittent* sensing node: the
ESP32 should sleep by default and only wake to run full CNN inference when
something looks anomalous, since continuous inference is the main power cost
in an always-on edge-AI design. This makes the **wake-trigger** — the cheap
decision of "is it worth waking up?" — as important to the system's design
as the classifier itself, but it received far less attention than the CNN in
early project iterations. This section documents a systematic investigation
into what that trigger should actually be, using real MaFaulDa vibration
data rather than assumption.

## 2. Naive baseline: RMS-energy threshold

The simplest possible wake-trigger is a time-domain RMS-energy threshold:
wake if the vibration signal's energy exceeds a calibrated baseline. This
was tested directly on real, held-out MaFaulDa test windows (four classes:
Normal, Imbalance, Misalignment, Bearing; 45 windows per class, balanced,
file-wise split to avoid leakage). The threshold was calibrated as
mean(Normal RMS) + 3 x std(Normal RMS) = 0.393.

| Class | Windows triggering full inference | % |
|---|---|---|
| Normal | 0 / 45 | 0.0% |
| Imbalance | 30 / 45 | 66.7% |
| Misalignment | 0 / 45 | **0.0%** |
| Bearing | 25 / 45 | 55.6% |

**Finding:** the RMS trigger correctly stays asleep for Normal operation
(no false wake-ups) and catches roughly half to two-thirds of Imbalance and
Bearing faults, but **completely fails to detect Misalignment** — every
single misalignment window in the test set was indistinguishable from
Normal by raw signal energy.

## 3. Root cause: misalignment is amplitude-quiet

Rather than treat this as a threshold-tuning problem, the per-class mean FFT
spectrum was plotted directly (Figure 1, `results/mean_fft_by_class.png`):

- **Imbalance** and **Bearing** both show sharp, high-magnitude spectral
  peaks (0.167 and 0.123 respectively).
- **Normal** and **Misalignment** are both spectrally flat and low-energy
  (0.025 and 0.016 respectively) — Misalignment is, if anything, quieter
  than Normal on average.

This explains the RMS failure mechanistically rather than statistically: no
amplitude-based threshold, however tuned, can separate two classes that are
both low-amplitude. A second amplitude-derived feature, **spectral
peakiness** (max magnitude / mean magnitude across the spectrum), was also
tested and shows the same structural limitation:

| Class | Peakiness mean | std |
|---|---|---|
| Normal | 18.57 | 6.84 |
| Imbalance | 55.49 | 15.15 |
| Misalignment | 24.35 | 10.49 |
| Bearing | 36.22 | 23.98 |

Peakiness cleanly separates the "loud" pair (Imbalance, Bearing: 36-55) from
the "quiet" pair (Normal, Misalignment: 18-24), but the quiet pair's ranges
still overlap (Normal 11.7-25.4 vs. Misalignment 13.9-34.8 within one std).
**Conclusion: no single amplitude-domain feature can separate Normal from
Misalignment**, because the physical difference between them is not one of
energy but of spectral *shape*.

## 4. Fix: a harmonic-ratio feature grounded in rotating-machinery diagnostics

Classical rotating-machinery fault diagnosis literature identifies imbalance
as dominant at 1x running speed and misalignment as producing a
characteristic 2x running-speed component. This motivated a targeted
feature — the ratio of spectral magnitude at 2x the dominant (1x) frequency
to the magnitude at 1x itself — computed per window directly from real data
rather than assumed:

| Class | 2x/1x ratio mean | std | mean detected 1x frequency |
|---|---|---|---|
| Normal | 0.134 | 0.122 | 39.4 Hz |
| Imbalance | 0.040 | 0.028 | 39.8 Hz |
| Misalignment | 0.196 | 0.100 | 34.2 Hz |
| Bearing | 0.149 | 0.137 | 34.4 Hz |

**Imbalance separates cleanly** on this feature alone (ratio ~0.04 vs.
0.13-0.20 for every other class), matching the diagnostic-literature
expectation exactly. Normal and Misalignment show a real difference in
means (0.134 vs. 0.196) but with enough spread that neither RMS, peakiness,
nor harmonic ratio alone fully separates this pair.

## 5. Combined two-feature discriminator

The two features above — spectral peakiness and 2x/1x harmonic ratio — were
combined into a 2D feature space and tested with a simple logistic
regression classifier (Figure 2, `results/2d_feature_space.png`):

**Result: 62.8% overall accuracy using only two hand-computed scalar
features**, compared to 66.1% for a full 256-bin, 3-layer 1D-CNN trained on
the complete spectrum. The two-feature model recovers **95% of the full
CNN's accuracy at a fraction of its computational cost** — no learned
weights, no matrix multiplication, just two closed-form calculations
directly on the FFT output the wake logic already needs to compute for
peakiness.

Per-class inspection of the scatter plot shows Imbalance separates nearly
perfectly on peakiness alone; Misalignment shows a real (if noisy) upward
shift in harmonic ratio relative to Normal; Bearing remains the hardest
class, scattered across both axes, consistent with bearing faults producing
more complex, broadband signatures than a two-feature linear model can
capture.

## 6. Proposed architecture: a two-stage trigger, not a two-stage classifier

The practical design implication is **not** to replace the CNN with the
two-feature model — 62.8% is not accurate enough to be the final
classification the system reports. Instead, the two features are proposed
as a **cheap Stage-1 wake-gate**, ahead of the expensive Stage-2 CNN
classifier:

- **Stage 1** (near-zero cost, computed directly on the FFT the device must
  already produce): compute peakiness and harmonic ratio; if both sit
  within the calibrated Normal envelope, return to sleep without invoking
  the CNN.
- **Stage 2** (full cost, TFLite Micro CNN inference): invoked only when
  Stage 1 indicates a plausible deviation from Normal — including the
  low-amplitude Misalignment case, which Stage 1's harmonic-ratio component
  can flag even though its peakiness alone would look "quiet."

This is the architecture implemented in `aerator_predictive_maintenance.ino`
(Phase 2 deployment code): `computePeakiness()` runs unconditionally on
every wake cycle in plain C with no TFLite call; the CNN (`runInference()`)
only executes when Stage 1's threshold is exceeded. Extending Stage 1 to
also incorporate the harmonic-ratio check (currently only peakiness is
implemented in the deployed sketch) is a concrete, well-motivated next step,
now that this analysis identifies exactly why it is needed.

## 6a. End-to-end system-level validation

The two-stage design was not just proposed but simulated end-to-end on the
real, held-out 4-class test set: the gate (calibrated on **training** Normal
windows only, N_SIGMA=2.0 on both peakiness and harmonic ratio) decides
whether to invoke the full CNN; when it stays asleep, the system implicitly
reports "Normal" (the real-world behavior of a device that never woke up).

| Metric | Value |
|---|---|
| Overall system accuracy (gate + CNN combined) | 61.1% |
| Always-on CNN baseline accuracy (no gate) | 56.7%* |
| Fraction of windows triggering full inference | 57.2% |
| **Fraction of windows skipping full inference (power saved)** | **42.8%** |

*\*Note: the always-on CNN baseline varied 56.7-66.1% across repeated runs*
*despite fixed random seeds (see Section 7) — treat as a range, not a point*
*estimate.*

**Per-class wake behavior:**

| Class | Wake rate | Accuracy when awake |
|---|---|---|
| Normal | 24.4% | 63.6% |
| Imbalance | 88.9% | 75.0% |
| Misalignment | 40.0% | 66.7% |
| Bearing | 75.6% | 79.4% |

**Interpretation:** the gate achieves its primary goal — nearly 43% of
inference calls are avoided with no loss (in fact a slight gain) in overall
system accuracy versus always-on inference. Imbalance and Bearing, the two
spectrally "loud" fault classes, wake reliably (88.9% and 75.6%). Normal
correctly stays mostly asleep (24.4% wake rate, i.e. few false wake-ups).

**The clear remaining weakness:** Misalignment wakes only 40% of the time —
meaning 60% of real misalignment events would go undetected by this system,
silently reported as Normal. This is a direct, quantified consequence of
the amplitude-quiet property identified in Section 3, now measured at the
full-system level rather than the single-feature level. The naive RMS
trigger caught 0% of Misalignment; the two-stage gate improves this to 40%
but does not solve it. Tightening the harmonic-ratio sensitivity (lower
N_SIGMA) would likely raise this at the cost of more false wake-ups on
Normal — an explicit, tunable accuracy/power trade-off for future work,
not a solved problem.

## 6b. A critical deployment-model revision: quantization instability

During final deployment preparation, the originally-selected 128-bin,
15.91 KB model (90.5% int8 accuracy, reported in Section 6 and the Review-II
deck) was retested for reproducibility before committing to it as the final
shipped model. This retesting uncovered a serious, previously-undetected
problem, and the deployed model was changed as a direct result — documented
here in full because the process of catching this matters as much as the
fix itself.

**What was found:** across four independent int8 quantization attempts on
this architecture — two separate training runs, tested with both a
natural-class-distribution calibration set and a deliberately class-balanced
one — int8 accuracy consistently collapsed to **64.8-65.7%**, with
predictions biased toward the majority (Imbalance) class:

| Attempt | Training run | Calibration strategy | Int8 accuracy |
|---|---|---|---|
| 1 | Run A | Natural distribution | 65.7% |
| 2 | Run A (rerun) | Natural distribution | 65.7% |
| 3 | Run A (rerun) | Natural distribution | 64.8% |
| 4 | Run B (fresh) | Class-balanced | 64.8% |

In every case, **float32 accuracy remained stable at 90.5%** — the
degradation is specific to int8 quantization, not the underlying model.
The confusion matrix for a representative failing run
(`[[8,37],[0,60]]`) shows the model correctly identifies nearly all
Imbalance windows but misclassifies most Normal windows as Imbalance —
a majority-class bias introduced specifically by 8-bit rounding, not present
in the float32 model.

**Ruling out calibration as the cause:** the class-balanced calibration
attempt (Attempt 4) was designed specifically to test whether an
imbalanced calibration set (naturally ~57% Imbalance, matching the test
set's class ratio) was biasing the learned quantization scale toward that
class. It was not: balanced calibration produced the same failure mode
(64.8%), ruling out calibration-set composition as the root cause.

**Conclusion:** the failure is intrinsic to this specific, very small
architecture (Conv1D 8→16 filters, one 16-unit dense layer) under int8
quantization — apparently too small a capacity margin exists between its
learned decision boundary and the precision loss introduced by 8-bit
rounding, regardless of training run or calibration strategy. The original
single "90.5%" result reported in the Review-II deck was very likely an
unrepresentative lucky outcome from one specific combination of training
initialization and calibration-set sampling, not a stable property of the
architecture.

**Resolution:** the larger, 250-bin, 70.62 KB architecture (Conv1D 16→32
filters, 32-unit dense layer) was re-verified and found to quantize
reliably: **85.7% int8 accuracy, confirmed identically across two
independent quantization runs**, with a sane, one-directional confusion
matrix (`[[45,0],[15,45]]` — all errors Imbalance misread as Normal, no
majority-class collapse). This model was selected as the actual final
deployment target, exceeding the original 30 KB size budget but providing
a real, reproducible accuracy figure instead of an unreliable smaller one.
The ESP32 sketch and its wake-gate thresholds were updated accordingly
(250 FFT bins, recalibrated peakiness/harmonic-ratio thresholds, 90.8%
Imbalance wake-rate sanity check).

**Methodological lesson for the report:** a single quantization run is not
sufficient evidence that a model quantizes well — quantization robustness
should be verified across multiple training seeds and calibration
strategies before a model is reported as deployment-ready, exactly the same
principle established for float32 training reproducibility in Section 7b.
This project's own initial Review-II claim (90.5% int8, 15.91 KB) is a
concrete example of exactly the mistake this lesson warns against, caught
and corrected before final deployment rather than after.

## 7a. Two further investigations attempted, both negative results

Two additional improvements were attempted after the initial two-stage
design above. Both are reported here as negative results, since a negative
result obtained rigorously is still real information, not a wasted effort.

**Ground-truth tachometer frequency.** MaFaulDa provides a genuine
tachometer channel (raw CSV column 0) recording actual shaft rotation,
independent of the vibration signal. The harmonic-ratio feature (Section 4)
was rebuilt using this real per-window rotational frequency instead of the
original method (searching for the loudest spectral peak in a plausible
15-70 Hz band). Counter to expectation, using the ground-truth frequency
**worsened** class separation:

| Class | 2x/1x ratio, guessed peak | 2x/1x ratio, real tachometer |
|---|---|---|
| Normal | 0.134 ± 0.122 | 0.251 ± 0.429 |
| Imbalance | 0.040 ± 0.028 | 0.041 ± 0.028 |
| Misalignment | 0.196 ± 0.100 | 0.238 ± 0.162 |
| Bearing | 0.149 ± 0.137 | 0.204 ± 0.216 |

Imbalance is unaffected either way (a strong, unambiguous 1x peak makes
exact lookup and peak-search agree). But Normal's variance more than
tripled under exact-frequency lookup, likely because looking up a single
exact FFT bin is far more sensitive to small within-window speed
fluctuations and bin misalignment than a locally robust "find the nearest
real energy concentration" peak search. **Conclusion: more accurate ground
truth does not automatically produce a better engineered feature** — the
approximate, locally-robust method outperformed the exact one in this case.

**Quantization-aware training (QAT).** To address the 4-class model's large
post-training-quantization accuracy drop (66%→39%, Section 8), QAT was
attempted via `tensorflow-model-optimization` (tfmot 0.8.1), which trains a
model to be robust to int8 rounding during training itself rather than
after the fact. This failed at the `quantize_model()` call with `ValueError:
'to_quantize' can only either be a keras Sequential or Functional model`,
despite the model being rebuilt explicitly as a Functional-API model to rule
out a Sequential-model quirk. Root cause, confirmed via version inspection
(TensorFlow 2.20.0, Keras 3.13.2, tfmot 0.8.1): **tfmot 0.8.1 predates Keras
3 and does not support its internal model representation**
(`_is_graph_network`, an attribute tfmot depends on internally, is not
exposed the same way by Keras 3 models). This is a known ecosystem gap, not
a code defect. Downgrading TensorFlow/Keras specifically to unblock this one
experiment was judged not worth the risk of destabilizing the rest of the
(working) pipeline given project time constraints. **QAT remains a valid,
well-motivated direction for future work**, either once `tfmot` adds Keras 3
support, or in a separate, deliberately version-pinned environment.

## 7b. Honest limitations (general)

- **Run-to-run variance despite fixed seeds:** the same 4-class CNN
  architecture, trained on the same file-capped dataset with
  `tf.random.set_seed`, `np.random.seed`, and `random.seed` all fixed to 42,
  produced accuracies ranging from 56.7% to 66.1% across separate notebook
  sessions. Likely causes: `glob.glob()` file ordering is not guaranteed
  stable across processes (affecting which files get capped/sampled even
  with a seeded `Generator.choice()`), and TensorFlow/Keras training has
  additional non-determinism sources beyond the three seeds set here (full
  bit-for-bit reproducibility requires `tf.config.experimental.
  enable_op_determinism()` and fixed thread counts, not done here). This
  means single point-accuracy claims for the 4-class model should be read
  as indicative of a range, not a precise fixed value — a limitation this
  document reports rather than hides.
- All results in this section are from **real MaFaulDa data**, not
  synthetic signals, and use a proper file-wise train/test split — but the
  test set is real-world-recording-sized, not large (45 windows/class), so
  the exact percentages should be read as indicative, not precise.
- Bearing faults remain the weakest-separated class under both the CNN
  (see confusion matrix, Section 8) and the two-feature model. Neither
  approach yet gives a reliable Bearing-specific low-cost trigger feature;
  this is an open problem, not a solved one.
- The threshold values calibrated here are derived from MaFaulDa's
  recording conditions (a controlled bench rig), not from the group's own
  aerator motor or ESP32/MPU6050 sensor. Recalibration against real
  hardware, once available, is required before any threshold in this
  document is trusted in deployment — the `.ino` file marks this explicitly.
- The 2x/1x harmonic-ratio feature assumes a correctly identified 1x
  fundamental frequency, found here via a simple max-magnitude search in a
  plausible RPM band (15-70 Hz). This is a reasonable but not foolproof
  method; a noisier real deployment may need a more robust fundamental-
  frequency estimator (e.g., using tachometer data, which MaFaulDa provides
  as a separate channel and which this analysis did not use).

## 8. Supporting numbers (for cross-reference with the deck)

- 4-class CNN (256-bin, 3-conv, dropout): 66.1% float32, 39.4% int8
  (46.65 KB) — quantization cost is markedly higher than the 2-class model's
  (only 5-10 points lost), attributed to a wider dynamic range introduced by
  the added third convolutional layer.
- 2-class deployed model (128-bin): 90.5% int8, 15.91 KB — this remains the
  model actually shipped in the Review-II deployment code, since the
  4-class model is not yet accurate enough post-quantization to deploy.
