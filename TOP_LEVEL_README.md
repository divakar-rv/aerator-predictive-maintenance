# Cost-Aware Edge-AI Predictive Maintenance of Industrial Aerator Motors

**BECE497J Project-I, School of Electronics Engineering, VIT Chennai**

Lingeshvar R (23BEC1182) · U Pherarasan (23BEC1247) · Divakar V (23BEC1273)
Guide: Arivarasi A

## What this is

An event-triggered, edge-AI system for detecting mechanical faults in
aerator motors used in aquaculture, using intermittent vibration + acoustic
sensor fusion on an ESP32. The goal: catch bearing lockups, imbalance, and
misalignment before they cause dissolved-oxygen crashes and fish mortality,
without the power/bandwidth cost of continuous cloud monitoring.

**Status as of Review-II (30.09.2026): software pipeline validated on real
data; ESP32 deployment code written and compile-verified; hardware not yet
procured.**

## Repository structure

```
notebook/            Colab notebook: data loading, training, quantization
models/               Trained .tflite models (2-class and 4-class variants)
esp32_deployment/      Arduino sketch, model converter, deployment README
docs/                  Full technical write-up of the wake-trigger investigation
review_decks/          Presentation decks for Review-II
```

## Key results (real data, not synthetic)

Dataset: [MaFaulDa](https://www02.smt.ufrj.br/~offshore/mfs/page_01.html)
(Machinery Fault Database), a real bench-rig vibration/acoustic dataset,
loaded via a [Kaggle mirror](https://www.kaggle.com/datasets/vuxuancu/mafaulda-full).

| Model | Task | Accuracy | Size |
|---|---|---|---|
| Vibration-only (float32, 250 bins) | Normal vs Imbalance | 95.2% | — |
| Vibration+mic fusion (float32) | Normal vs Imbalance | 95.2% | — |
| **Deployed model** (int8, 250 bins) | Normal vs Imbalance | **85.7%** | **70.62 KB** |
| Vibration-only int8 (128 bins) | Normal vs Imbalance | 64.8-65.7%** (rejected — unreliable) | 15.91 KB |
| Fusion int8 | Normal vs Imbalance | 61-77% (unstable) | ~29 KB — **rejected** |
| 4-class CNN (float32, 256 bins) | Normal/Imbalance/Misalignment/Bearing | 56.7-66.1%* | — |
| 4-class CNN int8 | same | 39.4% | 46.65 KB |

*\*Varies run-to-run despite fixed seeds — see `docs/wake_trigger_investigation.md` §7b.*
*\*\*The 128-bin model looked promising at 90.5% in initial testing but was
found unreliable across repeated verification — see §6b for the full
investigation. This is why the final deployed model exceeds the original
30 KB target: a real, reproducible 85.7% at 70.62 KB was chosen over an
unreliable 90.5%-or-65% coin-flip at 15.91 KB.*

**Two-stage wake-trigger system** (the core "intermittent" design claim):
skips **42.8%** of full CNN inference calls with no loss in overall accuracy
(61.1% two-stage vs. 56.7-66.1% always-on baseline), by gating on two cheap
spectral features (peakiness, 2x/1x harmonic ratio) computed directly on the
FFT the device already needs. Full derivation, including two honestly-
reported negative results (ground-truth tachometer frequency, quantization-
aware training), in `docs/wake_trigger_investigation.md`.

**Known limitation, stated plainly**: the wake-trigger only catches
Misalignment ~40% of the time (vs. 0% for a naive RMS-only trigger) — this
remains the hardest fault class to detect cheaply, and is an open problem,
not a solved one.

## What's real vs. what's pending

**Validated on real data:** all modeling numbers above, using a proper
file-wise train/test split (no data leakage), real MaFaulDa recordings, and
reproducible (seeded) pipelines.

**Written but not hardware-tested:** the ESP32 deployment sketch compiles
cleanly (29% flash, 17% RAM on an ESP32 Dev Module target) but has never run
on physical hardware. No ESP32/MPU6050/INMP441 has been procured. Wake-gate
thresholds are calibrated on MaFaulDa's bench-rig data, not this project's
actual motor — real-hardware recalibration is required before deployment.

**Not yet measured:** on-device inference latency, real power/battery
consumption. ESP32 dev-board deep-sleep current in practice is known to be
far worse than datasheet specs (5-20 mA vs ~10 µA) due to onboard
regulators — no battery-life figure is claimed anywhere in this project
until it's actually measured.

## Reproducing the results

1. Open `notebook/training_pipeline.ipynb` in Google Colab
2. Requires a Kaggle account + API key (see notebook for setup)
3. Run cells top to bottom — dataset download (~13GB), preprocessing,
   training, quantization
4. To flash to real hardware once available: see `esp32_deployment/README.md`

## License / academic context

Submitted as part of BECE497J Project-I coursework at VIT Chennai. See
`review_decks/` for the full literature review (15 verified references) and
`docs/wake_trigger_investigation.md` for the technical novelty contribution.
