# ESP32 Deployment — Aerator Predictive Maintenance

**Status: compile-verified (against the earlier 128-bin model — recompile
after the 250-bin update below to get fresh numbers), not yet tested on
hardware.** No ESP32/MPU6050/INMP441 has been procured as of this writing,
so nothing here has run on a real device. It HAS been successfully compiled
in Arduino IDE 2.3.10 targeting "ESP32 Dev Module" (Arduino ESP32 core,
Chirale_TensorFlowLite 2.0.0, arduinoFFT, Adafruit MPU6050) with no errors:

```
Sketch uses 383829 bytes (29%) of program storage space. Maximum is 1310720 bytes.
Global variables use 58844 bytes (17%) of dynamic memory, leaving 268836 bytes for local variables.
```

This was measured against the model actually deployed at the time (see
important note below) — the model, FFT size, and tensor arena have since
changed, so re-verify by compiling again after updating `model_data.h`.

This confirms the code is syntactically and structurally valid for the
target chip and fits comfortably within flash/RAM budget — it does not
confirm correct runtime behavior, sensor timing, or classification
accuracy on real hardware.

**Known fixes required to get this compiling** (documented here in case the
same library versions cause the same issues again):
- The installed `Chirale_TensorFlowLite` library's main header is
  `TensorFlowLite_ESP32.h`, not the generic `TensorFlowLite.h` used by some
  other TFLite Micro Arduino ports.
- `tflite::MicroInterpreter`'s constructor in this library version requires
  two additional trailing arguments (`ErrorReporter*`, `MicroResourceVariables*`
  or similar) beyond the 4-argument form used in some TFLite Micro
  documentation/examples — pass `nullptr, nullptr` if unused.
- Keep exactly one `.ino` file per sketch folder; Arduino IDE compiles every
  `.ino` in the folder together, so a stray duplicate causes confusing
  "already declared" or unrelated missing-symbol errors.

## IMPORTANT: deployed model changed after further testing

The originally-deployed model (128-bin, 15.91 KB, reported at 90.5% int8 in
the Review-II deck) was found to be **unreliable under int8 quantization**
during final verification — four separate quantization attempts all
collapsed to 64.8-65.7% accuracy with a majority-class bias, despite stable
90.5% float32 accuracy each time. Full investigation and root-cause analysis
in `wake_trigger_investigation.md` §6b.

**The sketch now deploys the 250-bin, 70.62 KB model instead**, which
quantizes reliably and reproducibly to **85.7% int8 accuracy** (confirmed
across two independent runs, sane confusion matrix, no majority-class
collapse). This exceeds the original 30 KB size target, but a real,
reproducible 85.7% is a stronger basis for deployment than an unreliable
smaller model.

## What's deployed

The **final deployment model**: vibration-only, 250 FFT bins, int8
quantized, 70.62 KB, 2 classes (Normal / Imbalance), 85.7% int8 test
accuracy on real MaFaulDa data (file-wise split, seeded, reproducible,
confirmed across multiple independent quantization runs).

The wake-gate logic (Stage 1) implements the two-feature design derived in
`wake_trigger_investigation.md`: spectral peakiness + 2x/1x harmonic ratio,
thresholds recalibrated for this 250-bin model (90.8% Imbalance wake-rate
sanity check). The same design was validated end-to-end on real 4-class
data (a different, 256-bin model) to skip 42.8% of full-inference calls
with no accuracy loss — though Misalignment detection remains weak (~40%
catch rate) even with both features, an open problem documented there.

The multimodal fusion model and the 4-class model are **not** wired up here —
fusion int8 was unstable in testing (61–77% across runs) and was rejected;
the 4-class model's int8 accuracy dropped to 39.4% and needs more work
before it's worth deploying. Swap-in points for both are marked as `// TODO
Phase 2` in the .ino once they're ready.

## Files

- `aerator_predictive_maintenance.ino` — main sketch
- `convert_model.py` — run in Colab to generate `model_data.h` and print the
  exact quantization constants the .ino needs
- `model_data.h` — **not included here**; generate it yourself by running
  `convert_model.py` in the same Colab session as your trained `tflite_small`

## Before you flash — three things you MUST fill in

1. **`model_data.h`** — generate via `convert_model.py`, place next to the
   `.ino`.
2. **`input_scale` / `input_zero_point`** in `runInference()` — the script
   above prints the exact values from your actual model. The placeholders
   in the `.ino` (`0.0f` / `0`) will produce meaningless predictions if left
   as-is.
3. **`MicroMutableOpResolver` op list** in `setupTFLite()` — must match what
   your specific model graph actually contains. The script attempts to list
   these automatically; cross-check against `tensorflow/lite/builtin_ops.h`.

## Required libraries (Arduino Library Manager)

- `Chirale_TensorFlowLite` (TFLite Micro for ESP32)
- `Adafruit MPU6050` + `Adafruit Unified Sensor` + `Adafruit BusIO`
- `arduinoFFT`

## Wiring (MPU6050, I2C default)

| MPU6050 | ESP32 |
|---|---|
| VCC | 3.3V |
| GND | GND |
| SDA | GPIO 21 (default) |
| SCL | GPIO 22 (default) |

INMP441 (I2S mic) wiring is **not yet implemented** in this sketch — the
deployed model is vibration-only, so the mic isn't wired in. Add this only
once/if a fusion model is actually validated and worth deploying.

## Known open issues / things to verify on real hardware

- **Sample rate achievability**: the sketch assumes MPU6050 can sustain
  ~1 kHz reads over I2C. If it can't (very possible — I2C overhead is real),
  the code logs negative wait times; you'll need to either reduce
  `MPU_RAW_SAMPLE_HZ` and re-run the Python training pipeline at the
  achieved rate, or switch the MPU6050 to its internal FIFO/DMA mode instead
  of polling.
- **Sensor axis orientation**: training used one axis of the MaFaulDa CSV
  (their "radial accelerometer"). Which physical MPU6050 axis corresponds to
  the equivalent orientation on your actual motor mount is a real
  installation decision, not a code decision — verify against a known-fault
  test run before trusting classifications.
- **Tensor arena size** (`kTensorArenaSize = 20KB`) is a guess based on model
  size, not a profiled value. If `AllocateTensors()` fails, increase it;
  Serial output will report the failure clearly.
- **Wake-gate thresholds**: the sketch implements the full two-feature gate
  from `wake_trigger_investigation.md` — spectral peakiness AND 2x/1x
  harmonic ratio, wake if either deviates from baseline. **These are
  calibrated on the actual deployed model's own pipeline** (2-class, 250-bin,
  real MaFaulDa Normal-class training windows, N_SIGMA=2.0): peakiness
  threshold 33.60, harmonic ratio range [0.0, 0.651]. Sanity-checked against
  held-out Imbalance windows (90.8% correctly trigger a wake). **What's still
  outstanding**: these are MaFaulDa bench-rig numbers, not this project's real
  motor/sensor noise floor — recalibrate once real hardware exists by
  swapping in real idle-motor recordings for the "Normal" calibration set and
  rerunning the same calibration cell.
- **True deep sleep vs. light sleep**: the sketch uses a simple `delay()`
  placeholder, not `esp_deep_sleep_start()`. Real ESP32 dev boards draw far
  more current in "deep sleep" than the datasheet spec (5–20 mA typical vs.
  ~10 µA ideal) due to onboard voltage regulators and USB-UART chips — no
  battery-life claim should be made until this is actually measured on your
  specific board.
- **Wireless alerting** (WiFi/BLE) is not implemented — transport choice
  affects the power budget and hasn't been decided yet.

## What this file is good for right now

- Showing a functioning, reasoned deployment architecture in a review/demo,
  explicitly labeled as "ready, pending hardware."
- A concrete starting point to debug against once hardware exists, rather
  than starting from a blank sketch.
- Evidence of Phase 2 progress that doesn't require hardware spend to show.
