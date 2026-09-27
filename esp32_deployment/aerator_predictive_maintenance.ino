/*
  Aerator Motor Predictive Maintenance — ESP32 Deployment Sketch
  BECE497J Project-I, VIT Chennai

  STATUS: Written and compile-verified in Arduino IDE (ESP32 Dev Module,
  383829 bytes flash / 58844 bytes RAM at the time of the 128-bin model —
  recheck after this 250-bin update), but UNTESTED ON REAL HARDWARE (no
  ESP32/MPU6050/INMP441 procured yet as of writing). Do not present this as
  "deployed and working" — present it as "deployment code ready and
  compile-verified, pending hardware validation."

  Deploys the FINAL deployment model: vibration-only, 250 FFT bins, int8
  quantized, 70.62 KB, 2-class (Normal / Imbalance), 85.7% int8 test
  accuracy on real MaFaulDa data — confirmed reproducibly across multiple
  independent quantization runs.

  IMPORTANT: an earlier, smaller candidate (128 bins, 15.91 KB, under the
  original 30KB target) was tried first and looked promising at 90.5% int8
  in initial testing, but proved UNRELIABLE under further scrutiny: four
  separate quantization attempts (two different training runs x two
  calibration strategies) all collapsed to 64.8-65.7% int8 accuracy, biased
  toward the majority class, despite float32 staying a stable 90.5% each
  time. This 250-bin/70.62KB model was selected instead because it
  quantizes reliably and reproducibly, even though it exceeds the original
  size target. Full investigation in wake_trigger_investigation.md.

  Architecture:
    1. Deep-sleep by default.
    2. Wake on timer (duty-cycled sampling) — true "anomaly wake-up" via a
       hardware interrupt/comparator is a further Phase-2 step; this version
       uses periodic wake, which is the honest, currently-implementable version.
    3. Read 1 second of vibration data from MPU6050 at ~1 kHz (128 windowed
       samples used for FFT after decimation logic matches training pipeline).
    4. STAGE 1 (cheap gate): compute TWO spectral features in plain C, no
       TFLite needed — (a) peakiness (max/mean of FFT magnitude) and
       (b) 2x/1x harmonic ratio (magnitude at ~2x the dominant rotational
       frequency, relative to the magnitude at that dominant frequency).
       Wake for full inference if EITHER feature deviates from a calibrated
       Normal baseline. This two-feature design was validated on real
       MaFaulDa data in Colab: peakiness alone catches "loud" faults
       (Imbalance, Bearing) but completely misses Misalignment (0% detection
       in testing), because Misalignment is amplitude-quiet; the harmonic
       ratio was added specifically to catch that case (raising detection to
       ~40% in end-to-end simulation — still the weakest case, not solved).
       See wake_trigger_investigation.md for the full derivation and honest
       limitations of this design.
    5. STAGE 2 (full inference): if Stage 1 flags a possible anomaly, run the
       quantized 1D-CNN via TensorFlow Lite Micro for the actual Normal/
       Imbalance classification, and report over WiFi/BLE/Serial as configured.

  REQUIRED LIBRARIES (install via Arduino Library Manager):
    - Chirale_TensorFlowLite      (TFLite Micro for ESP32)
    - Adafruit MPU6050            (+ Adafruit Unified Sensor, Adafruit BusIO)
    - arduinoFFT                  (for the Stage-1 lightweight FFT; TFLite
                                    model does its own internal FFT-equivalent
                                    via the dense/conv layers on raw FFT input
                                    we compute here in C, matching the Python
                                    feature extraction exactly)

  BEFORE FLASHING:
    - Generate model_data.h from your trained tflite_small model (see
      convert_model.py in this folder) and place it alongside this .ino.
    - Confirm MPU6050 I2C address (default 0x68) and wiring (SDA/SCL).
    - Confirm your real decimation factor / sample rate match training
      (fs=1000 Hz after decimation from raw MPU6050 output rate).
*/

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <arduinoFFT.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"   // generated from tflite_small — see convert_model.py

// ---------------------------------------------------------------------------
// CONFIG — must match the Python training pipeline exactly, or predictions
// will be meaningless even if the code "runs".
// ---------------------------------------------------------------------------
#define SAMPLE_RATE_HZ      1000      // post-decimation rate used in training
#define WINDOW_SIZE         1000      // 1-second window, pre-FFT (time domain)
#define FFT_BINS            250       // matches the 70.62 KB deployed model
                                       // (Xv_train_cnn/Xv_test_cnn, 250-bin FFT).
                                       // NOTE: an earlier 128-bin/15.91KB model was
                                       // tried first but its int8 quantization proved
                                       // unreliable (4 separate attempts collapsed to
                                       // 64.8-65.7%, biased toward the majority class,
                                       // despite stable 90.5% float32 each time) — see
                                       // wake_trigger_investigation.md. This 250-bin
                                       // model exceeds the original 30KB size target
                                       // but quantizes reliably and reproducibly to
                                       // 85.7% int8, confirmed across multiple runs.
#define NUM_CLASSES         2         // Normal=0, Imbalance=1
#define MPU_RAW_SAMPLE_HZ   1000      // sample MPU6050 directly at target rate
                                       // (simpler than decimating on-device;
                                       // revisit if MPU6050 can't sustain this
                                       // reliably over I2C — measure, don't assume)

// Stage-1 wake-gate thresholds — CALIBRATED on the actual deployed model's
// own pipeline (2-class, 250-bin, real MaFaulDa Normal-class training
// windows, N_SIGMA=2.0, seed=42). Sanity-checked: 90.8% of held-out
// Imbalance windows correctly trigger a wake with these values.
// ONE CAVEAT REMAINS: these are still MaFaulDa bench-rig numbers, not this
// project's real motor/sensor noise floor. Recalibrate once real hardware
// exists — swap in real idle-motor recordings for the "Normal" calibration
// set and rerun the same calibration cell (see wake_trigger_investigation.md).
#define PEAKINESS_WAKE_THRESHOLD     33.60f
#define HARMONIC_RATIO_LOW           0.0000f
#define HARMONIC_RATIO_HIGH          0.6505f
#define FUNDAMENTAL_FREQ_MIN_HZ      15.0f   // search band for the 1x rotational peak
#define FUNDAMENTAL_FREQ_MAX_HZ      70.0f   // matches the Colab study's search_band

Adafruit_MPU6050 mpu;

double vReal[WINDOW_SIZE];
double vImag[WINDOW_SIZE];
ArduinoFFT<double> FFT = ArduinoFFT<double>(vReal, vImag, WINDOW_SIZE, (double)SAMPLE_RATE_HZ);

// TFLite Micro globals
namespace {
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;
  constexpr int kTensorArenaSize = 90 * 1024;  // bumped from 20KB for the
                                                // larger 70.62KB deployed
                                                // model (more filters/units
                                                // than the earlier 15.91KB
                                                // attempt). Still a guess,
                                                // not a profiled value — if
                                                // AllocateTensors() fails,
                                                // increase further; Serial
                                                // output will report it.
  uint8_t tensor_arena[kTensorArenaSize];
}

// ---------------------------------------------------------------------------
// SETUP
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("Aerator predictive maintenance node booting...");

  if (!mpu.begin()) {
    Serial.println("FATAL: MPU6050 not found. Check wiring/I2C address.");
    while (1) delay(1000);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);   // pick range matching
                                                    // expected motor vibration
                                                    // amplitude; verify against
                                                    // real bench data, this is
                                                    // a reasonable default only
  mpu.setFilterBandwidth(MPU6050_BAND_260_HZ);

  if (!setupTFLite()) {
    Serial.println("FATAL: TFLite Micro setup failed.");
    while (1) delay(1000);
  }

  Serial.println("Setup complete. Entering main loop.");
}

// ---------------------------------------------------------------------------
// MAIN LOOP — periodic wake, sample, gate, (maybe) classify, sleep
// ---------------------------------------------------------------------------
void loop() {
  sampleVibrationWindow();          // fills vReal[] with WINDOW_SIZE samples
  runFFT();                          // FFT in place on vReal/vImag

  float peakiness = computePeakiness(vReal, FFT_BINS);
  float fundamental_freq;
  float harmonicRatio = computeHarmonicRatio(vReal, FFT_BINS, &fundamental_freq);

  Serial.print("Stage-1 peakiness: "); Serial.print(peakiness);
  Serial.print(" | 2x/1x ratio: "); Serial.print(harmonicRatio);
  Serial.print(" (1x @ "); Serial.print(fundamental_freq); Serial.println(" Hz)");

  bool peak_triggers = peakiness > PEAKINESS_WAKE_THRESHOLD;
  bool ratio_triggers = (harmonicRatio < HARMONIC_RATIO_LOW) || (harmonicRatio > HARMONIC_RATIO_HIGH);

  if (!peak_triggers && !ratio_triggers) {
    Serial.println("Stage 1: within Normal baseline on both features -> skipping full inference, sleeping.");
  } else {
    Serial.print("Stage 1: triggered by");
    if (peak_triggers) Serial.print(" [peakiness]");
    if (ratio_triggers) Serial.print(" [harmonic-ratio]");
    Serial.println(" -> running full CNN inference.");
    int predicted_class = runInference(vReal, FFT_BINS);
    reportResult(predicted_class);
  }

  enterLightSleep();   // NOTE: true deep-sleep with peripheral shutdown is a
                        // Phase-2 power-optimization step; light sleep is the
                        // safe default until real current-draw is measured on
                        // this specific board (see project notes on ESP32
                        // dev-board deep-sleep current being far worse than
                        // datasheet spec in practice).
}

// ---------------------------------------------------------------------------
// Sensor sampling
// ---------------------------------------------------------------------------
void sampleVibrationWindow() {
  sensors_event_t a, g, temp;
  const unsigned long period_us = 1000000UL / MPU_RAW_SAMPLE_HZ;
  unsigned long next = micros();

  for (int i = 0; i < WINDOW_SIZE; i++) {
    mpu.getEvent(&a, &g, &temp);
    // Training used column index 2 of the MaFaulDa CSV (radial accelerometer
    // axis). MPU6050's equivalent single axis is a design choice — confirm
    // orientation against your physical mount before trusting class labels.
    vReal[i] = a.acceleration.y;
    vImag[i] = 0.0;

    next += period_us;
    long wait = next - micros();
    if (wait > 0) delayMicroseconds(wait);
    // If 'wait' is consistently negative in real testing, MPU6050 I2C reads
    // can't keep up with 1 kHz and the sample rate needs to be re-measured
    // and the training pipeline re-matched to the ACHIEVED rate, not the
    // requested one.
  }
}

void runFFT() {
  FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);
  FFT.compute(FFTDirection::Forward);
  FFT.complexToMagnitude();
  // vReal[0..WINDOW_SIZE/2] now holds magnitude spectrum; we only use the
  // first FFT_BINS bins downstream, matching compute_fft_batch_small() in
  // the training notebook (mag[:128]).
}

// ---------------------------------------------------------------------------
// Stage 1: cheap wake-gate (no TFLite call)
// ---------------------------------------------------------------------------
float computePeakiness(double* spectrum, int n_bins) {
  double maxVal = 0, sumVal = 0;
  for (int i = 0; i < n_bins; i++) {
    if (spectrum[i] > maxVal) maxVal = spectrum[i];
    sumVal += spectrum[i];
  }
  double mean = sumVal / n_bins;
  return (float)(maxVal / (mean + 1e-9));
}

// Mirrors the Python two_stage_features() harmonic-ratio logic exactly:
// find the dominant (1x) peak within a plausible rotational-speed band,
// then measure the magnitude near 2x that frequency relative to it.
// bin_hz must match the frequency resolution of the spectrum passed in
// (SAMPLE_RATE_HZ / WINDOW_SIZE, NOT SAMPLE_RATE_HZ / n_bins — n_bins here
// is just how many of the full spectrum's bins we kept, matching Python's
// mag[:n_bins] truncation after a full-length FFT).
float computeHarmonicRatio(double* spectrum, int n_bins, float* out_fundamental_hz) {
  const float bin_hz = (float)SAMPLE_RATE_HZ / (float)WINDOW_SIZE;

  int band_start = (int)(FUNDAMENTAL_FREQ_MIN_HZ / bin_hz);
  int band_end   = (int)(FUNDAMENTAL_FREQ_MAX_HZ / bin_hz);
  if (band_end >= n_bins) band_end = n_bins - 1;

  int f1_idx = band_start;
  double f1_mag = spectrum[band_start];
  for (int i = band_start; i <= band_end; i++) {
    if (spectrum[i] > f1_mag) {
      f1_mag = spectrum[i];
      f1_idx = i;
    }
  }
  float f1_hz = f1_idx * bin_hz;
  if (out_fundamental_hz) *out_fundamental_hz = f1_hz;

  float f2_target_hz = 2.0f * f1_hz;
  int f2_idx = (int)roundf(f2_target_hz / bin_hz);
  if (f2_idx >= n_bins) f2_idx = n_bins - 1;
  if (f2_idx < 0) f2_idx = 0;
  double f2_mag = spectrum[f2_idx];

  return (float)(f2_mag / (f1_mag + 1e-9));
}

// ---------------------------------------------------------------------------
// Stage 2: TFLite Micro setup + inference
// ---------------------------------------------------------------------------
bool setupTFLite() {
  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  // NOTE: exact op set depends on what your specific tflite_small graph
  // contains. Run `xxd`-free introspection (Python: interpreter.get_
  // signature_list() / manually inspect the TFLite graph) BEFORE flashing,
  // and add/remove resolver.Add*() calls to match exactly what the converted
  // model actually uses, or MicroInterpreter's AllocateTensors() will fail.

  const tflite::Model* model = tflite::GetModel(aerator_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch.");
    return false;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, nullptr, nullptr);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed — increase kTensorArenaSize.");
    return false;
  }

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  Serial.print("Model input shape: [");
  for (int i = 0; i < model_input->dims->size; i++) {
    Serial.print(model_input->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println("]");

  return true;
}

int runInference(double* spectrum, int n_bins) {
  // Quantize float spectrum -> int8 using the SAME scale/zero-point the
  // Python converter chose. These two numbers MUST be copied from your
  // actual Colab output (input_details[0]['quantization']) — do not guess.
  const float input_scale = 0.00561093632131815f;
  const int input_zero_point = -128;

  for (int i = 0; i < n_bins; i++) {
    int32_t q = (int32_t)roundf((float)spectrum[i] / input_scale) + input_zero_point;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    model_input->data.int8[i] = (int8_t)q;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke() failed.");
    return -1;
  }

  // Find argmax over NUM_CLASSES outputs (also int8-quantized — dequantize
  // for readability, but argmax works directly on quantized values too
  // since the affine transform is monotonic).
  int best_class = 0;
  int8_t best_val = model_output->data.int8[0];
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (model_output->data.int8[i] > best_val) {
      best_val = model_output->data.int8[i];
      best_class = i;
    }
  }
  return best_class;
}

// ---------------------------------------------------------------------------
// Reporting + power management
// ---------------------------------------------------------------------------
void reportResult(int predicted_class) {
  const char* labels[NUM_CLASSES] = {"Normal", "Imbalance"};
  Serial.print("Prediction: ");
  Serial.println(predicted_class >= 0 ? labels[predicted_class] : "ERROR");

  // TODO Phase 2: send alert over WiFi/BLE/MQTT only when predicted_class
  // != Normal, matching the "wireless alert only if a fault is detected"
  // design from the Review-I architecture diagram. Not implemented here —
  // needs a decision on transport (WiFi vs BLE) which affects power budget
  // and hasn't been made yet.
}

void enterLightSleep() {
  // Placeholder duty-cycle sleep. Replace with esp_deep_sleep_start() once
  // real current-draw measurements justify deep sleep's wake-up latency
  // cost for your specific duty-cycle requirement (see project notes: dev
  // boards draw far more in "deep sleep" than the datasheet due to onboard
  // regulators/USB bridge — measure this board specifically before relying
  // on any assumed battery-life number).
  delay(5000);  // 5 s between samples — arbitrary placeholder, tune once
                // hardware exists and you know real inference+sample timing.
}
