# Vibration-Based Fault Classification of Motors Using Edge-AI

This repository contains the firmware, signal processing pipeline, and machine learning models for an Edge-AI condition monitoring node. Designed to monitor aerator motors in aquaculture, the system captures vibration data and classifies mechanical faults entirely on a low-cost microcontroller without needing a cloud connection.

## 🚀 Project Overview
Unexpected motor failures can cause massive losses in aquaculture. This project replaces expensive, expert-dependent vibration analyzers with a low-cost, intelligent edge node. 

The system samples accelerometer data at 1 kHz, computes a 256-bin log spectrum, and passes the features through a custom 1D Convolutional Neural Network (CNN). The model is fully quantized to `int8` and deployed on an ESP32 using TensorFlow Lite Micro.

## 🛠️ Hardware Setup
* **Microcontroller:** ESP32 Dev Module (240 MHz)
* **Sensor:** MPU6050 Accelerometer (communicating via I2C at 400 kHz)
* **Target Hardware:** Tested on a scaled 3-6V submersible pump prototype.
* **Power Isolation:** The pump and ESP32 are powered by strictly isolated battery packs to prevent motor inrush currents from causing logic brownouts.

## 🧠 Machine Learning Models
Two variations of the model were developed:
1. **4-Class Offline Model:** Trained on a public 22 kW industrial centrifugal pump dataset (Bruinsma et al.). Classifies between *Healthy, Unbalance, Misalignment, and Coupling Degradation*.
2. **2-Class Live Model:** Trained on physical prototype data. Classifies between *Healthy* and *Fault* (loose mount, choked flow) in real-time. 

### Model Performance Metrics (ESP32)
* **Model Size:** 7,112 Bytes (`int8` quantized)
* **Working RAM (Tensor Arena):** ~4.9 KB
* **Inference Speed:** 22.5 ms per 512-sample window (only 4.4% of the window capture time)
* **Accuracy:** 100.0% on live 2-class physical feasibility tests using a 5-window rolling vote.

## ⚙️ How It Works
1. **Windowing:** The ESP32 captures 512 acceleration samples (Y-axis) at exactly 1 kHz.
2. **Feature Extraction:** Mean removal, Hann windowing, and a custom direct DFT calculation output a 256-bin log spectrum.
3. **Inference:** The array is passed to the TFLite Micro model.
4. **Voting:** A rolling mean vote across 5 consecutive windows stabilizes the output to prevent false alarms, outputting the final mechanical state to the serial monitor.

## 📁 Repository Structure
* `/models/` - Keras training scripts, Colab notebooks, and TFLite quantization pipeline.
* `/firmware/` - Arduino/C++ source code for the ESP32 (including feature extraction and TFLM).
* `/data/` - Python scripts for decimating and windowing the raw 20 kHz dataset into 1 kHz chunks.

## 📝 Authors
Divakar V., Lingeshvar R., U. Pherarasan  
*School of Electronics Engineering, VIT Chennai*
