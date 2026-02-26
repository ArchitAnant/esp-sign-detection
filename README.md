# esp-sign-detection
![Espressif](https://img.shields.io/badge/espressif-E7352C.svg?style=for-the-badge&logo=espressif&logoColor=white) ![PlatformIO](https://img.shields.io/badge/platformio-%23000.svg?style=for-the-badge&logo=platformio&logoColor=F5822A) ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) 

A low-latency, real-time sign language recognition system powered by **TensorFlow Lite Micro** on the **ESP32**. This project uses a distributed architecture where a laptop performs hand detection (MediaPipe) and streams processed data to an ESP32 for high-speed edge inference.

## Overview

The system recognizes static hand signs (currently optimized for classes **A, Q, and T**) with high precision by offloading computer vision tasks to a host while performing the core AI classification on a microcontroller.

### System Architecture
1.  **Laptop (Python/MediaPipe):** Captures webcam video, detects hand landmarks, crops the hand ROI, and normalizes the image to a $28 \times 28$ grayscale patch.
2.  **Communication:** Streams raw 784-byte packets via **UDP** over Wi-Fi.
3.  **ESP32 (TFLite Micro):** Receives the packet, performs **INT8 Quantized Inference**, and outputs probabilities using a specialized C++ bridge.

---

## Hardware & Software

### Hardware
*   **Microcontroller:** ESP32 (WROOM/DevKit)
*   **Host:** Laptop with Webcam

### Software Stack
*   **Framework:** ESP-IDF (v5.x)
*   **ML Engine:** TensorFlow Lite Micro (optimized for ESP32 via `esp-tflite-micro`)
*   **Build Tool:** PlatformIO
*   **Training:** PyTorch (ResNet-style Architecture)
*   **Inference (Host):** MediaPipe, OpenCV, TensorFlow Lite

---

## Model Specifications

The project uses a custom **TinyRes28_ESP32** architecture designed for the strict SRAM limits of the ESP32.

*   **Input Shape:** $28 \times 28 \times 1$ (Grayscale)
*   **Quantization:** Full INT8 (Per-Channel Symmetric)
*   **Layers:** Depthwise Separable Convolutions + Residual (Skip) Connections.
*   **Memory Footprint:** ~33KB TFLite model, ~120KB Tensor Arena.

---

## Installation & Setup

### Python Client
1.  Install dependencies:
    ```bash
    cd python_cam_client
    pip install -r requirements.txt
    ```
2.  Set your ESP32's IP address in the inference script.
3.  Run the host-side pipeline:
    ```bash
    python img_test.py
    ```

---

## Performance Features

*   **Manual 16-byte Alignment:** Prevents memory allocation crashes in TFLite Micro's `SingleArenaBufferAllocator`.
*   **Softmax Temperature Scaling:** Match laptop confidence levels using temperature-scaled probabilities ($T=0.6364$).
*   **UDP Reliability:** Lightweight header-based packet verification (`0xAABB`).
*   **Flash Optimization:** Model weights stored strictly in Flash (`static const`) to preserve SRAM for the Wi-Fi stack.

---
## Future 
- The PyTorch/TFLite model has a very good confidence score for each class, which drops significalty on the header file model on ESP.

- Currently the model just classifies A, Q and T. We can train for more classes and even try increasing the model complexity to classifiy more classes.

- We can port the Image Capturing + Preprocessing code from python to an SoC but would require another ESP32-Cam board (Out of scope for this project).
