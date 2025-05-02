<div align="center">
  <a href="https://github.com/OEvortex/Webscout">
    <img src="https://img.shields.io/badge/WebScout-GGUF%20Converter-blue?style=for-the-badge&logo=python&logoColor=white" alt="GGUF Converter Logo">
  </a>

  <h1>GGUF Converter</h1>

  <p><strong>Convert Hugging Face models to GGUF format with advanced quantization options</strong></p>

  <p>
    Transform large language models from Hugging Face into optimized GGUF format for efficient inference on consumer hardware. 
    Balance size, speed, and quality with multiple quantization methods.
  </p>

  <!-- Badges -->
  <p>
    <a href="https://github.com/ggerganov/llama.cpp"><img src="https://img.shields.io/badge/Powered%20by-llama.cpp-orange?style=flat-square" alt="Powered by llama.cpp"></a>
    <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/Hugging%20Face-compatible-yellow?style=flat-square" alt="Hugging Face compatible"></a>
    <a href="#"><img src="https://img.shields.io/badge/GPU-acceleration-green?style=flat-square" alt="GPU acceleration"></a>
  </p>
</div>

<hr/>

## 📋 Table of Contents

- [🌟 Features](#-features)
- [⚙️ Installation](#️-installation)
- [🛠️ Basic Usage](#️-basic-usage)
- [🧩 Advanced Options](#-advanced-options)
- [📊 Quantization Methods](#-quantization-methods)
- [📏 Size & Quality Comparison](#-size--quality-comparison)
- [📦 Hardware Requirements](#-hardware-requirements)
- [⚡ Examples](#-examples)
- [🔍 Troubleshooting](#-troubleshooting)
- [🧠 Technical Details](#-technical-details)

<hr/>

## 🌟 Features

<details open>
<summary><b>Core Capabilities</b></summary>
<p>

* **Multiple Quantization Methods**: Support for various precision levels from 2-bit to 16-bit floating point
* **Importance Matrix Quantization**: Enhanced precision by focusing bits on the most important weights
* **Model Splitting**: Split large models into manageable chunks for easier distribution
* **Hardware Acceleration Detection**: Automatically detects and utilizes CUDA, Metal, OpenCL, Vulkan, and ROCm
* **Hugging Face Integration**: Direct download from and upload to Hugging Face repositories
* **README Generation**: Automatically creates documentation for your quantized models
</p>
</details>

<hr/>

## ⚙️ Installation

<div class="installation-box">
<p>The GGUF Converter is included with the WebScout package:</p>

```bash
pip install -U webscout
```
</div>

<hr/>

## 🛠️ Basic Usage

The simplest way to convert a model is with the default settings:

```bash
python -m webscout.Extra.gguf convert -m "organization/model-name"
```

This will:
1. Download the model from Hugging Face
2. Convert it to GGUF format with q4_k_m quantization (a good balance of size and quality)
3. Save the converted model in your current directory

<hr/>

## 🧩 Advanced Options

<details open>
<summary><b>Command Reference</b></summary>
<p>

The full command syntax is:

```
python -m webscout.Extra.gguf convert [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model-id` | The HuggingFace model ID (e.g., 'OEvortex/HelpingAI-Lite-1.5T') | **Required** |
| `-u, --username` | Your HuggingFace username for uploads | None |
| `-t, --token` | Your HuggingFace API token for uploads | None |
| `-q, --quantization` | Comma-separated quantization methods | "q4_k_m" |
| `-i, --use-imatrix` | Use importance matrix for quantization | False |
| `--train-data` | Training data file for imatrix quantization | None |
| `-s, --split-model` | Split the model into smaller chunks | False |
| `--split-max-tensors` | Maximum number of tensors per file when splitting | 256 |
| `--split-max-size` | Maximum file size when splitting (e.g., '256M', '5G') | None |
</p>
</details>

<details>
<summary><b>Multiple Quantization Methods</b></summary>
<p>

Apply multiple quantization methods at once:

```bash
python -m webscout.Extra.gguf convert -m "organization/model-name" -q "q4_k_m,q5_k_m"
```

This will create two versions of the model with different quantization methods.
</p>
</details>

<details>
<summary><b>Uploading to Hugging Face</b></summary>
<p>

Convert and upload the model to your Hugging Face account:

```bash
python -m webscout.Extra.gguf convert -m "organization/model-name" -u "your-username" -t "your-token"
```

This will create a new repository in your account named `model-name-GGUF` containing the converted model.
</p>
</details>

<details>
<summary><b>Importance Matrix Quantization</b></summary>
<p>

Use importance matrix for more efficient quantization:

```bash
python -m webscout.Extra.gguf convert -m "organization/model-name" -i --train-data "train_data.txt"
```

Importance matrix helps focus more bits on weights that matter most for the model's performance.
</p>
</details>

<details>
<summary><b>Model Splitting</b></summary>
<p>

Split large models for easier distribution:

```bash
# Split by number of tensors
python -m webscout.Extra.gguf convert -m "organization/model-name" -s --split-max-tensors 256

# Split by file size
python -m webscout.Extra.gguf convert -m "organization/model-name" -s --split-max-size "2G"
```

This is useful for very large models that may be difficult to distribute as a single file.
</p>
</details>

<hr/>

## 📊 Quantization Methods

<details open>
<summary><b>Standard Methods</b></summary>
<p>

| Method | Description |
|--------|-------------|
| `fp16` | 16-bit floating point - maximum accuracy, largest size |
| `q2_k` | 2-bit quantization (smallest size, lowest accuracy) |
| `q3_k_l` | 3-bit quantization (large) - balanced for size/accuracy |
| `q3_k_m` | 3-bit quantization (medium) - good balance for most use cases |
| `q3_k_s` | 3-bit quantization (small) - optimized for speed |
| `q4_0` | 4-bit quantization (version 0) - standard 4-bit compression |
| `q4_1` | 4-bit quantization (version 1) - improved accuracy over q4_0 |
| `q4_k_m` | 4-bit quantization (medium) - balanced for most models |
| `q4_k_s` | 4-bit quantization (small) - optimized for speed |
| `q5_0` | 5-bit quantization (version 0) - high accuracy, larger size |
| `q5_1` | 5-bit quantization (version 1) - improved accuracy over q5_0 |
| `q5_k_m` | 5-bit quantization (medium) - best balance for quality/size |
| `q5_k_s` | 5-bit quantization (small) - optimized for speed |
| `q6_k` | 6-bit quantization - highest accuracy, larger size |
| `q8_0` | 8-bit quantization - maximum accuracy, largest size |
</p>
</details>

<details>
<summary><b>Importance Matrix Methods</b></summary>
<p>

| Method | Description |
|--------|-------------|
| `iq3_m` | 3-bit imatrix quantization (medium) - balanced importance-based |
| `iq3_xxs` | 3-bit imatrix quantization (extra extra small) - maximum compression |
| `q4_k_m` | 4-bit imatrix quantization (medium) - balanced importance-based |
| `q4_k_s` | 4-bit imatrix quantization (small) - optimized for speed |
| `iq4_nl` | 4-bit imatrix quantization (non-linear) - best accuracy for 4-bit |
| `iq4_xs` | 4-bit imatrix quantization (extra small) - maximum compression |
| `q5_k_m` | 5-bit imatrix quantization (medium) - balanced importance-based |
| `q5_k_s` | 5-bit imatrix quantization (small) - optimized for speed |
</p>
</details>

<hr/>

## 📏 Size & Quality Comparison

> **TIP:**
> When choosing a quantization method, consider the tradeoff between model size and quality. Here's a quick guide:

<div class="comparison-table">

### 1. Maximum Quality (largest size)
- **fp16**: 100% of original size, best quality
- **q8_0**: 50% of original size, nearly identical to fp16

### 2. Balanced Quality/Size
- **q5_k_m with imatrix**: 31% of original size, excellent quality
- **q4_k_m with imatrix**: 25% of original size, good quality for most use cases

### 3. Minimum Size (reduced quality)
- **q3_k_s**: 18% of original size, acceptable for some tasks
- **q2_k**: 12% of original size, significantly reduced quality
</div>

<hr/>

## 📦 Hardware Requirements

Hardware requirements vary based on quantization method and model size:

<details open>
<summary><b>Memory Requirements</b></summary>
<p>

| Quantization | RAM Required |
|--------------|--------------|
| fp16 | ~2x model size |
| q8_0 | ~1x model size |
| q4_k_m | ~0.5x model size |
| q2_k | ~0.25x model size |

For example, a 7B parameter model requires:
- fp16: ~14GB RAM
- q4_k_m: ~3.5GB RAM
</p>
</details>

<details>
<summary><b>Hardware Acceleration</b></summary>
<p>

The converter automatically detects and utilizes:
- **CUDA** for NVIDIA GPUs
- **Metal** for Apple Silicon and AMD GPUs on macOS
- **OpenCL** for cross-platform GPU acceleration
- **Vulkan** for cross-platform GPU acceleration
- **ROCm** for AMD GPUs on Linux

If no acceleration is available, the converter will use CPU-only mode.
</p>
</details>

> **NOTE:**
> **GPU acceleration is highly recommended** for converting larger models (13B+).

<hr/>

## ⚡ Examples

<details open>
<summary><b>Basic Conversion with Upload</b></summary>
<p>

```bash
python -m webscout.Extra.gguf convert \
    -m "mistralai/Mistral-7B-Instruct-v0.2" \
    -q "q4_k_m" \
    -u "your-username" \
    -t "your-token"
```

This will convert Mistral-7B to q4_k_m quantization and upload it to your Hugging Face account.
</p>
</details>

<details>
<summary><b>Multiple Quantizations with Importance Matrix</b></summary>
<p>

```bash
python -m webscout.Extra.gguf convert \
    -m "mistralai/Mistral-7B-Instruct-v0.2" \
    -q "q4_k_m,q5_k_m" \
    -i \
    --train-data "my_training_data.txt"
```

This will create two versions of the model with different quantizations, both using importance matrix.
</p>
</details>

<details>
<summary><b>Split Large Model</b></summary>
<p>

```bash
python -m webscout.Extra.gguf convert \
    -m "meta-llama/Llama-2-70b-chat-hf" \
    -q "q4_k_m" \
    -s \
    --split-max-size "4G"
```

This will split the large 70B model into multiple files, each no larger than 4GB.
</p>
</details>

<hr/>

## 🔍 Troubleshooting

<details>
<summary><b>Missing Dependencies</b></summary>
<p>

```
Error: Missing required dependencies: git, cmake
```

**Solution:** Install the required system dependencies:

- **Ubuntu/Debian:** `sudo apt install git cmake python3-dev build-essential`
- **macOS:** `brew install git cmake`
- **Windows:** Install Git and CMake from their respective websites

For hardware acceleration, install relevant drivers (CUDA, ROCm, etc.)
</p>
</details>

<details>
<summary><b>Out of Memory</b></summary>
<p>

```
Error: CUDA out of memory
```

**Solutions:**
1. Try a lower precision quantization method: `q3_k_s` or `q2_k`
2. Enable model splitting with `-s`
3. Increase your system's swap space/virtual memory
4. Use a machine with more RAM
</p>
</details>

<details>
<summary><b>Download Failures</b></summary>
<p>

```
Error: Failed to download model
```

**Solutions:**
1. Check your internet connection
2. Verify you have access to the model on Hugging Face
3. Try using a Hugging Face token with `-t`
4. Check if the model repository exists and is public
</p>
</details>

<details>
<summary><b>Build Failures</b></summary>
<p>

```
Error: Failed to build llama.cpp
```

**Solutions:**
1. Check if you have a C++ compiler installed
2. Ensure you have sufficient disk space
3. Try building with CPU-only mode if GPU builds fail
4. Update your GPU drivers if using acceleration
</p>
</details>

<hr/>

## 🧠 Technical Details

The converter works by following these steps:

1. **Setup**: Clone and build llama.cpp with appropriate hardware acceleration
2. **Download**: Fetch the model from Hugging Face
3. **Convert**: Transform the model to fp16 GGUF format
4. **Quantize**: Apply the requested quantization methods
5. **Split**: Optionally split the model into smaller chunks
6. **Upload**: If credentials are provided, upload to Hugging Face

<details>
<summary><b>Advanced Configuration</b></summary>
<p>

For special cases, you may want to modify llama.cpp's build parameters. The converter automatically detects and enables available hardware acceleration, but you can also build llama.cpp manually with custom options before running the converter.
</p>
</details>

<hr/>

<div align="center">
  <p>
    <a href="https://github.com/OEvortex/Webscout">🔗 Part of the WebScout Project</a> |
    <a href="https://github.com/ggerganov/llama.cpp">🚀 Powered by llama.cpp</a>
  </p>
  
  <p>Made with ❤️ by the Webscout team</p>
</div>