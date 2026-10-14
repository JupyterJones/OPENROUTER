# 🛠️ Installation & Environment Setup Guide

This guide will walk you through setting up a complete, isolated Python virtual environment (`venv`) with **Kokoro TTS**, **OpenAI Whisper**, and all rendering tools required to generate autonomous HTML scrolling videos.

---

## 1. System Prerequisites

Before configuring the Python environment, ensure the required system packages are installed on your machine.

### Linux (Ubuntu / Debian / Mint)
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg google-chrome-stable espeak-ng git
```

> **Note on Chrome**: If `google-chrome-stable` is not in your package manager, install Chromium or download Chrome:
> ```bash
> # Or install Chromium:
> sudo apt install -y chromium-browser
> ```

### macOS (Homebrew)
```bash
brew update
brew install python ffmpeg espeak-ng
brew install --cask google-chrome
```

### Windows (WSL2 Recommended)
We recommend running this project inside **Ubuntu on WSL2** (Windows Subsystem for Linux) and following the Ubuntu instructions above.

---

## 2. Setting Up the Python Virtual Environment

Using a virtual environment (`venv`) keeps your dependencies isolated from your operating system's global Python packages.




### Step 1: Open your terminal and navigate to the project directory
```bash
git clone git@github.com:JupyterJones/OPENROUTER.git
cd OPENROUTER  # replace with your actual cloned repo path
```

### Step 2: Create a virtual environment
```bash
python3 -m venv venv
```

### Step 3: Activate the virtual environment
Whenever you work on this project or run scripts, activate the venv first:

```bash
source venv/bin/activate
```

*(Your terminal prompt should now show `(venv)` at the beginning).*

To deactivate when finished:
```bash
deactivate
```

---

## 3. Installing Python Dependencies

With your virtual environment activated, upgrade package installers and install all project requirements:

```bash
# Upgrade pip, wheel, and setuptools
pip install --upgrade pip setuptools wheel

# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

### What this installs:
- **`kokoro-tts`**: High-performance, low-latency neural text-to-speech engine.
- **`openai-whisper` & `torch`**: Automatic speech recognition for voice-to-DOM timeline alignment.
- **`beautifulsoup4`**: HTML parsing, text extraction, and audio-repeat elimination.
- **`pillow` & `numpy`**: Visual boundary calculations, image processing, and color diffing.
- **`soundfile`**: High-fidelity WAV reading/writing.

---

## 4. Downloading the Kokoro TTS Model & Voices

Kokoro TTS uses an ONNX weight file (`kokoro-v1.0.onnx`, ~335MB) and a voice embedding file (`voices-v1.0.bin`, ~26MB).

Download both files directly into your project directory:

```bash
# Download the Kokoro ONNX model (~335 MB)
wget https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx

# Download the voice embeddings file (~26 MB)
wget https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin
```

*(Alternatively, if `wget` is not installed, use `curl -L -O`)*:
```bash
curl -L -O https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
curl -L -O https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin
```

---

## 5. Verifying the Installation

Run these quick checks inside your activated `(venv)` to confirm all components are ready:

### 1. Test Kokoro TTS CLI
```bash
kokoro-tts --help
```
You should see the command-line help for `kokoro-tts`.

### 2. Test Whisper & PyTorch
```bash
python3 -c "import whisper, torch; print(f'Whisper ready! PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

### 3. Test Headless Chrome
```bash
google-chrome --version || chromium --version
```

### 4. Test FFmpeg
```bash
ffmpeg -version && ffprobe -version
```

---

## 6. Quick Smoke Test

Create a sample audio file with Kokoro to verify everything is working:

```bash
echo "Testing Kokoro neural voice synthesis with Whisper alignment." > test_input.txt
kokoro-tts test_input.txt test_output.wav --voice am_michael --model ./kokoro-v1.0.onnx --voices ./voices-v1.0.bin
```

If `test_output.wav` is generated, your environment is 100% operational!

---

## 7. Troubleshooting & FAQ

### `command not found: kokoro-tts`
- Ensure you activated your virtual environment first (`source venv/bin/activate`).
- Verify installation with `pip show kokoro-tts`.

### `espeak-ng library not found`
- Kokoro uses phonemization. Install `espeak-ng` via your OS package manager:
  ```bash
  sudo apt install espeak-ng  # Linux
  brew install espeak-ng      # macOS
  ```

### GPU vs CPU
- By default, PyTorch and Whisper will use an NVIDIA GPU (CUDA) if detected.
- If no GPU is found, they automatically fall back to CPU with multithreading.
