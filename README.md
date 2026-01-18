# Private Summarizer

A private AI text summarization tool that runs entirely on your machine. No data leaves your computer.

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)

---

## Why Private AI?

In an era where data privacy is increasingly important, **Private Summarizer** demonstrates that powerful AI doesn't have to mean sending your sensitive information to external servers.

**Your data stays with you.** Whether you're summarizing confidential business communications, personal notes, or sensitive documents, everything runs locally on your machine. No API keys needed. No cloud services required. No data collection.

This project is built on the belief that **private AI** should be accessible to everyone.

---

## Features

- **100% Private**: All processing happens locally on your machine
- **Multi-language Support**: English, Chinese (中文), and Japanese (日本語)
- **No Internet Required**: Works completely offline after initial model download
- **Cross-platform**: Works on Windows, macOS, and Linux
- **GPU Accelerated**: Supports CUDA GPUs and Apple Silicon (MPS)
- **Beautiful Interface**: Clean, responsive web UI with smooth animations
- **Easy Setup**: One command to get started

---

## Quick Start

### Prerequisites

- **Python**: 3.9 or higher
- **Disk Space**: ~3GB for models (downloaded on first run)
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: Optional but recommended for faster inference
- **Internet**: Required only for first run (model download)

### Setup

```bash
# Clone the repository
git clone https://github.com/Private-Intelligence/private_summarizer.git
cd private_summarizer

# Run setup (installs uv and dependencies)
./setup.sh

# Start the backend
uv run python backend.py
```

Then open `summarizer.html` in your browser.

### Manual Setup

If you prefer manual installation:

```bash
# Install uv (https://github.com/astral-sh/uv)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --no-install-project

# Run the backend
uv run python backend.py
```

### Windows

```powershell
# Install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install dependencies
uv sync --no-install-project

# Run the backend
uv run python backend.py
```

---

## Usage

1. **Start the backend**: Run `uv run python backend.py`
2. **Open the interface**: Open `summarizer.html` in any modern browser
3. **Select language**: Click EN, 中文, or 日本語 in the top-right corner
4. **Enter text**: Replace the sample call transcript with your own text
5. **Generate**: Click "Generate Summary" or press `Ctrl/Cmd + Enter`
6. **View results**: The structured summary appears in the right panel

> **Note**: On first run, the AI model (~2.5GB) will be downloaded automatically. This may take a few minutes depending on your connection. Subsequent runs will be much faster.

### Settings

Click the ⚙️ button to customize:
- **Model**: Use any compatible Hugging Face model
- **Custom Prompt**: Define your own summarization prompt template (use `{text}` as placeholder)
- **Temperature**: Adjust creativity (lower = more focused, higher = more creative)
- **Max Output Length**: Control the summary length

### Keyboard Shortcuts

- `Ctrl/Cmd + Enter`: Generate summary
- `Escape`: Close settings modal

---

## Models

This project uses the LFM (Liquid Foundation Model) series, which are efficient models well-suited for local deployment:

| Model | Language | Use Case |
|-------|----------|----------|
| [LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) | Multilingual | Default for English, Chinese, etc. |
| [LFM2.5-1.2B-JP](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP) | Japanese | Optimized for Japanese text |

The appropriate model is automatically selected based on your language choice. These ~1.2B parameter models offer a good balance between quality and resource usage for consumer hardware.

---

## Configuration

### Backend Settings

Edit `backend.py` to customize:

```python
# Generation parameters (in summarize function)
temperature=0.3,      # Lower = more deterministic
max_new_tokens=512,   # Maximum summary length
repetition_penalty=1.05,  # Prevents repetitive output
```

### Environment Variables

```bash
# Override default model
export LFM_MODEL="LiquidAI/LFM2.5-1.2B-Instruct"
```

### GPU Acceleration

**CUDA (NVIDIA)**:
The backend automatically detects and uses CUDA GPUs.

**Apple Silicon (M1/M2/M3)**:
MPS acceleration is automatically enabled on Apple Silicon Macs.

**Flash Attention 2**:
For compatible NVIDIA GPUs, you can enable Flash Attention 2 for faster inference by uncommenting the relevant line in `backend.py`:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"  # Uncomment this line
)
```

---

## API Reference

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "default_model": "LiquidAI/LFM2.5-1.2B-Instruct",
  "japanese_model": "LiquidAI/LFM2.5-1.2B-JP",
  "loaded_models": ["LiquidAI/LFM2.5-1.2B-Instruct"],
  "cuda_available": false,
  "mps_available": true
}
```

### Summarize Text

```http
POST /summarize
Content-Type: application/json

{
  "text": "Your text to summarize...",
  "language": "en"  // "en", "zh", or "ja"
}
```

**Response:**
```json
{
  "summary": "Structured summary...",
  "model": "LiquidAI/LFM2.5-1.2B-Instruct",
  "language": "en",
  "input_length": 1234
}
```

### List Models

```http
GET /models
```

**Response:**
```json
{
  "available_models": {
    "default": "LiquidAI/LFM2.5-1.2B-Instruct",
    "japanese": "LiquidAI/LFM2.5-1.2B-JP"
  },
  "loaded_models": ["LiquidAI/LFM2.5-1.2B-Instruct"]
}
```

---

## Project Structure

```
private_summarizer/
├── summarizer.html    # Frontend interface (multi-language)
├── backend.py         # Flask backend server
├── pyproject.toml     # Python dependencies & project config
├── setup.sh           # Setup script (uses uv)
├── LICENSE            # Apache 2.0 License
├── CONTRIBUTING.md    # Contribution guidelines
└── README.md          # This file
```

---

## Troubleshooting

### "Cannot connect to backend server"

- Ensure the backend is running (`uv run python backend.py`)
- Check that port 8000 is not in use by another application
- Try accessing http://localhost:8000/health in your browser

### Slow generation

- Use a GPU if available (CUDA or Apple Silicon)
- Reduce `max_new_tokens` in `backend.py`
- Close other memory-intensive applications

### Out of memory

- The models require approximately 3GB of RAM
- Close other applications to free up memory
- On systems with limited RAM, generation may be slower but should still work

### Model download issues

- Models are downloaded from Hugging Face Hub on first run
- Ensure you have a stable internet connection for the initial download
- After download, models are cached locally (~2.5GB per model)

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Acknowledgments

- [Liquid AI](https://www.liquid.ai/) for the LFM models
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [Flask](https://flask.palletsprojects.com/)
- [uv](https://github.com/astral-sh/uv)

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

The LFM models have their own license terms - see the model pages on Hugging Face for details.

---

## About & Connect

Hi, I'm Changyu. I focus on building private AI solutions that are practical, secure, and easy to use. My passion is applying AI to real-world problems—designing tools that help people create value while keeping full control of their data.

If you're interested in private AI applications or have ideas to share, feel free to connect:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/changyu-hu-607a55149/)

---

**Built with the belief that AI should empower, not compromise, your privacy.**
