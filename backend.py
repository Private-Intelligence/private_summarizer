"""
Private Summarizer Backend
A private AI text summarization server.

Models used:
- Default: LiquidAI/LFM2.5-1.2B-Instruct (multilingual)
- Japanese: LiquidAI/LFM2.5-1.2B-JP (optimized for Japanese)
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Model configuration
# Default model for multilingual support (English, Chinese, etc.)
DEFAULT_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"
# Specialized model for Japanese (recommended for Japanese text)
JAPANESE_MODEL = "LiquidAI/LFM2.5-1.2B-JP"

# Allow model override via environment variable
SELECTED_MODEL = os.environ.get("LFM_MODEL", DEFAULT_MODEL)

# Model storage
models = {}
tokenizers = {}

def load_model(model_id):
    """Load a model and tokenizer, with caching."""
    if model_id in models:
        return models[model_id], tokenizers[model_id]

    print(f"Loading model: {model_id}...")

    try:
        # Detect device and dtype
        if torch.cuda.is_available():
            device_map = "auto"
            dtype = torch.bfloat16
            print(f"  Using CUDA GPU")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_map = "mps"
            dtype = torch.float16  # MPS works better with float16
            print(f"  Using Apple Silicon (MPS)")
        else:
            device_map = "cpu"
            dtype = torch.float32
            print(f"  Using CPU")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=dtype,
            # Uncomment for compatible GPUs with Flash Attention 2:
            # attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        models[model_id] = model
        tokenizers[model_id] = tokenizer
        print(f"  Model loaded successfully!")

        return model, tokenizer

    except Exception as e:
        print(f"  Error loading model: {e}")
        return None, None


# Prompts for different languages
PROMPTS = {
    "en": """Summarize the following text into clear, concise bullet points.

Rules:
- Be brief and direct - no filler words
- Use short bullet points (1 line each)
- Group related points under bold headers if needed
- Extract only key information
- No introductions or conclusions

Text:
{text}

Summary:""",

    "zh": """将以下文本总结为简洁的要点。

要求：
- 简明扼要，不要废话
- 每个要点一行
- 相关内容可用粗体标题分组
- 只提取关键信息
- 不需要开场白或总结语

文本：
{text}

摘要：""",

    "ja": """以下のテキストを簡潔な箇条書きで要約してください。

ルール：
- 簡潔に、無駄な言葉は省く
- 各ポイントは1行で
- 関連する内容は太字の見出しでグループ化
- 重要な情報のみ抽出
- 前置きや結論は不要

テキスト:
{text}

要約:"""
}


def get_model_for_language(language):
    """Get the appropriate model for the given language."""
    if language == "ja":
        return JAPANESE_MODEL
    return DEFAULT_MODEL


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "default_model": DEFAULT_MODEL,
        "japanese_model": JAPANESE_MODEL,
        "loaded_models": list(models.keys()),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    })


@app.route('/summarize', methods=['POST'])
def summarize():
    """Summarize text in the specified language."""
    try:
        data = request.json
        input_text = data.get('text', '').strip()
        language = data.get('language', 'en')

        # Custom settings from frontend
        custom_model = data.get('model', '')
        custom_prompt = data.get('custom_prompt', '')
        temperature = data.get('temperature', 0.3)
        max_tokens = data.get('max_tokens', 512)

        if not input_text:
            return jsonify({"error": "No text provided"}), 400

        # Validate language
        if language not in PROMPTS:
            language = 'en'

        # Use custom model if provided, otherwise get appropriate model for language
        if custom_model:
            model_id = custom_model
        else:
            model_id = get_model_for_language(language)

        model, tokenizer = load_model(model_id)

        if model is None or tokenizer is None:
            return jsonify({"error": f"Failed to load model: {model_id}"}), 500

        # Use custom prompt if provided, otherwise use default
        if custom_prompt and '{text}' in custom_prompt:
            prompt = custom_prompt.replace('{text}', input_text)
        else:
            prompt = PROMPTS[language].format(text=input_text)

        # Prepare input
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(model.device)

        # Generate summary with custom settings
        with torch.no_grad():
            output = model.generate(
                input_ids,
                do_sample=True,
                temperature=float(temperature),
                min_p=0.15,
                repetition_penalty=1.05,
                max_new_tokens=int(max_tokens),
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode output
        summary = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract only the generated summary (remove the prompt)
        # Look for language-specific markers
        markers = {
            "en": "Summary:",
            "zh": "摘要：",
            "ja": "要約:"
        }
        marker = markers.get(language, "Summary:")

        if marker in summary:
            summary = summary.split(marker)[-1].strip()

        # Remove any "assistant" prefix or similar role markers
        summary = summary.replace("assistant", "").strip()

        # Normalize line breaks
        summary = "\n".join(line.strip() for line in summary.split("\n") if line.strip())

        return jsonify({
            "summary": summary,
            "model": model_id,
            "language": language,
            "input_length": len(input_text)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/models', methods=['GET'])
def list_models():
    """List available models."""
    return jsonify({
        "available_models": {
            "default": DEFAULT_MODEL,
            "japanese": JAPANESE_MODEL
        },
        "loaded_models": list(models.keys())
    })


def print_banner():
    """Print startup banner."""
    print("")
    print("=" * 56)
    print("  Private Summarizer - Private AI Text Summarization")
    print("=" * 56)
    print("")
    print("  Models:")
    print(f"    Default:  {DEFAULT_MODEL}")
    print(f"    Japanese: {JAPANESE_MODEL}")
    print("")
    print("  Device:")
    if torch.cuda.is_available():
        print(f"    CUDA GPU ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("    Apple Silicon (MPS)")
    else:
        print("    CPU")
    print("")
    print("=" * 56)
    print("  Server: http://localhost:8000")
    print("=" * 56)
    print("")


def main():
    """Main entry point."""
    print_banner()

    # Pre-load default model on startup
    print("Pre-loading default model...")
    load_model(DEFAULT_MODEL)
    print("")

    # Run the server
    app.run(host='0.0.0.0', port=8000, debug=False)


if __name__ == '__main__':
    main()
