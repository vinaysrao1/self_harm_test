# Self-Harm Content Moderation Test Suite

This repository contains tools for testing self-harm content moderation using different AI models and approaches.

## Overview

The project includes two main testing approaches:

1. **GPT OSS Model Testing** (`gpt_oss.py`) - Tests self-harm content classification using a local GPT OSS model via Ollama
2. **Qwen3Guard Testing** (`qwen3guardtest.py`) - Tests content safety classification using Hugging Face's Qwen3Guard model

## Project Structure

```
├── gpt_oss.py              # GPT OSS model testing script
├── qwen3guardtest.py       # Qwen3Guard model testing script
├── prompts.md              # Test prompts dataset (100 examples)
├── self_harm_policy.md     # Self-harm content moderation policy
├── pyproject.toml          # Project dependencies
└── README.md               # This file
```

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd self_harm_test
```

2. Install dependencies:
```bash
uv add openai transformers torch
```

## Usage

### GPT OSS Model Testing

The GPT OSS testing requires a local Ollama server running with the `gpt-oss:20b` model.

1. Install and start Ollama
2. Pull the GPT OSS model:
```bash
ollama pull gpt-oss:20b
```

3. Run the test:
```bash
uv run python gpt_oss.py
```

### Qwen3Guard Testing

The Qwen3Guard testing downloads the model from Hugging Face automatically.

```bash
uv run python qwen3guardtest.py
```

## Test Data

The `prompts.md` file contains 100 test prompts:
- **Prompts 1-50**: Examples that should be flagged as harmful self-harm content
- **Prompts 51-100**: Examples that should NOT be flagged (recovery content, educational material, etc.)

## Self-Harm Policy

The `self_harm_policy.md` file contains the detailed content moderation policy used for classification, including:

- Clear definitions of self-harm content
- Criteria for what should and shouldn't be flagged
- Examples with rationales
- Guidelines for handling edge cases

## Models

### GPT OSS
- **Model**: `gpt-oss:20b`
- **Approach**: Uses the self-harm policy as a system prompt
- **Output**: Binary classification (0/1)

### Qwen3Guard
- **Model**: `Qwen/Qwen3Guard-Gen-8B`
- **Approach**: Built-in safety classification
- **Output**: Safety label (Safe/Unsafe/Controversial) with categories

## Expected Results

For proper functioning:
- Prompts 1-50 should be classified as harmful/unsafe
- Prompts 51-100 should be classified as safe/educational/recovery-focused

## Contributing

When adding new test cases, ensure they follow the established pattern:
- Clear examples of harmful vs. helpful content
- Proper labeling and categorization
- Alignment with the self-harm policy guidelines