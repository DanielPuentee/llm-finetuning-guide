# Fine-Tuning Guide

[![Visual Studio Code](https://custom-icon-badges.demolab.com/badge/Visual%20Studio%20Code-0078d7.svg?logo=vsc&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](#)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.57-blue?logo=huggingface&logoColor=white)
![TRL](https://img.shields.io/badge/TRL-0.27-ff9900)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-6f42c1)
![GitHub last commit](https://img.shields.io/github/last-commit/DanielPuentee/llm-finetuning-guide)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> Author: [Daniel Puente Viejo](https://www.linkedin.com/in/danielpuenteviejo/)

A hands-on guide for fine-tuning the models TinyLlama1.1B & Llama3.1-8B-Instruct with low-rank adapters (LoRA). The repository bundles Colab and macOS workflows, curated datasets, and fully reproducible experiments so you can adapt compact open models to your own domain with minimal compute.

## 🎯 What You Get

- Dual workflows covering [google_colab/fine-tuning.ipynb](google_colab/fine-tuning.ipynb) and [mac/fine-tuning.ipynb](mac/fine-tuning.ipynb)
- Ready-to-use basketball dataset in [data](data) plus raw corpus in [data/data.txt](data/data.txt)
- 2 ways of training. One with Unsloth (google colab) and another with Hugging Face's SFTTrainer (macOS)
- Logging-ready setup (Accelerate, bitsandbytes, safetensors) for efficient experimentation on consumer GPUs or Colab T4s

## 🧱 Project Structure

~~~text
├── data/                      # Supervised fine-tuning corpora (JSON + raw text)
├── full_training.ipynb        # Scratchpad for extended experiments
├── google_colab/
│   ├── fine-tuning.ipynb      # Colab workflow (setup + training + evaluation)
│   └── imgs/                  # Notebook figures and configuration screenshots
├── mac/
│   ├── fine-tuning.ipynb      # Local M-series workflow with Accelerate + LoRA
│   └── requirements.txt       # Exact Python dependencies for local runs
└── README.md                  # You are here
~~~

## 🚀 Quick Start

### Option A — Google Colab (GPU in the cloud)

- Open [google_colab/fine-tuning.ipynb](google_colab/fine-tuning.ipynb) and connect to a GPU runtime (T4 or better)
- Run the setup cells to install Transformers, TRL, PEFT, bitsandbytes and Unsloth.
- Load datasets directly from [data](data)

### Option B — Local macOS (Apple Silicon)

1. Create a Python 3.10+ environment (uv, conda, or venv) and activate it
2. Install dependencies from [mac/requirements.txt](mac/requirements.txt)

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r mac/requirements.txt
~~~

## 📓 Notebooks at a Glance

- [google_colab/fine-tuning.ipynb](google_colab/fine-tuning.ipynb): Designed for quick iteration on Colab with Unsloth, showcasing the full fine-tuning loop and evaluation on the basketball dataset
- [mac/fine-tuning.ipynb](mac/fine-tuning.ipynb): Optimized for Apple Silicon with 4-bit loading, Accelerate configuration, and local inference tests
- [full_training.ipynb](full_training.ipynb): Sandbox for extended experiments (curriculum changes, alternative prompts, or ablation studies)

## 🧾 Data Expectations

- Supervised pairs live in [data/atomic_train.json](data/atomic_train.json). Each entry follows the schema below:

~~~json
{
	"question": "What type of sport is basketball?",
	"answer": "Basketball is a team sport played on a rectangular court."
}
~~~

- Raw context text for language-model warm-up resides in [data/data.txt](data/data.txt). Provide one or more paragraphs separated by blank lines.

To swap in your own dataset, keep the JSON keys (`question`, `answer`) or adjust the preprocessing cell in the notebooks to match your schema.

## 📦 Outputs & Artifacts

- Training checkpoints: [mac/results/checkpoint-*](mac/results)
- Trainer state, optimizer, and scheduler snapshots: [mac/results](mac/results)
- Final LoRA adapter pack: [mac/tiny-llama-finetuned](mac/tiny-llama-finetuned)
- Sample generations: captured inside the notebooks and exportable as markdown transcripts

## 🧠 Recommended Learning Path

1. Review the Colab notebook to understand the end-to-end flow
2. Inspect the dataset examples in [data](data).
3. Enjoy the notebooks!

## 📃 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.