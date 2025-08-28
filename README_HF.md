---
title: GraphGen Demo
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.10"
app_file: webui/app.py
suggested_hardware: cpu-basic
pinned: false
short_description: "Interactive knowledge-driven synthetic data generation demo powered by GraphGen & Gradio"
tags:
  - synthetic-data
  - knowledge-graph
  - gradio-demo
---

# GraphGen Space 🤖📊

This is the **official Hugging Face Space** for [GraphGen](https://github.com/open-sciencelab/GraphGen) – a framework that leverages knowledge graphs to generate high-quality synthetic question–answer pairs for supervised fine-tuning of LLMs.

🔗 Paper: [arXiv 2505.20416](https://arxiv.org/abs/2505.20416)  
🐙 GitHub: [open-sciencelab/GraphGen](https://github.com/open-sciencelab/GraphGen)

---

## How to use (🖱️ 3 clicks)

1. Open the **Gradio app** above.  
2. Upload or paste your source text → click **Generate KG**.  
3. Download the generated QA pairs directly.

---

## Local quick start (optional)

```bash
git clone https://github.com/open-sciencelab/GraphGen
cd GraphGen
uv venv --python 3.10 && uv pip install -r requirements.txt
uv run webui/app.py   # http://localhost:7860
```
