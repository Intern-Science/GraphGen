<p align="center">
  <img src="resources/images/logo.png"/>
</p>

<!-- icon -->

[![stars](https://img.shields.io/github/stars/open-sciencelab/GraphGen.svg)](https://github.com/open-sciencelab/GraphGen)
[![forks](https://img.shields.io/github/forks/open-sciencelab/GraphGen.svg)](https://github.com/open-sciencelab/GraphGen)
[![open issues](https://img.shields.io/github/issues-raw/open-sciencelab/GraphGen)](https://github.com/open-sciencelab/GraphGen/issues)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/open-sciencelab/GraphGen)](https://github.com/open-sciencelab/GraphGen/issues)
[![documentation](https://img.shields.io/badge/docs-latest-blue)](https://graphgen-cookbook.readthedocs.io/en/latest/)
[![wechat](https://img.shields.io/badge/wechat-brightgreen?logo=wechat&logoColor=white)](https://cdn.vansin.top/internlm/dou.jpg)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-white)](https://arxiv.org/abs/2505.20416)
[![Hugging Face](https://img.shields.io/badge/Paper-on%20HF-white?logo=huggingface&logoColor=yellow)](https://huggingface.co/papers/2505.20416)

[![Hugging Face](https://img.shields.io/badge/Demo-on%20HF-blue?logo=huggingface&logoColor=yellow)](https://huggingface.co/spaces/chenzihong/GraphGen)
[![OpenXLab](https://img.shields.io/badge/Demo-on%20OpenXLab-blue?logo=openxlab&logoColor=yellow)](https://g-app-center-120612-6433-jpdvmvp.openxlab.space)

GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation

[English](README.md) | [中文](README_ZH.md)

<details close>
<summary><b>📚 目录</b></summary>

- 📝 [什么是 GraphGen？](#-什么是-graphgen)
- 📌 [最新更新](#最新更新)
- 🚀 [快速开始](#快速开始)
- 🏗️ [系统架构](#系统架构)
- 🍀 [致谢](#致谢)
- 📚 [引用](#引用)
- 📜 [许可证](#许可证)
- 📅 [星标历史](#星标历史)


[//]: # (- 🌟 [主要特性](#主要特性))
[//]: # (- 💰 [成本分析](#成本分析))
[//]: # (- ⚙️ [配置说明](#配置说明))

</details>


## 📝 什么是 GraphGen？

GraphGen 是一个基于知识图谱的数据合成框架。请查看[**论文**](https://arxiv.org/abs/2505.20416)和[最佳实践](https://github.com/open-sciencelab/GraphGen/issues/17)。

以下是在超过 50 % 的 SFT 数据来自 GraphGen 及我们的数据清洗流程时的训练后结果：

| 领域 | 数据集 | 我们的方案 | Qwen2.5-7B-Instruct（基线） |
| :-: | :-: | :-: | :-: |
| 植物 | [SeedBench](https://github.com/open-sciencelab/SeedBench) | **65.9** | 51.5 |
| 常识 | CMMLU | 73.6 | **75.8** |
| 知识 | GPQA-Diamond | **40.0** | 33.3 |
| 数学 | AIME24 | **20.6** | 16.7 |
| | AIME25 | **22.7** | 7.2 |

GraphGen 首先根据源文本构建细粒度的知识图谱，然后利用期望校准误差指标识别大语言模型中的知识缺口，优先生成针对高价值长尾知识的问答对。  
此外，GraphGen 采用多跳邻域采样捕获复杂关系信息，并使用风格控制生成来丰富问答数据的多样性。

在数据生成后，您可以使用[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 和 [xtuner](https://github.com/InternLM/xtuner)对大语言模型进行微调。

## 📌 最新更新

- **2025.08.14**：支持利用 Leiden 社区发现算法对知识图谱进行社区划分，合成 CoT 数据。
- **2025.07.31**：新增 Google、Bing、Wikipedia 和 UniProt 作为搜索后端，帮助填补数据缺口。  
- **2025.04.21**：发布 GraphGen 初始版本。

## 🚀 快速开始

通过 [Web](https://g-app-center-120612-6433-jpdvmvp.openxlab.space) 或 [备用 Web 入口](https://openxlab.org.cn/apps/detail/chenzihonga/GraphGen) 体验 GraphGen。

如有任何问题，请查看 [FAQ](https://github.com/open-sciencelab/GraphGen/issues/10)、提交新的 [issue](https://github.com/open-sciencelab/GraphGen/issues) 或加入我们的[微信群](https://cdn.vansin.top/internlm/dou.jpg)咨询。

### 准备工作

1. 安装 [uv](https://docs.astral.sh/uv/reference/installer/)

    ```bash
    # 若遇到网络问题，可尝试使用 pipx 或 pip 安装 uv，详见 uv 文档
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2. 克隆仓库

    ```bash
    git clone --depth=1 https://github.com/open-sciencelab/GraphGen
    cd GraphGen
    ```
3. 创建新的 uv 环境

    ```bash
    uv venv --python 3.10
    ```
4. 安装依赖

    ```bash
    uv pip install -r requirements.txt
    ```

### 运行 Gradio 演示

   ```bash
   python -m webui.app
   ```

![ui](https://github.com/user-attachments/assets/3024e9bc-5d45-45f8-a4e6-b57bd2350d84)

### 从 PyPI 运行

1. 安装 GraphGen
   ```bash
   uv pip install graphg
   ```

2. CLI 运行
    ```bash
    SYNTHESIZER_MODEL=your_synthesizer_model_name \
    SYNTHESIZER_BASE_URL=your_base_url_for_synthesizer_model \
    SYNTHESIZER_API_KEY=your_api_key_for_synthesizer_model \
    TRAINEE_MODEL=your_trainee_model_name \
    TRAINEE_BASE_URL=your_base_url_for_trainee_model \
    TRAINEE_API_KEY=your_api_key_for_trainee_model \
    graphg --output_dir cache
    ```

### 源码运行

1. 配置环境
   - 在项目根目录创建 `.env` 文件
     ```bash
     cp .env.example .env
     ```
   - 设置以下环境变量：
     ```bash
     # Synthesizer 用于构建知识图谱并生成数据
     SYNTHESIZER_MODEL=your_synthesizer_model_name
     SYNTHESIZER_BASE_URL=your_base_url_for_synthesizer_model
     SYNTHESIZER_API_KEY=your_api_key_for_synthesizer_model
     # Trainee 用于使用生成数据进行训练
     TRAINEE_MODEL=your_trainee_model_name
     TRAINEE_BASE_URL=your_base_url_for_trainee_model
     TRAINEE_API_KEY=your_api_key_for_trainee_model
     ```
2. （可选）如需修改默认生成配置，可编辑 `graphgen/configs/` 文件夹中的 YAML 文件.

   例如：

    ```yaml
      # configs/cot_config.yaml
      input_file: resources/input_examples/jsonl_demo.jsonl
      output_data_type: cot
      tokenizer: cl100k_base
      # 其他设置...
    ```

3. 生成数据

   选择所需格式并运行对应脚本：
   
   | 格式           | 运行脚本                                           | 说明           |
   |--------------|------------------------------------------------|--------------|
   | `cot`        | `bash scripts/generate/generate_cot.sh`        | 思维链问答对       |
   | `atomic`     | `bash scripts/generate/generate_atomic.sh`     | 覆盖基础知识的原子问答对 |
   | `aggregated` | `bash scripts/generate/generate_aggregated.sh` | 整合复杂知识的聚合问答对 |
   | `multi-hop`  | `bash scripts/generate/generate_multihop.sh`   | 多跳推理问答对      |


4. 查看生成结果
   ```bash
   ls cache/data/graphgen
   ```

### 使用 Docker 运行
1. 构建镜像
   ```bash
   docker build -t graphgen .
   ```
2. 启动容器
   ```bash
    docker run -p 7860:7860 graphgen
    ```


## 🏗️ 系统架构
参阅 deepwiki 的[分析](https://deepwiki.com/open-sciencelab/GraphGen)了解 GraphGen 系统、架构与核心功能的技术概览。


### 工作流程
![workflow](resources/images/flow.png)


## 🍀 致谢
- [SiliconFlow](https://siliconflow.cn) 提供丰富的 LLM API，部分模型免费
- [LightRAG](https://github.com/HKUDS/LightRAG) 简单高效的图检索方案
- [ROGRAG](https://github.com/tpoisonooo/ROGRAG) 鲁棒优化版 GraphRAG 框架
- [DB-GPT](https://github.com/eosphoros-ai/DB-GPT) AI 原生数据应用开发框架


## 📚 引用
如果本项目对你有帮助，请考虑引用我们的工作：
```bibtex
@misc{chen2025graphgenenhancingsupervisedfinetuning,
      title={GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation}, 
      author={Zihong Chen and Wanli Jiang and Jinzhe Li and Zhonghang Yuan and Huanjun Kong and Wanli Ouyang and Nanqing Dong},
      year={2025},
      eprint={2505.20416},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.20416}, 
}
```

## 📜 许可证
本项目采用 [Apache License 2.0](LICENSE) 许可证。

## 📅 星标历史

[![Star History Chart](https://api.star-history.com/svg?repos=open-sciencelab/GraphGen&type=Date)](https://www.star-history.com/#open-sciencelab/GraphGen&Date)

