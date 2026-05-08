# FCGformer

本项目主要用于长时间序列预测（Long-Term Forecasting）。包含多种核心预测模型架构。

## 项目结构

- `model/`: 核心预测模型架构，每个模型暴露一个 `Model` 类供 `experiments/exp_long_term_forecasting.py` 调用。
- `layers/`: 共享的注意力机制（attention）、门控机制（gating）以及低秩块（low-rank blocks）等组件。
- `data_provider/`: 数据集读取器。
- `dataset/`: 存放原始 CSV 数据文件。
- `checkpoints/`: 训练过程中保存的模型权重以及训练日志。
- `results/`: 存放输出结果。
- `scripts/`: 包含超参数的实验启动脚本。
- `light_result/`: 存放测试日志、计算图分析 (FLOPs) 等输出结果。

## 环境要求

本项目使用 **`uv`** 进行环境和依赖管理。项目依赖已统一记录在 `pyproject.toml` 中。

### 1. 安装 `uv` 工具
根据您的操作系统安装 `uv`，Windows系统请尽量在WSL下使用：
- **macOS / Linux / Windows(WSL)**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 2. 初始化环境并安装依赖
在项目根目录下执行以下命令，`uv` 会自动读取 `pyproject.toml`，在本地创建虚拟环境（`.venv`）并极速安装包括 `torch`、`numpy`、`pandas` 在内的所有依赖：
```bash
uv sync
```

### 3. 激活虚拟环境
根据您的系统类型，激活刚才生成的 `.venv` 虚拟环境：
- **Mac / Linux / Windows(WSL)**:
  ```bash
  source .venv/bin/activate
  ```

## 数据集准备 (Dataset Preparation)

本项目使用到的自定义空气质量与气象数据集（如 `Center.csv`, `EastNorth.csv`, `Test.csv` 等大文件）由于体积过大，未直接包含在代码仓库中。我已将它们托管在了 Google Drive 上。

请在运行实验前完成以下准备工作：
1. 点击 [此 Google Drive 链接](https://drive.google.com/drive/folders/1xS70xUYvRajNtVtA-xlHMH4BjasWz0j5?usp=drive_link) 下载相关的数据集文件。
2. 将下载好的所有 `.csv` 数据文件放入项目根目录下的 `dataset/` 文件夹中。

## 快速开始

### 1. 模型训练 (Training)
推荐使用 `scripts/` 目录下的 bash 脚本进行模型训练，这些脚本内已经预设了各种任务和数据集的最佳超参数。

以多元预测（Multivariate Forecasting）下的 ETTh1 数据集为例，可以直接执行对应的 shell 脚本：

```bash
# 运行脚本启动训练
bash scripts/Fcgformer.sh
```

如需调整参数，可以直接编辑这些 `.sh` 脚本文件，它们在内部会自动调用 `run.py` 启动相关的实验。

### 2. 模型评估 (Evaluation)
要对保存的模型进行评估（推断），将 `--is_training` 设置为 `0`，并通过 `--checkpoints` 指定保存的权重文件夹路径：

## 提交与开发规范
- **代码风格**: 遵循 PEP 8 规范，使用 4 个空格缩进，类名使用驼峰命名法（如 `Lite_v5_mobile.py`），变量和函数使用蛇形命名法（snake_case）。
- **Git 提交**: 提交信息通常为简明扼要的中文指令格式（如 `fgnn模型修改` 或 `结果排版优化`），保持每次提交修改的聚焦性。
- **配置与日志**: 使用提供的 `ArgParser` 传递参数，不要使用模块级别的全局变量。实验运行时使用普通的 `print` 输出，库代码中尽量使用 `logging` 模块。
