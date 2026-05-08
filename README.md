# FCGformer

本项目主要用于长时间序列预测（Long-Term Forecasting）。包含多种核心预测模型架构（如 Lite, iTransformer, FourierGNN, TimeMixer 等）。

## 项目结构

- `model/`: 核心预测模型架构（例如 Lite, iTransformer, FourierGNN, TimeMixer 等），每个模型暴露一个 `Model` 类供 `experiments/exp_long_term_forecasting.py` 调用。
- `layers/`: 共享的注意力机制（attention）、门控机制（gating）以及低秩块（low-rank blocks）等组件。
- `data_provider/` & `data_preprocess/`: 数据集读取器以及数据预处理步骤。
- `dataset/`: 存放原始 CSV 数据文件的目录。
- `checkpoints/`: 训练过程中保存的模型权重和张量。
- `results/`, `frequency_outputs/`, `visual/`: 存放输出指标、日志和可视化的图表。
- `scripts/`: 包含精心调优超参数的实验启动脚本。
- `test_results/`, `light_result/`: 存放测试日志、计算图分析 (FLOPs) 等输出结果。

## 环境要求

建议使用 Python 3.9+。可以通过 `conda` 或 `pip` 配置环境。

### 方法 1：使用 Conda (推荐)
如果环境配置文件存在，可以使用以下命令创建并激活环境：
```bash
conda env create -f environment.yml
conda activate fcgformer
```

### 方法 2：使用 Pip
也可以直接通过 `requirements.txt` 安装相关依赖：
```bash
python -m pip install -r requirements.txt
```

**主要依赖包：**
- PyTorch >= 1.10.0
- NumPy == 1.23.5
- Pandas == 1.5.3
- SciPy >= 1.7.0
- Scikit-learn == 1.2.2
- Matplotlib == 3.7.0
- reformer-pytorch == 1.4.4

## 快速开始

### 1. 模型训练 (Training)
推荐使用 `scripts/` 目录下的 bash 脚本进行模型训练，这些脚本内已经预设了各种任务和数据集的最佳超参数。

以多元预测（Multivariate Forecasting）下的 ETTh1 数据集为例，可以直接执行对应的 shell 脚本：

```bash
# 赋予脚本执行权限（如果需要）
chmod +x ./scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh

# 运行脚本启动训练
./scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh
```

如需调整参数，可以直接编辑这些 `.sh` 脚本文件，它们在内部会自动调用 `run.py` 启动相关的实验。

### 2. 模型评估 (Evaluation)
要对保存的模型进行评估（推断），将 `--is_training` 设置为 `0`，并通过 `--checkpoints` 指定保存的权重文件夹路径：

```bash
python run.py \
  --is_training 0 \
  --checkpoints ./checkpoints/[你的模型保存路径]/ \
  --model Lite \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --seq_len 96 \
  --pred_len 24
```

### 3. 测试与回归 (Testing)
为了保证代码修改没有引入 Regression，如果修改了 Lite 等相关代码，请运行以下测试脚本并确保全部通过。控制台输出会保存在 `test_results/` 或 `light_result/` 目录下。

```bash
python test_lite_compatibility.py
python test_optimizations.py
```

如果你需要增加新的测试功能，请在根目录下添加命名类似于 `test_<capability>.py` 的脚本。

## 提交与开发规范
- **代码风格**: 遵循 PEP 8 规范，使用 4 个空格缩进，类名使用驼峰命名法（如 `Lite_v5_mobile.py`），变量和函数使用蛇形命名法（snake_case）。
- **Git 提交**: 提交信息通常为简明扼要的中文指令格式（如 `fgnn模型修改` 或 `结果排版优化`），保持每次提交修改的聚焦性。
- **配置与日志**: 使用提供的 `ArgParser` 传递参数，不要使用模块级别的全局变量。实验运行时使用普通的 `print` 输出，库代码中尽量使用 `logging` 模块。
