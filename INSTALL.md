# FCGFormer 安装指南

本文档提供了FCGFormer项目的详细安装指南，支持多种安装方式。

## 📋 系统要求

- **Python**: 3.8 或更高版本
- **操作系统**: Windows, Linux, macOS
- **GPU**: NVIDIA GPU (推荐，用于加速训练)
- **CUDA**: 11.8 或兼容版本 (如果使用GPU)

## 🚀 安装方式

### 方式一：使用pip安装（推荐）

1. **克隆项目**
```bash
git clone <your-repo-url>
cd fcgformer
```

2. **创建虚拟环境**
```bash
# 使用venv
python -m venv fcgformer_env
source fcgformer_env/bin/activate  # Linux/macOS
# 或
fcgformer_env\Scripts\activate     # Windows

# 使用virtualenv
virtualenv fcgformer_env
source fcgformer_env/bin/activate  # Linux/macOS
# 或
fcgformer_env\Scripts\activate     # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

### 方式二：使用conda安装

1. **克隆项目**
```bash
git clone <your-repo-url>
cd fcgformer
```

2. **创建conda环境**
```bash
conda env create -f environment.yml
```

3. **激活环境**
```bash
conda activate fcgformer
```

## 🔧 GPU支持配置

### CUDA版本检查
首先检查你的CUDA版本：
```bash
nvidia-smi
nvcc --version
```

### PyTorch GPU版本安装
根据你的CUDA版本，可能需要安装特定版本的PyTorch：

**CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CPU版本:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## ✅ 验证安装

### 1. 测试基本功能
```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

### 2. 测试资源监控功能
```bash
python test_resource_monitor.py
```

### 3. 运行完整训练测试
```bash
python run.py --is_training 0 --model_id test --model Lite --data custom --root_path ./dataset/AIR-EW/ --data_path AW-NEW.csv
```

## 📦 可选依赖

如果你需要额外功能，可以安装以下可选依赖：

### Jupyter支持
```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name fcgformer --display-name "FCGFormer"
```

### 可视化增强
```bash
pip install seaborn plotly
```

### 开发工具
```bash
pip install pytest black flake8
```

## 🛠️ 常见问题

### 1. CUDA相关错误
如果遇到CUDA版本不兼容：
```bash
# 卸载现有PyTorch
pip uninstall torch torchvision
# 重新安装匹配的版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. 内存不足
如果训练时显存不足，调整batch_size：
```bash
python run.py --batch_size 16  # 减小batch_size
```

### 3. 依赖冲突
如果出现依赖冲突，建议使用全新的虚拟环境：
```bash
# 删除旧环境
conda env remove -n fcgformer
# 重新创建
conda env create -f environment.yml
```

### 4. Windows路径问题
Windows用户如果遇到路径问题，确保使用正斜杠：
```bash
python run.py --root_path "./dataset/AIR-EW/" --data_path "AW-NEW.csv"
```

## 📊 性能优化建议

1. **GPU加速**: 确保正确安装CUDA版本的PyTorch
2. **内存管理**: 适当调整batch_size和num_workers参数
3. **数据预处理**: 预先处理数据集以加速训练
4. **混合精度**: 使用`--use_amp`参数启用自动混合精度训练

## 🔍 故障排除

如果安装过程中遇到问题：

1. **检查Python版本**: `python --version`
2. **检查pip版本**: `pip --version`
3. **更新pip**: `pip install --upgrade pip`
4. **清理缓存**: `pip cache purge`
5. **使用清华镜像**: `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/`

## 📞 获取帮助

如果仍然遇到问题：
1. 检查项目的Issue页面
2. 确保按照正确的步骤安装
3. 提供详细的错误信息和系统环境信息

---
**注意**: 首次运行可能需要下载预训练模型和数据集，请确保网络连接稳定。