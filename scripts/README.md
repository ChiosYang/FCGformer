# FCGformer 模型实验脚本说明

本目录包含了用于运行 FCGformer 模型的训练与测试脚本。

## 运行方式

要启动 FCGformer 模型在指定数据集上的训练，请在项目根目录下直接运行以下命令：

```bash
bash ./scripts/Fcgformer.sh
```

## `Fcgformer.sh` 参数逐行详解

`Fcgformer.sh` 脚本通过调用核心的 `run.py` 传入了详尽的超参数配置。以下是脚本中每一行参数的具体作用与说明：

* `export CUDA_VISIBLE_DEVICES=0`：设置环境变量，指定程序只使用第 0 号 GPU 显卡。
* `model_name=FCGformer`：定义一个 Bash 变量，存储当前使用的模型名称(需要首先在./experiments/exp_basic.py下定义)。
* `python -u run.py \`：使用无缓冲模式 (`-u`) 运行主程序 `run.py`，确保日志能实时输出到控制台。
* `--is_training 1 \`：指定当前任务模式为训练模式（1 代表训练，0 代表仅测试）。
* `--root_path <你的数据集目录> \`：指定数据集文件所在的根目录路径。
* `--data_path <你的数据集文件名> \`：指定要加载的具体数据集 CSV 文件名。
* `--model_id <你的实验ID> \`：实验 ID。将作为日志、权重保存文件夹名称的一部分，用于区分不同的实验。
* `--model $model_name \`：指定要实例化的模型类名（这里解析为 `FCGformer`）。
* `--data custom \`：指定数据加载器的类型，`custom` 表示使用自定义格式的 CSV 数据加载器。
* `--features <你的预测任务类型> \`：预测任务类型。`MS` 代表多变量输入预测单变量输出 (Multivariate to Univariate)。其他选项如 `M` 代表多变到多变，`S` 代表单变到单变。
* `--seq_len <你的输入序列长度> \`：输入序列长度。即模型观测的过去历史时间步的数量。
* `--pred_len <你的预测序列长度> \`：预测序列长度。即模型需要预测的未来时间步的数量。
* `--enc_in <你的编码器 (Encoder) 的输入特征维度> \`：编码器 (Encoder) 的输入特征维度（变量数量）。
* `--dec_in <你的解码器 (Decoder) 的输入特征维度> \`：解码器 (Decoder) 的输入特征维度。
* `--c_out <你的模型的输出特征维度> \`：模型的输出特征维度。
* `--des <你的实验描述性标签> \`：实验的描述性标签，会附加在保存结果的文件夹名称后面。
* `--d_model <你的模型内部隐藏层的特征维度大小> \`：模型内部隐藏层的特征维度大小。
* `--d_ff <你的前馈神经网络层 (Feed Forward Network) 的隐藏层维度大小> \`：前馈神经网络层 (Feed Forward Network) 的隐藏层维度大小。
* `--itr 1 \`：实验重复运行的次数。为了减小随机误差，可以设置大于 1 的值，这里只运行 1 次。
* `--batch_size <每次训练传入模型的批量数据大小 (Batch Size)> \`：每次训练传入模型的批量数据大小 (Batch Size)。
* `--patience <早停机制 (Early Stopping) 的容忍度> \`：早停机制 (Early Stopping) 的容忍度。如果验证集 Loss 连续 3 个 Epoch 没有下降，则提前终止训练。
* `--dropout <Dropout 层的丢弃概率> \`：Dropout 层的丢弃概率，用于防止模型过拟合。
* `--e_layers <编码器 (Encoder) 包含的网络层数> \`：编码器 (Encoder) 包含的网络层数。
* `--n_heads <多头注意力机制 (Multi-Head Attention) 的头数> \`：多头注意力机制 (Multi-Head Attention) 的头数。
* `--d_layers <解码器 (Decoder) 包含的网络层数> \`：解码器 (Decoder) 包含的网络层数。
* `--num_workers <PyTorch DataLoader 加载数据时使用的后台工作线程数> \`：PyTorch DataLoader 加载数据时使用的后台工作线程数。
* `--learning_rate <优化器（Adam）的初始学习率> \`：优化器（Adam）的初始学习率。
* `--dilation_channel <膨胀卷积（通常用在 WaveNet 结构中）的通道维度大小> \`：膨胀卷积（通常用在 WaveNet 结构中）的通道维度大小。
* `--residual_channels <残差连接层的通道维度大小> \`：残差连接层的通道维度大小。
* `--features_len <数据集中总的特征数量> \`：数据集中总的特征数量。
* `--lradj <学习率衰减调整策略> \`：学习率衰减调整策略。
