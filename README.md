# 🎯 基于 BERT 模型的中文情感分类微调训练

本项目使用预训练的 BERT 中文模型（bert-base-chinese）对中文情感分析数据集（ChnSentiCorp）进行微调训练，实现了一个二分类情感分析模型。

## 📁 项目结构

```
demo_04/
├── data/                          # 数据目录
│   ├── ChnSentiCorp/             # 中文情感分析数据集
│   │   ├── train/                # 训练集 (9600条)
│   │   ├── validation/           # 验证集 (1200条)
│   │   └── test/                 # 测试集 (1200条)
│   └── hermes-function-calling-v1.csv  # 备用数据集
│
├── model/                         # 模型目录 bert-base-chinese
│   └──/        # BERT 中文预训练模型
│
├── params/                        # 模型参数保存目录
│   ├── best_bert.pth             # 验证集最优参数
│   └── last_bert.pth             # 最后一轮参数
│
├── MyData.py                      # 自定义数据集加载类
├── net.py                         # 下游任务模型定义
├── train_val.py                   # 训练与验证主程序
├── data_test.py                   # 数据加载测试脚本
├── token_test.py                  # 分词器测试脚本
└── README.md                      # 项目说明文档
```

## 🚀 快速开始

### 1. 环境安装

```bash
pip install torch transformers datasets
```

### 2. 数据加载测试

```bash
python data_test.py
```

### 3. 分词器测试

```bash
python token_test.py
```

### 4. 开始训练

```bash
python train_val.py
```

## 📖 代码说明

### MyData.py - 自定义数据集

自定义 PyTorch Dataset 类，用于加载缓存的 Hugging Face 格式数据集：

```python
class MyDataset(Dataset):
    def __init__(self, split):  # split: "train" / "test" / "validation"
        # 从磁盘加载数据
        self.dataset = load_from_disk("./data/ChnSentiCorp")
```

### net.py - 模型定义

下游任务模型设计：
- 使用 BERT 预训练模型提取文本特征
- 冻结 BERT 参数，仅训练新增的全连接层
- 输出 768 维 → 2 维（正/负二分类）

```python
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)  # 全连接层
```

### train_val.py - 训练与验证

主训练脚本包含：
- 数据批处理与编码
- 模型训练（EPOCH=30000）
- 验证集评估
- 自动保存最优参数

## 📊 数据集简介

**ChnSentiCorp** 是中文情感分析经典数据集，包含酒店、书籍、电脑等领域的用户评论：

- **训练集**：9600 条评论
- **验证集**：1200 条评论
- **测试集**：1200 条评论
- **标签**：0（负面）/ 1（正面）

## 🛠️ 技术栈

| 技术 | 用途 |
|------|------|
| **PyTorch** | 深度学习框架 |
| **Transformers** | BERT 模型与分词器 |
| **Datasets** | Hugging Face 数据集管理 |
| **BERT-base-chinese** | 12层中文预训练模型 |

## 📝 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| EPOCH | 30000 | 训练轮次 |
| batch_size | 50 | 批次大小 |
| max_length | 512 | 文本最大长度 |
| learning_rate | AdamW默认 | 优化器学习率 |
| 优化器 | AdamW | 权重衰减优化 |

## 🎯 模型特点

1. **参数冻结**：BERT 参数不参与训练，减少计算开销
2. **增量学习**：仅训练新增的全连接分类层
3. **自动保存**：每轮验证后自动保存最优参数到 `params/best_bert.pth`
4. **实时监控**：每 5 个批次输出训练 loss 和准确率

## 📈 预期效果

在 ChnSentiCorp 验证集上，模型准确率预期可达 **85% 以上**。

## 📄 许可证

本项目仅供学习研究使用。

## 🤝 贡献

欢迎提出问题或 Pull Requests！
