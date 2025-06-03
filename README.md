# 高级推荐系统实验平台

本项目实现并集成了多种经典与深度学习推荐算法，支持特征工程、自动降维、聚类、混合建模和可视化分析。可一键完成数据加载、模型训练、评估、保存与复现，适用于课程实验、学术研究和工程测试。

## 特性

- **支持算法丰富**  
  - 传统协同过滤（SVD、基于用户/物品的KNN、NMF、Baseline）
  - 多种深度学习推荐模型（NCF、MF、NeuMF、FM 及混合特征版本）

- **完备的实验流程**  
  - 数据加载与分析
  - 特征工程（统计+图特征）、自编码降维、聚类
  - 批量模型训练，交叉验证与测试集评估
  - 可视化与日志管理
  - 支持模型保存/加载、单用户预测与Top-K推荐

- **易用的命令行接口**  
  - 一键运行，参数化训练/测试/分析
  - 灵活指定模型、数据集、超参数和保存路径

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模式

#### 使用默认参数（全流程）

```bash
python main.py
```

#### 指定模型和参数

```bash
python main.py --mode train --models ncf mf --epochs 30 --learning-rate 0.005
```

#### 只使用传统算法（不启用深度学习）

```bash
python main.py --no-deep-learning
```

#### 自定义数据集和参数

```bash
python main.py --dataset ml-1m --test-size 0.3 --batch-size 512 --embedding-dim 128
```

---

### 3. 测试模式（加载已训练模型）

```bash
python main.py --mode test --model-path models/saved/ncf_20241128_143022.pth
```

---

## 常用参数说明

| 参数                 | 说明                                              | 示例                          |
|----------------------|---------------------------------------------------|-------------------------------|
| `--mode`             | 训练或测试模式，`train` 或 `test`                 | `--mode train`                |
| `--models`           | 指定模型列表，空格分隔，如 `ncf mf neumf fm`      | `--models ncf mf`             |
| `--no-deep-learning` | 只运行传统算法                                    | `--no-deep-learning`          |
| `--dataset`          | 数据集名称（如 ml-100k、ml-1m）                   | `--dataset ml-1m`             |
| `--test-size`        | 测试集占比（0~1）                                 | `--test-size 0.3`             |
| `--epochs`           | 训练轮数                                          | `--epochs 30`                 |
| `--batch-size`       | 批量大小                                          | `--batch-size 512`            |
| `--embedding-dim`    | Embedding 维度                                    | `--embedding-dim 128`         |
| `--learning-rate`    | 学习率                                            | `--learning-rate 0.005`       |
| `--autoencoder-hidden-dims` | 自编码器隐藏层维度列表，逗号分隔           | `--autoencoder-hidden-dims 128,64` |
| `--autoencoder-epochs`      | 自编码器训练轮数                            | `--autoencoder-epochs 20`     |
| `--n-clusters`       | KMeans 聚类数                                    | `--n-clusters 10`             |
| `--model-dir`        | 模型保存目录                                      | `--model-dir models/saved`    |
| `--model-path`       | 测试时指定模型路径                                | `--model-path models/saved/ncf_xxx.pth` |
| `--log-file`         | 日志文件名                                        | `--log-file run.log`          |
| `--log-level`        | 日志等级（INFO/DEBUG/WARNING/ERROR）              | `--log-level INFO`            |

---

## 结果输出与可视化

- **终端输出**：各算法训练/测试集 RMSE、MAE，最佳模型信息等
- **日志文件**：详细运行与异常日志
- **可视化报告**：自动生成数据分布与算法性能对比图（如 PDF/PNG）

---

## 实验结果参考

### 传统算法（ml-100k）

| 算法                | CV RMSE         | CV MAE         | Test RMSE | Test MAE |
|---------------------|-----------------|---------------|-----------|----------|
| SVD                 | 0.9361 ± 0.0055 | 0.7378 ± 0.0038| 0.9348    | 0.7369   |
| 基于用户协同过滤     | 0.9564 ± 0.0030 | 0.7546 ± 0.0029| 0.9538    | 0.7526   |
| 基于物品协同过滤     | 0.9438 ± 0.0049 | 0.7418 ± 0.0038| 0.9402    | 0.7399   |
| NMF                 | 1.0289 ± 0.0037 | 0.7847 ± 0.0040| 1.0284    | 0.7842   |
| Baseline            | 0.9437 ± 0.0028 | 0.7481 ± 0.0016| 0.9442    | 0.7490   |

### 深度学习算法（ml-1m）

| 算法类型                      | CV RMSE      | CV MAE      | Test RMSE | Test MAE |
|-------------------------------|--------------|-------------|-----------|----------|
| DeepLearning_NCF              | 0.9167       | 0.7141      | 0.9167    | 0.7141   |
| DeepLearning_MF               | 1.2647       | 0.9560      | 1.2647    | 0.9560   |
| DeepLearning_NEUMF            | 0.9131       | 0.7149      | 0.9131    | 0.7149   |
| DeepLearning_FM               | 0.8903       | 0.6945      | 0.8903    | 0.6945   |
| DeepLearning_NCF_HYBRID       | 0.9156       | 0.7099      | 0.9156    | 0.7099   |
| DeepLearning_MF_HYBRID        | 0.9903       | 0.7704      | 0.9903    | 0.7704   |
| DeepLearning_NEUMF_HYBRID     | 0.9144       | 0.7122      | 0.9144    | 0.7122   |
| DeepLearning_FM_HYBRID        | 0.8894       | 0.6951      | 0.8894    | 0.6951   |


---

## 常见问题与说明

- 训练/测试时如遇显存不足，可降低 `--batch-size` 或 `--embedding-dim`。
- 推荐使用 Python 3.8+，依赖见 requirements.txt。
- 如需自定义数据集，请参考 `data/` 目录的数据格式示例。
- 日志与可视化文件默认输出在当前目录下，可通过命令行参数自定义位置。

---

## 参考文献

- [Surprise 推荐系统库](https://surpriselib.com/)
- [PyTorch 官方文档](https://pytorch.org/)
- He, X. et al. "Neural Collaborative Filtering", WWW 2017.
- Rendle, S. "Factorization Machines", ICDM 2010.
- Darban, Z. Z., & Valipour, M. H. (2022). GHRS: Graph-based hybrid recommendation system with application to movie recommendation. Expert Systems with Applications, 116850.
- Harper, F. M., & Konstan, J. A. (2015). The movielens datasets: History and context. Acm transactions on interactive intelligent systems (tiis), 5(4), 1-19.
