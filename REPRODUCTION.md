# DeDiff 本机复现记录

复现日期：2026-08-18  
设备：NVIDIA GeForce RTX 5090 D（32 GB）  
已验证环境：`D:\conda\envs\cgt_gpu128\python.exe`

## 环境

- Python 3.10.20
- PyTorch 2.7.0+cu128
- PyTorch Geometric 2.7.0
- torch-scatter 2.1.2+pt27cu128
- NumPy 1.26.4
- SciPy 1.15.3
- NetworkX 3.4.2

仓库的 `requirements.txt` 不能直接用于本机复现：其中同时固定了 NetworkX 2.8.4 和 3.1，且 PyTorch 2.0.1 不支持 RTX 5090（sm_120）。本次没有覆盖原 requirements，而是复用了已验证的 CUDA 12.8 环境。

## 运行方式

在 PowerShell 中执行：

```powershell
cd D:\DeDiff
.\reproduce.ps1 -Dataset christian -Epochs 50
.\reproduce.ps1 -Dataset android -Epochs 50
```

等价的训练参数为：batch size 64、learning rate 5e-4、10 attention heads、seed 21、50 epochs。

## 实测结果

| Dataset | Source | H@10 | H@50 | H@100 | M@10 | M@50 | M@100 |
|---|---|---:|---:|---:|---:|---:|---:|
| Christianity | Paper | 0.3281 | 0.5357 | 0.6339 | 0.2098 | 0.2186 | 0.2199 |
| Christianity | Reproduced | 0.3304 | 0.5268 | 0.6161 | 0.2003 | 0.2093 | 0.2105 |
| Android | Paper | 0.1180 | 0.2220 | 0.2958 | 0.0752 | 0.0798 | 0.0808 |
| Android | Reproduced | 0.1079 | 0.2034 | 0.2935 | 0.0698 | 0.0742 | 0.0755 |

独立加载 checkpoint 后的评测值与训练进程报告值一致。

## 输出

- `checkpoint/christian.bin`（约 15.2 MB）
- `checkpoint/android.bin`（约 42.7 MB）
- `dataset/christian/frequency.pt`
- `dataset/android/frequency.pt`

## 无法严格逐数值复现的原因

1. 论文写明按验证集 MAP@100 选择 checkpoint；`main.py` 实际每轮在测试集评估，并按六项测试指标之和保存 checkpoint。
2. 论文写明嵌入维度统一为 64；代码会把 Android 的维度强制覆盖为 128。
3. 论文描述低秩 `P Q^T` 掩码并只在稀疏观测边上计算；`model.py` 实际使用完整的 `N x N` 参数矩阵，并执行稠密矩阵乘法。
4. 论文表 1 与随仓库数据不一致：Android/Christianity 用户数写为 9,958/2,897，而文件矩阵规模为 2,929/1,653；Twitter 的级联数写为 3,442，实际为 3,435。
5. 当前 disagreement/triplet 项没有 hinge 截断，训练总损失会持续变为较大的负数；虽然排序指标能够收敛，但这与常规 triplet 目标不一致。

由于第 3 点，Twitter 和 Douban 会触发大规模 `N x N` 稠密参数、邻接和近似立方复杂度运算；在不把实现改写成论文所述稀疏低秩版本前，不宜把现有代码的长时间运行当作可信的论文复现。
