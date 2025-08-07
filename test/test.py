"""
使用 DMTLearn 模型对 MNIST 数据集进行降维可视化
- 数据预处理与加载
- 可视化图像保存
- DMT 模型训练与降维
- 结果保存与绘图
"""

# ===== 导入依赖 =====
import torch
import numpy as np
from dmt_learn import DMTLearn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import os

# ===== 工具函数 =====
def flatten_mnist_data(dataset):
    """
    将MNIST数据集中的图像展平为向量，并提取标签。
    """
    data_list, label_list = [], []
    for img, label in dataset:
        data_list.append(img.numpy().squeeze())
        label_list.append(label)
    return np.stack(data_list).reshape((-1, 784)), np.array(label_list)

# ===== 主流程函数 =====
def main(all_g_l_weight=0.5):
    # 设置精度模式
    torch.set_float32_matmul_precision('medium')
    transform = transforms.Compose([transforms.ToTensor()])

    path = 'data/E9.5_E1S1.MOSTA.h5ad'
    
    import scanpy as sc
    adata = sc.read_h5ad(path)
    
    data = adata.X.toarray()
    high_var_genes_500 = np.var(data, axis=0).argsort()[-500:]
    data = data[:, high_var_genes_500]
    
    LABEL_list_str = list(adata.obs['annotation'].values)
    laebl_set = list(set(LABEL_list_str))
    laebl_set.sort()
    LABEL_list = [laebl_set.index(label) for label in LABEL_list_str]
    
    DATA = data
    LABEL = np.array(LABEL_list)


    # 初始化并训练模型
    dmt = DMTLearn(
        random_state=0,
        max_epochs=300,
        temp=1,
        nu=0.002,
        loss_type='A',
        n_neighbors=3,
        all_g_l_weight=all_g_l_weight,
    )

    # 使用 DMT 模型进行降维
    vis_data = dmt.fit_transform(DATA)
    print(f"vis_data.shape: {vis_data.shape}")

    # 绘制并保存可视化结果
    plt.figure(figsize=(10, 8))
    plt.scatter(vis_data[:, 0], vis_data[:, 1], marker='.', c=LABEL, cmap='tab20', s=0.5)
    plt.colorbar()
    plt.savefig(f'output_images/dimensionality_reduction{all_g_l_weight}.png', dpi=300)
    plt.close()

# ===== 脚本运行入口 =====
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)  # 强制使用 spawn，适配 Mac 和 Lightning
    main(all_g_l_weight=0.1)
    main(all_g_l_weight=0.2)
    main(all_g_l_weight=0.3)
    main(all_g_l_weight=0.4)
    main(all_g_l_weight=0.5)
    main(all_g_l_weight=0.6)
    main(all_g_l_weight=0.7)
    main(all_g_l_weight=0.8)
    main(all_g_l_weight=0.9)
    main(all_g_l_weight=1.0)