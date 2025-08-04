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
def main():
    # 设置精度模式
    torch.set_float32_matmul_precision('medium')
    transform = transforms.Compose([transforms.ToTensor()])

    # 加载 MNIST 数据集
    train_data = MNIST(root='data', train=True, download=True, transform=transform)

    # 使用 DataLoader 增加数据加载效率
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=48, pin_memory=True)

    # 显示并保存前40张图像
    os.makedirs('output_images', exist_ok=True)
    fig, axes = plt.subplots(5, 8, figsize=(10, 6), facecolor='white')
    for i, ax in enumerate(axes.flat):
        ax.imshow(train_data[i][0].numpy().squeeze(), cmap='gray_r')
        ax.set_title(f"Label: {train_data[i][1]}", color='black')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('output_images/mnist_images.png', dpi=300)
    plt.close()

    # 图像展平为向量
    DATA, LABEL = flatten_mnist_data(train_data)
    print(f"DATA.shape: {DATA.shape}")

    # 初始化并训练模型
    dmt = DMTLearn(
        random_state=0,
        max_epochs=200,
        temp=1,
        nu=0.002,
        loss_type='G',
        n_neighbors=3,
    )

    # 使用 DMT 模型进行降维
    vis_data = dmt.fit_transform(DATA)
    print(f"vis_data.shape: {vis_data.shape}")

    # 绘制并保存可视化结果
    plt.figure(figsize=(10, 8))
    plt.scatter(vis_data[:, 0], vis_data[:, 1], marker='.', c=LABEL, cmap='tab10', s=0.5)
    plt.colorbar()
    plt.savefig('output_images/dimensionality_reduction.png', dpi=300)
    plt.close()

# ===== 脚本运行入口 =====
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)  # 强制使用 spawn，适配 Mac 和 Lightning
    main()