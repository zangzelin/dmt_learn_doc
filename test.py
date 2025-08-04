import torch
import numpy as np
from dmt_learn import DMTLearn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import os

def main():
    # 设置数据预处理
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

    # 将图像展平为784维向量
    data_list = []
    label_list = []
    for i in range(len(train_data)):
        img_array = train_data[i][0].numpy().squeeze()
        data_list.append(img_array)
        label_list.append(train_data[i][1])
    DATA = np.stack(data_list).reshape((-1, 784))
    LABEL = np.array(label_list)

    print(f"DATA.shape: {DATA.shape}")

    # 初始化并训练 DMT 模型
    dmt = DMTLearn(
        random_state=0,
        max_epochs=1000,
        temp=1,
        loss_type='G'
    )

    vis_data = dmt.fit_transform(DATA)
    print(f"vis_data.shape: {vis_data.shape}")

    # 保存降维结果
    plt.figure(figsize=(10, 8))
    plt.scatter(vis_data[:, 0], vis_data[:, 1], marker='.', c=LABEL, cmap='tab10', s=0.5)
    plt.colorbar()
    plt.savefig('output_images/dimensionality_reduction.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)  # 强制使用 spawn，适配 Mac 和 Lightning
    main()