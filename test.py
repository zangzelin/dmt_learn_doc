import torch
import numpy as np
from dmt_learn import DMTLearn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# 设置数据预处理
torch.set_float32_matmul_precision('medium')
transform = transforms.Compose([transforms.ToTensor()])

# 加载 MNIST 数据集
train_data = MNIST(root='data', train=True, download=True, transform=transform)

# 使用 DataLoader 增加数据加载效率
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=48, pin_memory=True)

# Display the first 40 images with a white background
fig, axes = plt.subplots(5, 8, figsize=(10, 6), facecolor='white')
for i, ax in enumerate(axes.flat):
    ax.imshow(train_data[i][0].numpy().squeeze(), cmap='gray_r')  # Use 'gray_r' to invert colors
    ax.set_title(f"Label: {train_data[i][1]}", color='black')  # Set title color to black for visibility
    ax.axis('off')
plt.tight_layout()
plt.savefig('output_images/mnist_images.png', dpi=300)
plt.close()  # 关闭当前图形

# 数据预处理：将图像展平为784维的向量
data_list = []
label_list = []

for i in range(len(train_data)):
    img_array = train_data[i][0].numpy().squeeze()
    data_list.append(img_array)
    label_list.append(train_data[i][1])

# 转换数据为 NumPy 数组
DATA = np.stack(data_list).reshape((-1, 784))
LABEL = np.array(label_list)

print(f"DATA.shape: {DATA.shape}")  # output: (60000, 784)

# 初始化 DMT-EV 模型
dmt = DMTLearn(
    random_state=0,  # Set a random seed for reproducibility
    max_epochs=100,  # Number of training epochs
    nu=1.5,
    temp=2.0,
)

# Fit the model and transform the dataset into a lower-dimensional space
vis_data = dmt.fit_transform(DATA)

print(f"vis_data.shape: {vis_data.shape}")  # output: (60000, 2)

# 保存降维后的散点图
plt.figure(figsize=(10, 8))
plt.scatter(vis_data[:, 0], vis_data[:, 1], marker='.', c=LABEL, cmap='tab10', s=0.5)
plt.colorbar()
plt.savefig('output_images/dimensionality_reduction.png', dpi=300)
plt.close()  # 关闭当前图形