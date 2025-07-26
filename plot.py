import os
import re
import matplotlib.pyplot as plt

stack = 9

tier  = 4
model = "CNN"
# 读取文件并解析成浮点数列表
def read_tensor_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    # 匹配所有浮点数
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", content)
    return [float(x) for x in numbers]

# 四个文件路径
file_acc1 = f"result/{stack}-{tier}-{model}/{stack}-{tier}test_accTOP1.txt"
file_acc2 = f"result/{stack}-{tier}-{model}/{stack}-{tier}test_accTOP2.txt"
file_mae  = f"result/{stack}-{tier}-{model}/{stack}-{tier}test_loss_mae.txt"
file_loss = f"result/{stack}-{tier}-{model}/{stack}-{tier}test_loss.txt"

# 读取数据
acc1 = read_tensor_file(file_acc1)
acc2 = read_tensor_file(file_acc2)
mae = read_tensor_file(file_mae)
loss = read_tensor_file(file_loss)

# 绘图
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# accTOP1
axs[0, 0].plot(acc1, label='Acc Top1', color='blue')
axs[0, 0].set_title('Accuracy Top-1')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].grid(True)

# accTOP2
axs[0, 1].plot(acc2, label='Acc Top2', color='green')
axs[0, 1].set_title('Accuracy Top-2')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Accuracy')
axs[0, 1].grid(True)

# loss_mae
axs[1, 0].plot(mae, label='MAE', color='orange')
axs[1, 0].set_title('MAE Loss')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Loss')
axs[1, 0].grid(True)

# loss
axs[1, 1].plot(loss, label='Loss', color='red')
axs[1, 1].set_title('Loss')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Loss')
axs[1, 1].grid(True)

plt.tight_layout()

save_directory = f"fig"
os.makedirs(save_directory,exist_ok = True)
plt.savefig(f"fig/{stack}-{tier}-{model}.png", dpi=300)  # 保存图像
print(f"图像已保存到 fig/{stack}-{tier}-{model}.png")