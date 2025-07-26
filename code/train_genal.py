import os
import sys
import json
import pickle
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torcheval.metrics.functional import r2_score
from Module import Genal_CNN
# ============ 参数区 ============
STACK = 10
TIER = 4
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
TRAIN_SIZE_PER_TARGET = 24576
TEST_SIZE_PER_TARGET = 6144
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============ 数据加载 ============
def load_datasets(stack, tier, dataset_dir, n_list=(2,)):
    train_datasets, test_datasets = [], []
    for n in n_list:
        with open(os.path.join(dataset_dir, f"{stack}-{tier}-{n}custom_dataset_train.pkl"), 'rb') as f:
            train_datasets.append(pickle.load(f))
        with open(os.path.join(dataset_dir, f"{stack}-{tier}-{n}custom_dataset_test.pkl"), 'rb') as f:
            test_datasets.append(pickle.load(f))
    return ConcatDataset(train_datasets), ConcatDataset(test_datasets)


# ============ 训练与验证 ============
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc_top1, num_samples = 0, 0, 0
    for data, global_feature, label in dataloader:
        data, global_feature, label = data.to(device), global_feature.to(device), label.to(device)
        global_feature = global_feature.reshape(BATCH_SIZE, -1)
        optimizer.zero_grad()
        output = model(data, global_feature)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        total_acc_top1 += ((output - label).abs() < 0.5).sum().item()
        num_samples += data.size(0)
    return total_loss / num_samples, total_acc_top1 / num_samples


@torch.no_grad()
def evaluate(model, dataloader, criterion_mse, criterion_mae, device):
    model.eval()
    total_loss, total_mae, total_r2, total_acc_top1, total_acc_top2, num_samples = 0, 0, 0, 0, 0, 0
    for data, global_feature, label in dataloader:
        data, global_feature, label = data.to(device), global_feature.to(device), label.to(device)
        global_feature = global_feature.reshape(BATCH_SIZE, -1)
        output = model(data, global_feature)
        total_loss += criterion_mse(output, label).item() * data.size(0)
        total_mae += criterion_mae(output, label).item() * data.size(0)
        total_r2 += r2_score(output, label).item() * data.size(0)
        total_acc_top1 += ((output - label).abs() < 0.5).sum().item()
        total_acc_top2 += ((output - label).abs() < 1).sum().item()
        num_samples += data.size(0)
    return {
        "loss_mse": total_loss / num_samples,
        "mae": total_mae / num_samples,
        "r2": total_r2 / num_samples,
        "acc_top1": total_acc_top1 / num_samples,
        "acc_top2": total_acc_top2 / num_samples
    }


# ============ 主函数 ============
def main():
    total_number = STACK * TIER - 1
    dataset_dir = os.path.join("dataset", f"{STACK}-{TIER}-{total_number}")
    train_dataset, test_dataset = load_datasets(STACK, TIER, dataset_dir)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Genal_CNN().to(DEVICE)
    criterion_mse = torch.nn.MSELoss()
    criterion_mae = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    metrics = {"train_loss": [], "train_acc_top1": [], "test_loss": [], "test_mae": [],
               "test_r2": [], "test_acc_top1": [], "test_acc_top2": []}

    for epoch in range(EPOCHS):
        print('--------------第{}/{}轮迭代开始--------------'.format(epoch + 1, EPOCHS))
        train_loss, train_acc_top1 = train_one_epoch(model, train_loader, criterion_mse, optimizer, DEVICE)
        eval_metrics = evaluate(model, test_loader, criterion_mse, criterion_mae, DEVICE)

        metrics["train_loss"].append(train_loss)
        metrics["train_acc_top1"].append(train_acc_top1)
        metrics["test_loss"].append(eval_metrics["loss_mse"])
        metrics["test_mae"].append(eval_metrics["mae"])
        metrics["test_r2"].append(eval_metrics["r2"])
        metrics["test_acc_top1"].append(eval_metrics["acc_top1"])
        metrics["test_acc_top2"].append(eval_metrics["acc_top2"])

        print(f"  Train Loss: {train_loss:8.4f} | Train AccTop1: {train_acc_top1:6.4f} ")
        print(f"  Test Loss: {eval_metrics['loss_mse']:.4f} | MAE: {eval_metrics['mae']:.4f} | "
              f"R2: {eval_metrics['r2']:.4f} | AccTop1: {eval_metrics['acc_top1']:.4f} | AccTop2: {eval_metrics['acc_top2']:.4f}")

    # 保存模型和指标
    save_dir = f"{STACK}-{TIER}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"{STACK}-{TIER}-{total_number}model.pkl"))
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print("模型与指标已保存")


if __name__ == "__main__":
    main()
