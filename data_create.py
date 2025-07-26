# 3 
# 生成真正运行的数据集，保存在dataset里，包含训练数据和测试数据
# 对于训练集的数据而言，贝的大小为stack × tier，一种类别的贝有train_size_per_target个样本（类别以容器数totalnumber划分）
# 对于测试集的数据而言，贝的大小为stack × tier，一种类别的贝有test_size_per_target个样本（类别以容器数totalnumber划分）
# train_size ：一个数据集的样本数，例如10×4的情况集装箱，包含的样本数有 4096*37 = 151552 个样本，即 train_size = 151552

import os
import pickle
import numpy as np
from gel import generate21, standardlize, LBlow_exact, generate21_optimized
from Dataset import CustomDataset1

STACK = 7
TIER = 6


def generate_dataset(stack, tier, sample_size, target_range):
    """生成单个数据集（训练或测试）"""
    inputs, global_features, labels = [], [], []
    count = {target: 0 for target in target_range}

    while True:
        for target in target_range:
            config = generate21_optimized(stack, tier, target + 1)
            print(config)
            label = LBlow_exact(config, stack, tier)
            if label == -1:
                continue

            if count[target] >= sample_size:
                continue

            bays, global_feature = standardlize(config, stack, tier)
            inputs.append(bays.reshape(stack, tier))
            global_features.append(global_feature)
            labels.append(label)
            count[target] += 1

        if all(v >= sample_size for v in count.values()):
            break

    return np.array(inputs), np.array(global_features), np.array(labels)


def main():
    stack, tier = STACK, TIER

    total_number = stack * tier - 1
    bound_high = stack * tier - tier + 1

    train_per_target = 4096
    test_per_target = 1024
    train_targets = range(2, total_number)
    test_targets = range(2, total_number)

    # 新建 dataset/{stack}-{tier} 文件夹
    save_directory = os.path.join('dataset', f'{stack}-{tier}-{total_number}')
    os.makedirs(save_directory, exist_ok=True)

    print(f'参数：stack={stack}, tier={tier}, total={total_number}, '
          f'train_size={train_per_target*(total_number-2)}, '
          f'test_size={test_per_target*(total_number-2)}')

    for n in range(1, 7):
        print(f'正在生成第 {n} 个数据集')

        # 训练集
        train_inputs, train_globals, train_labels = generate_dataset(stack, tier, train_per_target, train_targets)
        train_dataset = CustomDataset1(train_inputs, train_globals, train_labels)
        print('训练集已生成')

        # 测试集
        test_inputs, test_globals, test_labels = generate_dataset(stack, tier, test_per_target, test_targets)
        test_dataset = CustomDataset1(test_inputs, test_globals, test_labels)
        print('测试集已生成')

        # 文件保存到 dataset/{stack}-{tier}/
        train_path = os.path.join(save_directory, str(stack) + '-' + str(tier) + '-' + str(total_number) + 'custom_dataset_train.pkl')
        test_path = os.path.join(save_directory, str(stack) + '-' + str(tier) + '-' + str(total_number) + 'custom_dataset_test.pkl')
        with open(train_path, 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(test_path, 'wb') as f:
            pickle.dump(test_dataset, f)
        print(f'数据集已存储：{train_path}, {test_path}')


if __name__ == "__main__":
    main()