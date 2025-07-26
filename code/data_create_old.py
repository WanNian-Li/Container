# 3 
# 生成真正运行的数据集，保存在dataset里，包含训练数据和测试数据
# 对于训练集的数据而言，贝的大小为stack × tier，一种类别的贝有train_size_per_target个样本（类别以容器数totalnumber划分）
# 对于测试集的数据而言，贝的大小为stack × tier，一种类别的贝有test_size_per_target个样本（类别以容器数totalnumber划分）
# train_size ：一个数据集的样本数，例如10×4的情况集装箱，包含的样本数有 4096*37 = 151552 个样本，即 train_size = 151552

from gel import generate21, generate21_optimized
from gel import standardlize
from gel import LBlow_exact
from Dataset import CustomDataset1
import numpy as np
import torch
import os
import pickle

stack=5
tier=4
totalnumber=stack * tier - 1   # 总的集装箱数量（最多情况是堆满所有位置减1）
Boundnum_high = stack * tier - tier + 1
train_size_per_target = 4096     # 每个目标值要生成的训练样本数量
test_size_per_target = 1024       # 训练集与测试集按4:1划分（4096/1024 = 4）
train_size = train_size_per_target * (totalnumber - 2) # 从目标 target = 2 开始生成数据（跳过0和1），直到 totalnumber-1，一共是 totalnumber - 2 种 target 值
test_size = test_size_per_target * (totalnumber - 2)  
print('本次生成参数：{}-{}-{}翻箱问题，train_size={}，test_size={}'.format(stack, tier, totalnumber,  train_size, test_size))

for n in range(1,7):
# 生成训练集
    save_directory = f'dataset/{stack}-{tier}-{totalnumber}'
    os.makedirs(save_directory, exist_ok=True)
    print(f'正在生成第{n}个数据集')
    inputs = []
    global_features = []
    labels = []
    count = [0 for i in range(Boundnum_high)]
    while True:
        for target in range(2, Boundnum_high):
            XX = generate21(stack,tier,target+1)
            
            label = LBlow_exact(XX, stack, tier)
            if label == -1:
                continue
            count[target] = count[target] + 1
            if count[target] == train_size_per_target + 1:
                count[target] = count[target] - 1
                continue
            nowbayss, global_feature = standardlize(XX, stack, tier)
            input = nowbayss.reshape(stack, tier)
            inputs.append([input])
            global_features.append([global_feature])
            labels.append([label])
        if all(x == train_size_per_target for x in count[2:]):
            break
    
    count = [0 for i in range(Boundnum_high, totalnumber)]
    while True:
        for target in range(Boundnum_high, totalnumber):
            XX = generate21(stack,tier,target+1)
            label = LBlow_exact(XX, stack, tier)
            if label == -1:
                continue
            target = target - Boundnum_high
            count[target] = count[target] + 1
            if count[target] == train_size_per_target + 1:
                count[target] = count[target] - 1
                continue
            nowbayss, global_feature = standardlize(XX, stack, tier)
            input = nowbayss.reshape(stack, tier)
            inputs.append([input])
            global_features.append([global_feature])
            labels.append([label])
        if all(x == train_size_per_target for x in count):
            break
    inputs = np.asarray(inputs)
    global_features = np.asarray(global_features)
    labels = np.asarray(labels)
    custom_dataset_train = CustomDataset1(inputs, global_features, labels)
    print('   训练集已全部生成')
    # 生成测试集
    inputs = []
    global_features = []
    labels = []
    count = [0 for i in range(Boundnum_high)]
    while True:
        for target in range(2, Boundnum_high):
            XX = generate21(stack,tier,target+1)
            label = LBlow_exact(XX, stack, tier)
            if label == -1:
                continue
            count[target] = count[target] + 1
            if count[target] == test_size_per_target + 1:
                count[target] = count[target] - 1
                continue
            nowbayss, global_feature = standardlize(XX, stack, tier)
            input = nowbayss.reshape(stack, tier)
            inputs.append([input])
            global_features.append([global_feature])
            labels.append([label])
        if all(x == test_size_per_target for x in count[2:]):
            break
    count = [0 for i in range(Boundnum_high, totalnumber)]
    while True:
        for target in range(Boundnum_high, totalnumber):
            XX = generate21(stack,tier,target+1)
            label = LBlow_exact(XX, stack, tier)
            if label == -1:
                continue
            target = target - Boundnum_high
            count[target] = count[target] + 1
            if count[target] == test_size_per_target + 1:
                count[target] = count[target] - 1
                continue
            nowbayss, global_feature = standardlize(XX, stack, tier)
            input = nowbayss.reshape(stack, tier)
            inputs.append([input])
            global_features.append([global_feature])
            labels.append([label])
        if all(x == test_size_per_target for x in count):
            break
    inputs = np.asarray(inputs)
    global_features = np.asarray(global_features)
    labels = np.asarray(labels)
    custom_dataset_test = CustomDataset1(inputs, global_features, labels)
    print('   测试集已全部生成\n   对数据集进行存储')


    with open(os.path.join(save_directory, str(stack) + '-' + str(tier) + '-' + str(n) + 'custom_dataset_train.pkl'), 'wb') as datase_train:
        pickle.dump(custom_dataset_train, datase_train)
    with open(os.path.join(save_directory, str(stack) + '-' + str(tier) + '-' + str(n) +   'custom_dataset_test.pkl'), 'wb') as datase_test:
        pickle.dump(custom_dataset_test, datase_test)
    print('数据集已完成存储')