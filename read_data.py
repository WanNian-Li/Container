import pickle
with open('dataset/10-4-2custom_dataset_train.pkl', 'rb') as f:
    dataset = pickle.load(f)

print(f"样本总数为：{len(dataset)}")
label = input("需要查看第几个数据样本:")

input_tensor, global_feature_tensor, label = dataset[eval(label)]

print("输入堆叠状态 input_tensor：", input_tensor.shape)
print(input_tensor)

print("全局特征 global_feature_tensor：", global_feature_tensor.shape)
print(global_feature_tensor)

print("标签 label：", label)
