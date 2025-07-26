import pickle
#==============修改需要读取的数据集参数==============#
stack = 7
tier  = 5
n     = 3
#===================================================#
total_num = stack * tier - 1
with open(f'dataset/{stack}-{tier}-{total_num}/{stack}-{tier}-{n}custom_dataset_train.pkl', 'rb') as f:
    dataset = pickle.load(f)

print(f"样本总数为：{len(dataset)}")
label = input("需要查看第几个数据样本:")

input_tensor, global_feature_tensor, label = dataset[eval(label)]

print("输入堆叠状态 input_tensor：", input_tensor.shape)
print(input_tensor)

print("全局特征 global_feature_tensor：", global_feature_tensor.shape)
print(global_feature_tensor)

print("标签 label：", label)
