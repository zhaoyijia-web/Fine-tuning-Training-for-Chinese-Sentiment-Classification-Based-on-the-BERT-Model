from datasets import load_dataset,load_from_disk

# 加载缓存数据
datasets = load_from_disk(r"./data/ChnSentiCorp")
print(datasets)

train_data = datasets["test"]
for data in train_data:
    print(data)

# 扩展：加载CSV格式数据
# dataset = load_dataset(path="csv",data_files=r"./data/hermes-function-calling-v1.csv")
# print(dataset)