import torch
import numpy as np


# 1. 必须定义和训练时一模一样的模型结构
class TinyHandModel(torch.nn.Module):
    def __init__(self):
        super(TinyHandModel, self).__init__()
        self.fc1 = torch.nn.Linear(63, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 3)
        self.relu = torch.nn.ReLU()


# 2. 加载模型
model = TinyHandModel()
model.load_state_dict(torch.load("hand_model.pth", map_location='cpu'))
model.eval()


def format_float_list(arr):
    return "{" + ", ".join([f"{x:.8f}f" for x in arr.flatten()]) + "}"


# 3. 提取权重并生成 C++ 头文件
with open("model_data.h", "w") as f:
    f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")

    for name, param in model.named_parameters():
        name = name.replace(".", "_")
        shape = list(param.shape)
        data = param.detach().numpy()

        f.write(f"// Shape: {shape}\n")
        f.write(f"const float {name}[] = {format_float_list(data)};\n\n")

    f.write("#endif")

print("🎉 导出成功！请查看项目文件夹下的 model_data.h")