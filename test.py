import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from Networks import Vmamba
# 测试自定义模型
model = Vmamba.Backbone_VSSM(pretrained=False)
input_tensor = torch.rand(1, 3, 224, 224)
stage_outputs = model(input_tensor)

# 输出每个阶段的输出尺寸以及 Regression 模型的输出
for i, output in enumerate(stage_outputs):
    print(f"Stage {i+1} output size: {output.size()}")
print(f"Regression output size: {y.size()}")