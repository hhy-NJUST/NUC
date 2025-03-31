import torch
import torch.nn as nn
import onnx
import numpy as np
from onnx import load_model, save_model
from onnx.shape_inference import infer_shapes
# from models_DnCNN import DnCNN
from model.cbdnet import Network

# 加载模型
# dncnn_model = DnCNN(input_chnl=1, groups=1)
# dncnn_model = torch.load("./model_DnCNN_datav1-sigma11/model_DnCNN_datav1_epoch_500.pth")["model"]
# dncnn_model.load_state_dict(torch.load("./model_DnCNN_datav1/model_DnCNN_datav1_best.pth", map_location="cuda:1")["model"].state_dict())

model = Network()
# state_dict = torch.load('save_model/checkpoint.pth.tar', map_location=torch.device('cpu'))["state_dict"]
state_dict = torch.load('./save_model/checkpoint_0001.pth.tar')["state_dict"]

# 创建一个新的state_dict，其键没有'module.'前缀
from collections import OrderedDict

new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k[7:]  # 删除'module.'前缀
    print("name", name)
    new_state_dict[name] = v

# 加载新的state_dict
model.load_state_dict(new_state_dict)

# 设置为eval模式，固定bn等操作
# dncnn_model.eval()
# dncnn_model.to("cuda:1")

model.eval()
model.to("cuda:0")
torch.no_grad()
# 设置模型的输入
input = torch.randn((1, 3, 960, 540), dtype=torch.float).to("cuda:0")
torch.onnx.export(model, input, "./export_onnx/CDBNet_01.onnx", input_names=["inputs"], output_names=["outputs"], opset_version=14, verbose=1)

# torch.onnx.export(model, input, "./dncnn-sigma11-light.onnx", input_names=["inputs-jl"], output_names=["outputs-jl"], opset_version=14, verbose=1,
#                   dynamic_axes={"inputs-jl":{2:"inputs_height", 3:"inputs_weight"}, "outputs-jl":{2:"outputs_height", 3:"outputs_weight"}})

print("Model has benn converted to onnx")

# onnx_model = load_model("./dncnn-sigma11.onnx")
# onnx_model = infer_shapes(onnx_model)

# save_model(onnx_model, "dncnn-sigma11-shape.onnx")
