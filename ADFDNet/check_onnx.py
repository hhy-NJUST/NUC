import torch
import numpy as np
import onnxruntime
# from models.ecbsr import ECBSR
# from models.plainsr import PlainSR
from model.cbdnet import Network


def torch_model():
    device = torch.device('cpu')
    ## definitions of model, loss, and optimizer
    # model_ecbsr = ECBSR(module_nums=4, channel_nums=16, with_idt=0, act_type='prelu', scale=4, colors=1).to(device)
    # model_plain = PlainSR(module_nums=4, channel_nums=16, act_type='prelu', scale=4, colors=1).to(device)

    # print("load pretrained model: {}!".format("/home/jl/Project/ECBSR/experiments/Visible-light-1channel-noise5-psnr/models/model_x4_514.pt"))
    # model_ecbsr.load_state_dict(torch.load("/home/jl/Project/ECBSR/experiments/Visible-light-1channel-noise5-psnr/models/model_x4_514.pt", map_location='cpu'))

    model = Network()

    print("load pretrained model: {}!".format("save_model/checkpoint_2001.pth.tar"))

    state_dict = torch.load('save_model/checkpoint_2001.pth.tar')["state_dict"]

    # 创建一个新的state_dict，其键没有'module.'前缀
    from collections import OrderedDict

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]  # 删除'module.'前缀
        new_state_dict[name] = v

    # 加载新的state_dict
    model.load_state_dict(new_state_dict)
    return model

    ## copy weights from ecbsr to plainsr
    # depth = len(model_ecbsr.backbone)
    # for d in range(depth):
    #     module = model_ecbsr.backbone[d]
    #     act_type = module.act_type
    #     RK, RB = module.rep_params()
    #     model_plain.backbone[d].conv3x3.weight.data = RK
    #     model_plain.backbone[d].conv3x3.bias.data = RB
    #
    #     if act_type == 'relu':     pass
    #     elif act_type == 'linear': pass
    #     elif act_type == 'prelu':  model_plain.backbone[d].act.weight.data = module.act.weight.data
    #     else: raise ValueError('invalid type of activation!')
    # return model_ecbsr


def pytorch_out(input):
    model = torch_model()  # model.eval
    # input = input.cuda()
    # model.cuda()
    torch.no_grad()
    model.eval()
    output = model(input)
    # print output[0].flatten()[70:80]
    out1 = output[0]
    out2 = output[1]
    out = torch.stack((out1, out2))
    return out


def pytorch_onnx_test():
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # 测试数据
    torch.manual_seed(66)
    dummy_input = torch.randn(1, 3, 480, 480, device='cpu')

    sess = onnxruntime.InferenceSession("./export_onnx/CDBNet_0417.onnx")

    # onnx 网络输出
    onnx_out = np.array(sess.run(None, {"inputs": to_numpy(dummy_input)}))  # fc 输出是三维列表
    print("==============>")
    print(onnx_out)
    print(onnx_out.shape)
    print("==============>")
    torch_out_res = pytorch_out(dummy_input).detach().numpy()  # fc输出是二维 列表
    print(torch_out_res)
    print(torch_out_res.shape)

    print("===================================>")
    print("输出结果验证小数点后四位是否正确,都变成一维np")

    torch_out_res = torch_out_res.flatten()
    onnx_out = onnx_out.flatten()

    pytor = np.array(torch_out_res, dtype="float32")  # need to float32
    onn = np.array(onnx_out, dtype="float32")  ##need to float32
    np.testing.assert_almost_equal(pytor, onn, decimal=5)  # 精确到小数点后4位，验证是否正确，不正确会自动打印信息
    print("恭喜你 ^^ , onnx 和 pytorch 结果一致, Exported model has been executed decimal=5 and the result looks good!")


pytorch_onnx_test()
