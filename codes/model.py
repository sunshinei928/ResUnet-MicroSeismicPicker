
import torch
import torch.nn as nn
import torch.nn.functional as F
from codes import configs

class _Res1D_convsizefixed_v1(nn.Module):
    def __init__(self,in_channels, out_channels, convsize):
        super().__init__()
        self.overlapTile = nn.ReflectionPad1d(int(convsize/2))  #对称填充
        # bn layers
        self.BN_1 = nn.BatchNorm1d(in_channels)
        self.BN_2 = nn.BatchNorm1d(out_channels)

        self.conv_1 = nn.Conv1d(in_channels,in_channels,convsize,1)
        self.conv_2 = nn.Conv1d(in_channels,out_channels,convsize,1)

        # 1*1 conv
        if in_channels != out_channels:
            self.bottleneck_conv = nn.Conv1d(in_channels,out_channels,1,1)
        else:
            self.bottleneck_conv = None

    
    def forward(self,x):
        X = x
        output = self.overlapTile(x)
        output = self.conv_1(output)
        output = self.BN_1(output)
        output = torch.relu(output)
        output = self.overlapTile(output)
        output = self.conv_2(output)
        output = self.BN_2(output)

        if self.bottleneck_conv:
            X = self.bottleneck_conv(x)
        
        return output + X

class ResUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_down_1 = _Res1D_convsizefixed_v1(3,6,5)
        self.res_down_2 = _Res1D_convsizefixed_v1(6,12,5)
        self.res_down_3 = _Res1D_convsizefixed_v1(12,24,5)
        self.res_retain = _Res1D_convsizefixed_v1(24,24,5)  # 准备用注意力替换
        #self.multihead_att = nn.MultiheadAttention()
        self.res_up_1 = _Res1D_convsizefixed_v1(48,12,5)
        self.res_up_2 = _Res1D_convsizefixed_v1(24,6,5)
        self.res_up_3 = _Res1D_convsizefixed_v1(12,3,5)
        self.res_output = _Res1D_convsizefixed_v1(6,3,5)
        self.upsample = torch.nn.Upsample(scale_factor=4, mode='linear')

    def forward(self,x):
        Intermediate1 = x
        output = torch.relu(self.res_down_1(x))         # 6*1600
        output = torch.max_pool1d(output,4,4)           # 6*800
        Intermediate2 = output
        output = torch.relu(self.res_down_2(output))    # 12*800
        output = torch.max_pool1d(output,4,4)           # 12*200
        Intermediate3 = output
        output = torch.relu(self.res_down_3(output))    # 24*200
        output = torch.max_pool1d(output,4,4)           # 24*50
        Intermediate4 = output
        print(output.shape)
        output = torch.relu(self.res_retain(output))    # 24*50
        output = torch.relu(self.res_up_1(torch.cat((Intermediate4,output),1)))# 12*50
        output = self.upsample(output)                  # 12*200
        output = torch.relu(self.res_up_2(torch.cat((Intermediate3,output),1)))# 6*200
        output = self.upsample(output)                  # 6*800
        output = torch.relu(self.res_up_3(torch.cat((Intermediate2,output),1)))# 3*800
        output = self.upsample(output)                  # 3*1600
        output = torch.sigmoid(self.res_output(torch.cat((Intermediate1,output),1)))# 3*1600
        return output

class _QuakePicker_v4(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_down_1 = _Res1D_convsizefixed_v1(3,6,5)
        self.res_down_2 = _Res1D_convsizefixed_v1(6,12,5)
        self.res_down_3 = _Res1D_convsizefixed_v1(12,24,5)
        # self.res_retain = _Res1D_convsizefixed_v1(24,24,5)  # 准备用注意力替换
        self.multihead_att = nn.MultiheadAttention()
        self.res_up_1 = _Res1D_convsizefixed_v1(48,12,5)
        self.res_up_2 = _Res1D_convsizefixed_v1(24,6,5)
        self.res_up_3 = _Res1D_convsizefixed_v1(12,3,5)
        self.res_output = _Res1D_convsizefixed_v1(6,3,5)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='linear')

    def forward(self,x):
        Intermediate1 = x
        output = torch.relu(self.res_down_1(x))         # 6*3000
        output = torch.max_pool1d(output,2,2)           # 6*1500
        Intermediate2 = output
        output = torch.relu(self.res_down_2(output))    # 12*1500
        output = torch.max_pool1d(output,2,2)           # 12*750
        Intermediate3 = output
        output = torch.relu(self.res_down_3(output))    # 24*750
        output = torch.max_pool1d(output,2,2)           # 24*375
        Intermediate4 = output
        output = torch.relu(self.res_retain(output))    # 24*375
        output = torch.relu(self.res_up_1(torch.cat((Intermediate4,output),1)))# 12*375
        output = self.upsample(output)                  # 12*750
        output = torch.relu(self.res_up_2(torch.cat((Intermediate3,output),1)))# 6*750
        output = self.upsample(output)                  # 6*1500
        output = torch.relu(self.res_up_3(torch.cat((Intermediate2,output),1)))# 3*1500
        output = self.upsample(output)                  # 3*3000
        output = torch.sigmoid(self.res_output(torch.cat((Intermediate1,output),1)))# 3*3000
        return output

#   避免预测直接躺平，得给损失根据标签增加权重
#   softmax,tanh 不能直接接mse损失，改用交叉熵
#   (1,10)(1,10)
def Loss_With_Weight(prob,label):
    prob = prob.reshape(1,-1)
    label = label.reshape(1,-1)
    weight = -0.2*torch.log(-0.99*label + 1) + 0.01
    loss = -(label*torch.log(prob+0.000001)+(1-label)*torch.log(1-prob+0.000001))
    return (weight * loss).sum()
