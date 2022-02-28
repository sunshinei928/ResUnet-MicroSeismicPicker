
from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
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


class ResUnet_LSTM(nn.Module):
    def __init__(self,train_mode):
        super().__init__()
        self.train_mode = train_mode
        self.res_down_1 = _Res1D_convsizefixed_v1(3,6,5)
        self.res_down_2 = _Res1D_convsizefixed_v1(6,12,5)
        self.res_down_3 = _Res1D_convsizefixed_v1(12,24,5)
        # self.res_retain = _Res1D_convsizefixed_v1(24,24,5)  # 准备用注意力替换
        if self.train_mode == True:
            self.h_0 = torch.zeros(2*1, configs.BATCH_SIZE, 12).to(configs.device)#direct*layer,batch,hidden_size
            self.c_0 = torch.zeros(2*1, configs.BATCH_SIZE, 12).to(configs.device)
        else:
            self.h_0 = torch.zeros(1*2, 1, 12).to(configs.device)#direct*layer,batch,hidden_size
            self.c_0 = torch.zeros(1*2, 1, 12).to(configs.device)
        self.bi_lstm = LSTM(input_size=24, hidden_size=12, num_layers=1, batch_first=False, bidirectional = True)
        self.leakyRelu = nn.LeakyReLU(0.1)
        self.res_up_1p = _Res1D_convsizefixed_v1(48,12,5)
        self.res_up_1s = _Res1D_convsizefixed_v1(48,12,5)
        self.res_up_2p = _Res1D_convsizefixed_v1(24,6,5)
        self.res_up_2s = _Res1D_convsizefixed_v1(24,6,5)
        self.res_up_3p = _Res1D_convsizefixed_v1(12,3,5)
        self.res_up_3s = _Res1D_convsizefixed_v1(12,3,5)
        self.res_dense_p = _Res1D_convsizefixed_v1(6,3,5)
        self.res_dense_s = _Res1D_convsizefixed_v1(6,3,5)
        self.res_output_p = _Res1D_convsizefixed_v1(3,1,1)
        self.res_output_s = _Res1D_convsizefixed_v1(3,1,1)
        self.upsample4 = torch.nn.Upsample(scale_factor=4, mode='linear')
        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode='linear')

    def encode(self,x):
        self.Intermediate1 = x
        output = torch.relu(self.res_down_1(x))         # 6*1600
        output = torch.max_pool1d(output,2,2)           # 6*800
        self.Intermediate2 = output
        output = torch.relu(self.res_down_2(output))    # 12*800
        output = torch.max_pool1d(output,4,4)           # 12*200
        self.Intermediate3 = output
        output = torch.relu(self.res_down_3(output))    # 24*200
        output = torch.max_pool1d(output,4,4)           # 24*50
        self.Intermediate4 = output
        return output

    def decode_p(self,x):
        output = self.leakyRelu(self.res_up_1p(torch.cat((self.Intermediate4,x),1)))
        output = self.upsample4(output)
        output = self.leakyRelu(self.res_up_2p(torch.cat((self.Intermediate3,output),1)))
        output = self.upsample4(output)
        output = self.leakyRelu(self.res_up_3p(torch.cat((self.Intermediate2,output),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_dense_p(torch.cat((self.Intermediate1,output),1)))
        output = torch.sigmoid(self.res_output_p(output))
        return output

    def decode_s(self,x):
        # 重置h c
        if self.train_mode == True:
            self.h_0 = torch.zeros(2*1, configs.BATCH_SIZE, 12).to(configs.device)#direct*layer,batch,hidden_size
            self.c_0 = torch.zeros(2*1, configs.BATCH_SIZE, 12).to(configs.device)
        else:
            self.h_0 = torch.zeros(1*2, 1, 12).to(configs.device)#direct*layer,batch,hidden_size
            self.c_0 = torch.zeros(1*2, 1, 12).to(configs.device)
        output = x.permute(2,0,1) # b,24,50=>50,b,24
        output,_ = self.bi_lstm(output,(self.h_0,self.c_0))# 24*50
        output = output.permute(1,2,0)
        output = self.leakyRelu(self.res_up_1s(torch.cat((self.Intermediate4,x),1)))
        output = self.upsample4(output)
        output = self.leakyRelu(self.res_up_2s(torch.cat((self.Intermediate3,output),1)))
        output = self.upsample4(output)
        output = self.leakyRelu(self.res_up_3s(torch.cat((self.Intermediate2,output),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_dense_s(torch.cat((self.Intermediate1,output),1)))
        output = torch.sigmoid(self.res_output_s(output))
        return output

    def forward(self,x):
        vec = self.encode(x)
        output_p = self.decode_p(vec)
        output_s = self.decode_s(vec)
        output = torch.cat((output_p,output_s),1)
        return output


class ResUnet_LSTM_L(nn.Module):
    def __init__(self,train_mode):
        super().__init__()
        self.train_mode = train_mode
        self.res_down_1 = _Res1D_convsizefixed_v1(3,6,5)
        self.res_down_2 = _Res1D_convsizefixed_v1(6,9,5)
        self.res_down_3 = _Res1D_convsizefixed_v1(9,12,5)
        self.res_down_4 = _Res1D_convsizefixed_v1(12,18,3)
        self.res_down_5 = _Res1D_convsizefixed_v1(18,24,3)
        if self.train_mode == True:
            self.h_0 = torch.zeros(2*1, configs.BATCH_SIZE, 12).to(configs.device)#direct*layer,batch,hidden_size
            self.c_0 = torch.zeros(2*1, configs.BATCH_SIZE, 12).to(configs.device)
        else:
            self.h_0 = torch.zeros(1*2, 1, 12).to(configs.device)#direct*layer,batch,hidden_size
            self.c_0 = torch.zeros(1*2, 1, 12).to(configs.device)
        self.bi_lstm = LSTM(input_size=24, hidden_size=12, num_layers=1, batch_first=False, bidirectional = True)
        self.leakyRelu = nn.LeakyReLU(0.1)
        self.res_up_1p = _Res1D_convsizefixed_v1(48,18,3)
        self.res_up_1s = _Res1D_convsizefixed_v1(48,18,3)

        self.res_up_2p = _Res1D_convsizefixed_v1(36,12,3)
        self.res_up_2s = _Res1D_convsizefixed_v1(36,12,3)

        self.res_up_3p = _Res1D_convsizefixed_v1(24,9,5)
        self.res_up_3s = _Res1D_convsizefixed_v1(24,9,5)

        self.res_up_4p = _Res1D_convsizefixed_v1(18,6,5)
        self.res_up_4s = _Res1D_convsizefixed_v1(18,6,5)

        self.res_up_5p = _Res1D_convsizefixed_v1(12,3,5)
        self.res_up_5s = _Res1D_convsizefixed_v1(12,3,5)        

        self.res_dense_p = _Res1D_convsizefixed_v1(6,3,3)
        self.res_dense_s = _Res1D_convsizefixed_v1(6,3,3)
        self.res_output_p = _Res1D_convsizefixed_v1(3,1,1)
        self.res_output_s = _Res1D_convsizefixed_v1(3,1,1)

        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode='linear')

    def encode(self,x):
        self.Intermediate1 = x
        output = self.leakyRelu(self.res_down_1(x))         # 6*1600
        output = torch.max_pool1d(output,2,2)               # 6*800
        self.Intermediate2 = output
        output = self.leakyRelu(self.res_down_2(output))    # 9*800
        output = torch.max_pool1d(output,2,2)               # 9*400
        self.Intermediate3 = output
        output = self.leakyRelu(self.res_down_3(output))    # 12*400
        output = torch.max_pool1d(output,2,2)               # 12*200
        self.Intermediate4 = output
        output = self.leakyRelu(self.res_down_4(output))    # 18*200
        output = torch.max_pool1d(output,2,2)               # 18*100
        self.Intermediate5 = output
        output = self.leakyRelu(self.res_down_5(output))    # 24*100
        output = torch.max_pool1d(output,2,2)               # 24*50
        self.Intermediate6 = output
        return output

    def decode_p(self,x):
        output = self.leakyRelu(self.res_up_1p(torch.cat((self.Intermediate6,x),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_up_2p(torch.cat((self.Intermediate5,output),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_up_3p(torch.cat((self.Intermediate4,output),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_up_4p(torch.cat((self.Intermediate3,output),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_up_5p(torch.cat((self.Intermediate2,output),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_dense_p(torch.cat((self.Intermediate1,output),1)))              
        output = torch.sigmoid(self.res_output_p(output))
        return output

    def decode_s(self,x):
        # 重置h c
        if self.train_mode == True:
            self.h_0 = torch.zeros(2*1, configs.BATCH_SIZE, 12).to(configs.device)#direct*layer,batch,hidden_size
            self.c_0 = torch.zeros(2*1, configs.BATCH_SIZE, 12).to(configs.device)
        else:
            self.h_0 = torch.zeros(1*2, 1, 12).to(configs.device)#direct*layer,batch,hidden_size
            self.c_0 = torch.zeros(1*2, 1, 12).to(configs.device)
        output = x.permute(2,0,1) # b,24,50=>50,b,24
        output,_ = self.bi_lstm(output,(self.h_0,self.c_0))# 24*50
        output = output.permute(1,2,0)

        output = self.leakyRelu(self.res_up_1s(torch.cat((self.Intermediate6,x),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_up_2s(torch.cat((self.Intermediate5,output),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_up_3s(torch.cat((self.Intermediate4,output),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_up_4s(torch.cat((self.Intermediate3,output),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_up_5s(torch.cat((self.Intermediate2,output),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_dense_s(torch.cat((self.Intermediate1,output),1)))              
        output = torch.sigmoid(self.res_output_s(output))
        return output

    def forward(self,x):
        vec = self.encode(x)
        output_p = self.decode_p(vec)
        output_s = self.decode_s(vec)
        output = torch.cat((output_p,output_s),1)
        return output


class ResUnet_LSTM(nn.Module):
    def __init__(self,train_mode=True):
        super().__init__()
        self.res_down_1 = _Res1D_convsizefixed_v1(3,6,5)
        self.res_down_2 = _Res1D_convsizefixed_v1(6,9,5)
        self.res_down_3 = _Res1D_convsizefixed_v1(9,12,5)
        self.res_down_4 = _Res1D_convsizefixed_v1(12,18,3)
        self.res_down_5 = _Res1D_convsizefixed_v1(18,24,3)
        self.leakyRelu = nn.LeakyReLU(0.1)
        self.res_up_1p = _Res1D_convsizefixed_v1(48,18,3)
        self.res_up_1s = _Res1D_convsizefixed_v1(48,18,3)

        self.res_up_2p = _Res1D_convsizefixed_v1(36,12,3)
        self.res_up_2s = _Res1D_convsizefixed_v1(36,12,3)

        self.res_up_3p = _Res1D_convsizefixed_v1(24,9,5)
        self.res_up_3s = _Res1D_convsizefixed_v1(24,9,5)

        self.res_up_4p = _Res1D_convsizefixed_v1(18,6,5)
        self.res_up_4s = _Res1D_convsizefixed_v1(18,6,5)

        self.res_up_5p = _Res1D_convsizefixed_v1(12,3,5)
        self.res_up_5s = _Res1D_convsizefixed_v1(12,3,5)        

        self.res_dense_p = _Res1D_convsizefixed_v1(6,3,3)
        self.res_dense_s = _Res1D_convsizefixed_v1(6,3,3)
        self.res_output_p = _Res1D_convsizefixed_v1(3,1,1)
        self.res_output_s = _Res1D_convsizefixed_v1(3,1,1)

        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode='linear')

    def encode(self,x):
        self.Intermediate1 = x
        output = self.leakyRelu(self.res_down_1(x))         # 6*1600
        output = torch.max_pool1d(output,2,2)               # 6*800
        self.Intermediate2 = output
        output = self.leakyRelu(self.res_down_2(output))    # 9*800
        output = torch.max_pool1d(output,2,2)               # 9*400
        self.Intermediate3 = output
        output = self.leakyRelu(self.res_down_3(output))    # 12*400
        output = torch.max_pool1d(output,2,2)               # 12*200
        self.Intermediate4 = output
        output = self.leakyRelu(self.res_down_4(output))    # 18*200
        output = torch.max_pool1d(output,2,2)               # 18*100
        self.Intermediate5 = output
        output = self.leakyRelu(self.res_down_5(output))    # 24*100
        output = torch.max_pool1d(output,2,2)               # 24*50
        self.Intermediate6 = output
        return output

    def decode_p(self,x):
        output = self.leakyRelu(self.res_up_1p(torch.cat((self.Intermediate6,x),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_up_2p(torch.cat((self.Intermediate5,output),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_up_3p(torch.cat((self.Intermediate4,output),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_up_4p(torch.cat((self.Intermediate3,output),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_up_5p(torch.cat((self.Intermediate2,output),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_dense_p(torch.cat((self.Intermediate1,output),1)))              
        output = torch.sigmoid(self.res_output_p(output))
        return output

    def decode_s(self,x):
        output = self.leakyRelu(self.res_up_1s(torch.cat((self.Intermediate6,x),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_up_2s(torch.cat((self.Intermediate5,output),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_up_3s(torch.cat((self.Intermediate4,output),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_up_4s(torch.cat((self.Intermediate3,output),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_up_5s(torch.cat((self.Intermediate2,output),1)))
        output = self.upsample2(output)
        output = self.leakyRelu(self.res_dense_s(torch.cat((self.Intermediate1,output),1)))              
        output = torch.sigmoid(self.res_output_s(output))
        return output

    def forward(self,x):
        vec = self.encode(x)
        output_p = self.decode_p(vec)
        output_s = self.decode_s(vec)
        output = torch.cat((output_p,output_s),1)
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
