# import torch.nn as nn
# import torch


# class GLU(nn.Module):
#     def __init__(self, input_num):
#         super(GLU, self).__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.linear = nn.Linear(input_num, input_num)

#     def forward(self, x):
        
#         #batch_size, channels, height, width = x.size()
        
#         lin = self.linear(x.permute(0, 2, 3, 1))
#         #lin = self.linear(x.permute(0, 2, 3, 1)).reshape(-1, channels)
#         lin = lin.permute(0, 3, 1, 2)
#         #lin = lin.reshape(batch_size, height, width, channels).permute(0,3,1,2)
#         sig = self.sigmoid(x)
#         res = lin * sig
#         return res


# class ContextGating(nn.Module):
#     def __init__(self, input_num):
#         super(ContextGating, self).__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.linear = nn.Linear(input_num, input_num)

#     def forward(self, x):
#         lin = self.linear(x.permute(0, 2, 3, 1))
#         lin = lin.permute(0, 3, 1, 2)
#         sig = self.sigmoid(lin)
#         res = x * sig
#         return res


# class CNN(nn.Module):

#     def __init__(self, n_in_channel, activation="Relu", conv_dropout=0,
#                  kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
#                  pooling=[(1, 4), (1, 4), (1, 4)]
#                  ):
#         super(CNN, self).__init__()
#         self.nb_filters = nb_filters
#         self.cnn = nn.Sequential()

#         ########
#         # 
#         ########
#         # Layer1
#         self.cnn.add_module('conv0', nn.Conv2d(1, 16, 3,1,1))
#         self.cnn.add_module('batchnorm0', nn.BatchNorm2d(16, eps=0.001, momentum = 0.99))
#         self.cnn.add_module('glu0', GLU(16))
#         self.cnn.add_module('dropout0', nn.Dropout(conv_dropout))
#         self.cnn.add_module('avg_pooling0', nn.AvgPool2d([2,2]))
#         self.cnn.add_module('max_pooling0', nn.MaxPool2d([2,2]))
        
#         self.cnn.add_module('pconv0', nn.Conv2d(16*2, 16, 1, 1, 0)) # ?

#         # Layer2
#         self.cnn.add_module('conv1', nn.Conv2d(16, 32, 3,1,1))
#         self.cnn.add_module('batchnorm1', nn.BatchNorm2d(32, eps=0.001, momentum = 0.99))
#         self.cnn.add_module('glu1', GLU(32))
#         self.cnn.add_module('dropout1', nn.Dropout(conv_dropout))
#         self.cnn.add_module('avg_pooling1', nn.AvgPool2d([2,2]))
#         self.cnn.add_module('max_pooling1', nn.MaxPool2d([2,2]))
        
#         self.cnn.add_module('pconv1', nn.Conv2d(32*2, 32, 1, 1, 0)) # ?

#         # Layer3
#         self.cnn.add_module('conv2', nn.Conv2d(32, 64, 3,1,1))
#         self.cnn.add_module('batchnorm2', nn.BatchNorm2d(64, eps=0.001, momentum = 0.99))
#         self.cnn.add_module('glu2', GLU(64))
#         self.cnn.add_module('dropout2', nn.Dropout(conv_dropout))
#         self.cnn.add_module('pooling2', nn.AvgPool2d([1,2]))

        
#         # Layer4
#         self.cnn.add_module('conv3', nn.Conv2d(64, 128, 3,1,1))
#         self.cnn.add_module('batchnorm3', nn.BatchNorm2d(128, eps=0.001, momentum = 0.99))
#         self.cnn.add_module('glu3', GLU(128))
#         self.cnn.add_module('dropout3', nn.Dropout(conv_dropout))
#         self.cnn.add_module('pooling3', nn.AvgPool2d([1,2]))

# ################################################################################

#         # Layer5
#         self.cnn.add_module('conv4', nn.Conv2d(128, 128, 3,1,1))
#         self.cnn.add_module('batchnorm4', nn.BatchNorm2d(128, eps=0.001, momentum = 0.99))
#         self.cnn.add_module('glu4', GLU(128))
#         self.cnn.add_module('dropout4', nn.Dropout(conv_dropout))
#         self.cnn.add_module('pooling4', nn.AvgPool2d([1,2]))

#         # Layer6
#         self.cnn.add_module('conv5', nn.Conv2d(128, 128, 3,1,1))
#         self.cnn.add_module('batchnorm5', nn.BatchNorm2d(128, eps=0.001, momentum = 0.99))
#         self.cnn.add_module('glu5', GLU(128))
#         self.cnn.add_module('dropout5', nn.Dropout(conv_dropout))
#         self.cnn.add_module('pooling5', nn.AvgPool2d([1,2]))

#         # Layer7
#         self.cnn.add_module('conv6', nn.Conv2d(128, 128, 3,1,1))
#         self.cnn.add_module('batchnorm6', nn.BatchNorm2d(128, eps=0.001, momentum = 0.99))
#         self.cnn.add_module('glu6', GLU(128))
#         self.cnn.add_module('dropout6', nn.Dropout(conv_dropout))
#         self.cnn.add_module('pooling6', nn.AvgPool2d([1,2]))


#     def load_state_dict(self, state_dict, strict=True):
#         self.cnn.load_state_dict(state_dict)

#     def state_dict(self, destination=None, prefix='', keep_vars=False):
#         return self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

#     def save(self, filename):
#         torch.save(self.cnn.state_dict(), filename)

#     def forward(self, x):
#         # input size : (batch_size, n_channels, n_frames, n_freq)
#         # conv features
#         # x = self.cnn(x)            # x: (batch_size, 1, 628, 128)
#         # Layer1
#         x1=self.cnn.conv0(x)        # x1: (batch_size, 16, 628, 128)
#         x2=self.cnn.batchnorm0(x1)  # x2: (batch_size, 16, 628, 128)
#         x3=self.cnn.glu0(x2)        # x3: (batch_size, 16, 628, 128)
#         x4=self.cnn.dropout0(x3)    # x4: (batch_size, 16, 628, 128)
#         x4_avg=self.cnn.avg_pooling0(x4)    # x4_avg: (batch_size, 16, 314, 64)
#         #x4_max=self.cnn.max_pooling0(x4)    # x4_max: (batch_size, 16, 314, 64)
#         # x4_min=-self.cnn.max_pooling0(-x4)  # x4_min: (batch_size, 16, 314, 64)
#         #x5 = torch.concat((x4_avg,x4_max), dim=1)   # x5: (batch_size, 48, 314, 64)
#         #x5 = self.cnn.pconv0(x4_avg)

#         # Layer2
#         x6=self.cnn.conv1(x4_avg)   # x6: (batch_size, 96, 314, 64)
#         x7=self.cnn.batchnorm1(x6)  # x7: (batch_size, 96, 314, 64)
#         x8=self.cnn.glu1(x7)        # x8: (batch_size, 96, 314, 64)
#         x9=self.cnn.dropout1(x8)    # x9: (batch_size, 96, 314, 64)
#         x9_avg=self.cnn.avg_pooling1(x9)    # x9_avg: (batch_size, 96, 157, 32)
#         #x9_max=self.cnn.max_pooling0(x9)    # x4_max: (batch_size, 96, 157, 32)
#         # x9_min=-self.cnn.max_pooling0(-x9)  # x4_min: (batch_size, 96, 157, 32)
#         #x10 = torch.concat((x9_avg,x9_max), dim=1)   # x5: (batch_size, 288, 157, 32)
#         #x10 = self.cnn.pconv1(x9_avg)

#         # Layer3
#         x11=self.cnn.conv2(x9_avg)    # x11: (batch_size, 192, 157, 32)
#         x12=self.cnn.batchnorm2(x11)  # x12: (batch_size, 192, 157, 32)
#         x13=self.cnn.glu2(x12)        # x13: (batch_size, 192, 157, 32)
#         x14=self.cnn.dropout2(x13)    # x14: (batch_size, 192, 157, 32)
#         x14_avg=self.cnn.pooling2(x14)    # x14_avg: (batch_size, 192, 157, 16)

#         # Layer4
#         x16=self.cnn.conv3(x14_avg)   # x16: (batch_size, 384, 157, 16)
#         x17=self.cnn.batchnorm3(x16)  # x17: (batch_size, 384, 157, 16)
#         x18=self.cnn.glu3(x17)        # x18: (batch_size, 384, 157, 16)
#         x19=self.cnn.dropout3(x18)    # x19: (batch_size, 384, 157, 16)
#         x19_avg=self.cnn.pooling3(x19)    # x19_avg: (batch_size, 16, 157, 8)

# #########################################################################

#         # Layer5
#         x21=self.cnn.conv4(x19_avg)   # x21: (batch_size, 384, 157, 8)
#         x22=self.cnn.batchnorm4(x21)  # x22: (batch_size, 384, 157, 8)
#         x23=self.cnn.glu4(x22)        # x23: (batch_size, 384, 157, 8)
#         x24=self.cnn.dropout4(x23)    # x24: (batch_size, 384, 157, 8)
#         x24_avg=self.cnn.pooling4(x24)    # x24_avg: (batch_size, 384, 157, 4)

#         # Layer6
#         x26=self.cnn.conv5(x24_avg)   # x26: (batch_size, 384, 628, 4)
#         x27=self.cnn.batchnorm5(x26)  # x27: (batch_size, 384, 628, 4)
#         x28=self.cnn.glu5(x27)        # x28: (batch_size, 384, 628, 4)
#         x29=self.cnn.dropout5(x28)    # x29: (batch_size, 384, 628, 4)
#         x29_avg=self.cnn.pooling5(x29)    # x29_avg: (batch_size, 384, 157, 2)

#         # Layer7
#         x31=self.cnn.conv6(x29_avg)   # x31: (batch_size, 384, 157, 2)
#         x32=self.cnn.batchnorm6(x31)  # x32: (batch_size, 384, 157, 2)
#         x33=self.cnn.glu6(x32)        # x33: (batch_size, 384, 157, 2)
#         x34=self.cnn.dropout6(x33)    # x34: (batch_size, 384, 157, 2)
#         x34_avg=self.cnn.pooling6(x34)    # x34_avg: (batch_size, 384, 157, 1)

#         return x34_avg

import torch.nn as nn
import torch
import torch.nn.functional as F

class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        
        #batch_size, channels, height, width = x.size()
        
        lin = self.linear(x.permute(0, 2, 3, 1))
        #lin = self.linear(x.permute(0, 2, 3, 1)).reshape(-1, channels)
        lin = lin.permute(0, 3, 1, 2)
        #lin = lin.reshape(batch_size, height, width, channels).permute(0,3,1,2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        res = x * sig
        return res


class CNN(nn.Module):

    def __init__(self, n_in_channel, activation="Relu", conv_dropout=0,
                 kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)]
                 ):
# [3,3,3] : CNN Layer에서 사용하는 필터의 크기. 여기서는 3D 리스트로 지정되어있으며, 각 레이어에 대한 가로.세로.깊이 방향의 패딩 크기를 의미한다.
# [1,1,1] : 입력 이미지 주변에 추가되는 패딩의 크기를 결정. 마찬가지로 여기서는 3D 리스트로 지정되어있고, 각 레이어에 대한 가로.세로.깊이 방향의 패딩 크기를 나타냄. 각 차원에 대해 1만큼의 패딩이 추가됨을 의미
# [1,1,1] : 각 레이어에서 사용되는 커널의 이동거리를 결정. 마찬가지로 여기서는 3D 리스트로 지정되어있고, 각 레이어에 대한 가로.세로.깊이 방향의 스트라이드를 나타냄. 각 레이어에 대한 차원에 대해 스트라이드가 1임을 의미
# [64,64,64] : 각 Conv Layer에서 사용되는 필터의 개수를 결정. 여기서는 1D리스트로 지정되어있고(?), 각 레이어에서 사용되는 필터의 개수를 나타냄. 각 레이어에서 64개의 필터가 사용되는 것을 의미한다.
# [(1,4), (1,4), (1,4)] : 레이어의 유형과 크기를 결정. 여기서는 2D 튜플의 리스트로 지정되어있으며, 각 레이어에 대해 사용되는 풀링 레이어의 유형과 크기를 나타냄(풀링 유형은 일반적으로 평균, 최대값 풀링 중 하나). 각 레이어에 대해 높이 방향으로 4의 크기를 가진 평균 풀링이 적용됨을 의미한다.
        super(CNN, self).__init__()
        self.nb_filters = nb_filters
        self.cnn = nn.Sequential()

        # Layer1
        self.cnn.add_module('conv0', nn.Conv2d(1, 16, 3,1,1))
# (1,16, 3,1,1) : 
# 1 : 입력 채널의 수(이미지였다면, 흑백은 채널 수가 1이고 컬러이면 채널 수가 3)
# 16 : 출력 채널의 수(CNN 레이어에 사용되는 필터의 개수. 각 필터는 입력 이미지에서 특정 피쳐를 감지하는 기능 수행)
# 3 : 커널의 크기(커널의 높이와 너비를 나타냄. 3x3 크기의 커널을 사용하고있음)
# 1 : 스트라이드('커널이' 입력 이미지를 따라 이동하는 간격을 결정. 커널이 한번에 1픽셀씩 이동)
# 1 : 패딩(입력 이미지 주변에 추가되는 가상의 픽셀 수를 의미. 패딩을 사용하면 출력 크기를 결정할 수 있으며, 주로 입력과 출력의 크기를 동일하게 유지하기 위해 사용됨. 패딩값이 1로 설정되어있으므로 입력 이미지 주변에 한 픽셀씩 패딩이 추가된다.)
        self.cnn.add_module('batchnorm0', nn.BatchNorm2d(16, eps=0.001, momentum = 0.99))
        self.cnn.add_module('glu0', GLU(16))
        self.cnn.add_module('dropout0', nn.Dropout(conv_dropout))
        self.cnn.add_module('avg_pooling0', nn.AvgPool2d([2,2]))
        self.cnn.add_module('max_pooling0', nn.MaxPool2d([2,2]))
        
        self.cnn.add_module('pconv0', nn.Conv2d(16*2, 16, 1, 1, 0)) # ?

        # Layer2
        self.cnn.add_module('conv1', nn.Conv2d(16, 32, 3,1,1))
        self.cnn.add_module('batchnorm1', nn.BatchNorm2d(32, eps=0.001, momentum = 0.99))
        self.cnn.add_module('glu1', GLU(32))
        self.cnn.add_module('dropout1', nn.Dropout(conv_dropout))
        self.cnn.add_module('avg_pooling1', nn.AvgPool2d([2,2]))
        self.cnn.add_module('max_pooling1', nn.MaxPool2d([2,2]))
        
        self.cnn.add_module('pconv1', nn.Conv2d(32*2, 32, 1, 1, 0)) # ?

        # Layer3
        self.cnn.add_module('conv2', nn.Conv2d(32, 64, 3,1,1))
        self.cnn.add_module('batchnorm2', nn.BatchNorm2d(64, eps=0.001, momentum = 0.99))
        self.cnn.add_module('glu2', GLU(64))
        self.cnn.add_module('dropout2', nn.Dropout(conv_dropout))
        self.cnn.add_module('avg_pooling2', nn.AvgPool2d([1,2]))

        
        # Layer4
        self.cnn.add_module('conv3', nn.Conv2d(64, 128, 3,1,1))
        self.cnn.add_module('batchnorm3', nn.BatchNorm2d(128, eps=0.001, momentum = 0.99))
        self.cnn.add_module('glu3', GLU(128))
        self.cnn.add_module('dropout3', nn.Dropout(conv_dropout))
        self.cnn.add_module('avg_pooling3', nn.AvgPool2d([1,2]))

################################################################################

        # Layer5
        self.cnn.add_module('conv4', nn.Conv2d(128, 128, 3,1,1))
        self.cnn.add_module('batchnorm4', nn.BatchNorm2d(128, eps=0.001, momentum = 0.99))
        self.cnn.add_module('glu4', GLU(128))
        self.cnn.add_module('dropout4', nn.Dropout(conv_dropout))
        self.cnn.add_module('avg_pooling4', nn.AvgPool2d([1,2]))

        # Layer6
        self.cnn.add_module('conv5', nn.Conv2d(128, 128, 3,1,1))
        self.cnn.add_module('batchnorm5', nn.BatchNorm2d(128, eps=0.001, momentum = 0.99))
        self.cnn.add_module('glu5', GLU(128))
        self.cnn.add_module('dropout5', nn.Dropout(conv_dropout))
        self.cnn.add_module('avg_pooling5', nn.AvgPool2d([1,2]))

        # Layer7
        self.cnn.add_module('conv6', nn.Conv2d(128, 128, 3,1,1))
        self.cnn.add_module('batchnorm6', nn.BatchNorm2d(128, eps=0.001, momentum = 0.99))
        self.cnn.add_module('glu6', GLU(128))
        self.cnn.add_module('dropout6', nn.Dropout(conv_dropout))
        self.cnn.add_module('avg_pooling6', nn.AvgPool2d([1,2]))

######################################################

    def time_pooling(self, x, pooling_type='avg'):
        if pooling_type == 'avg':
            return F.avg_pool2d(x, [2, 1], [2, 1]) # 인자는 각각 필터 사이즈에 대한(time, freq)과 스트라이드에 대한 (time, freq)
        elif pooling_type == 'max':
            return F.max_pool2d(x, [2, 1], [2, 1])

    def freq_pooling(self, x, pooling_type='avg'):
        if pooling_type == 'avg':
            return F.avg_pool2d(torch.abs(x), [1, 2], [1, 2]) # 입력 텐서 x를 2차원 실수 푸리에 변환(rFFT2), 이 후 변환된 주파수 영역에서 2차원 평균 풀링 혹은 최대값 풀링을 수행 / FT x!!!
        elif pooling_type == 'max':
            return F.max_pool2d(torch.abs(x), [1, 2], [1, 2])
    # freq_pooling 함수 마지막에 적용하는 irfft2는 주파수 영역의 데이터를 다시 시간 영역으로 변환하는 것  / IFT x !!!  

#########################################################

    def load_state_dict(self, state_dict, strict=True):
        self.cnn.load_state_dict(state_dict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def save(self, filename):
        torch.save(self.cnn.state_dict(), filename)

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        # conv features
        # x = self.cnn(x)            # x: (batch_size, 1, 628, 128)
    
        # Layer1
        x1=self.cnn.conv0(x)       # x1: (batch_size, 16, 628, 128)
        x2=self.cnn.batchnorm0(x1) # x2: (batch_size, 16, 628, 128)
        x3=self.cnn.glu0(x2)       # x3: (batch_size, 16, 628, 128)
        x4=self.cnn.dropout0(x3)   # x4: (batch_size, 16, 628, 128)
        #x4_avg=self.cnn.avg_pooling0(x4)    # x4_avg: (batch_size, 16, 314, 64)
	    # 원큐에 하던 시간-주파수 영역 pooling연산을 시간영역과 주파수 영역에 대해 나누어서 진행한다고 생각하면 됨.
        x4_time_avg = self.time_pooling(x4, 'avg') # x4_time_avg : (batch_size, 16, 314, 128)
        x4_freq_avg = self.freq_pooling(x4_time_avg, 'avg') # x4_freq_avg : (batch_size, 16, 314, 64)
        x5 = x4_freq_avg
	    #x4_combined = torch.cat((x4_time, x4_freq), dim=1)
        #x4_max=self.cnn.max_pooling0(x4)    # x4_max: (batch_size, 16, 314, 64)
        # x4_min=-self.cnn.max_pooling0(-x4)  # x4_min: (batch_size, 16, 314, 64)
        #x5 = torch.concat((x4_avg,x4_max), dim=1)   # x5: (batch_size, 48, 314, 64)
        #x5 = self.cnn.pconv0(x4_avg)

        # Layer2
        #x6=self.cnn.conv1(x4_avg)   # x6: (batch_size, 96, 314, 64)
        x6 = self.cnn.conv1(x5)
        x7=self.cnn.batchnorm1(x6)  # x7: (batch_size, 96, 314, 64)
        x8=self.cnn.glu1(x7)        # x8: (batch_size, 96, 314, 64)
        x9=self.cnn.dropout1(x8)    # x9: (batch_size, 96, 314, 64)
        #x9_avg=self.cnn.avg_pooling1(x9)    # x9_avg: (batch_size, 96, 157, 32)
        x9_time_avg = self.time_pooling(x9, 'avg') # x9_time_avg : (batch_size, 96, 157, 64)
        x9_freq_avg = self.freq_pooling(x9_time_avg, 'avg') # x9_freq_avg : (batch_size, 96, 157, 32)
        x10 = x9_freq_avg
        
        #x9_max=self.cnn.max_pooling0(x9)    # x4_max: (batch_size, 96, 157, 32)
        # x9_min=-self.cnn.max_pooling0(-x9)  # x4_min: (batch_size, 96, 157, 32)
        #x10 = torch.concat((x9_avg,x9_max), dim=1)   # x5: (batch_size, 288, 157, 32)
        #x10 = self.cnn.pconv1(x9_avg)

        # Layer3
        x11=self.cnn.conv2(x10)    # x11: (batch_size, 192, 157, 32)
        x12=self.cnn.batchnorm2(x11)  # x12: (batch_size, 192, 157, 32)
        x13=self.cnn.glu2(x12)        # x13: (batch_size, 192, 157, 32)
        x14=self.cnn.dropout2(x13)    # x14: (batch_size, 192, 157, 32)
        x14_avg=self.cnn.avg_pooling2(x14)    # x14_avg: (batch_size, 192, 157, 16)

 	    # Layer4
        x16=self.cnn.conv3(x14_avg)   # x16: (batch_size, 384, 157, 16)
        x17=self.cnn.batchnorm3(x16)  # x17: (batch_size, 384, 157, 16)
        x18=self.cnn.glu3(x17)        # x18: (batch_size, 384, 157, 16)
        x19=self.cnn.dropout3(x18)    # x19: (batch_size, 384, 157, 16)
        x19_avg=self.cnn.avg_pooling3(x19)    # x19_avg: (batch_size, 16, 157, 8)

#######################################################################

 	    # Layer5
        x21=self.cnn.conv4(x19_avg)   # x21: (batch_size, 384, 157, 8)
        x22=self.cnn.batchnorm4(x21)  # x22: (batch_size, 384, 157, 8)
        x23=self.cnn.glu4(x22)        # x23: (batch_size, 384, 157, 8)
        x24=self.cnn.dropout4(x23)    # x24: (batch_size, 384, 157, 8)
        x24_avg=self.cnn.avg_pooling4(x24)    # x24_avg: (batch_size, 384, 157, 4)

	    # Layer6
        x26=self.cnn.conv5(x24_avg)   # x26: (batch_size, 384, 628, 4)
        x27=self.cnn.batchnorm5(x26)  # x27: (batch_size, 384, 628, 4)
        x28=self.cnn.glu5(x27)        # x28: (batch_size, 384, 628, 4)
        x29=self.cnn.dropout5(x28)    # x29: (batch_size, 384, 628, 4)
        x29_avg=self.cnn.avg_pooling5(x29)    # x29_avg: (batch_size, 384, 157, 2)

        # Layer7
        x31=self.cnn.conv6(x29_avg)   # x31: (batch_size, 384, 157, 2)
        x32=self.cnn.batchnorm6(x31)  # x32: (batch_size, 384, 157, 2)
        x33=self.cnn.glu6(x32)        # x33: (batch_size, 384, 157, 2)
        x34=self.cnn.dropout6(x33)    # x34: (batch_size, 384, 157, 2)
        x34_avg=self.cnn.avg_pooling6(x34)    # x34_avg: (batch_size, 384, 157, 1)

        return x34_avg
