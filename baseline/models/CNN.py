import torch.nn as nn
import torch


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
        super(CNN, self).__init__()
        self.nb_filters = nb_filters
        self.cnn = nn.Sequential()

        ########
        # 
        ########
        # Layer1
        self.cnn.add_module('conv0', nn.Conv2d(1, 16, 3,1,1))
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
        self.cnn.add_module('pooling2', nn.AvgPool2d([1,2]))

        
        # Layer4
        self.cnn.add_module('conv3', nn.Conv2d(64, 128, 3,1,1))
        self.cnn.add_module('batchnorm3', nn.BatchNorm2d(128, eps=0.001, momentum = 0.99))
        self.cnn.add_module('glu3', GLU(128))
        self.cnn.add_module('dropout3', nn.Dropout(conv_dropout))
        self.cnn.add_module('pooling3', nn.AvgPool2d([1,2]))

################################################################################

        # Layer5
        self.cnn.add_module('conv4', nn.Conv2d(128, 128, 3,1,1))
        self.cnn.add_module('batchnorm4', nn.BatchNorm2d(128, eps=0.001, momentum = 0.99))
        self.cnn.add_module('glu4', GLU(128))
        self.cnn.add_module('dropout4', nn.Dropout(conv_dropout))
        self.cnn.add_module('pooling4', nn.AvgPool2d([1,2]))

        # Layer6
        self.cnn.add_module('conv5', nn.Conv2d(128, 128, 3,1,1))
        self.cnn.add_module('batchnorm5', nn.BatchNorm2d(128, eps=0.001, momentum = 0.99))
        self.cnn.add_module('glu5', GLU(128))
        self.cnn.add_module('dropout5', nn.Dropout(conv_dropout))
        self.cnn.add_module('pooling5', nn.AvgPool2d([1,2]))

        # Layer7
        self.cnn.add_module('conv6', nn.Conv2d(128, 128, 3,1,1))
        self.cnn.add_module('batchnorm6', nn.BatchNorm2d(128, eps=0.001, momentum = 0.99))
        self.cnn.add_module('glu6', GLU(128))
        self.cnn.add_module('dropout6', nn.Dropout(conv_dropout))
        self.cnn.add_module('pooling6', nn.AvgPool2d([1,2]))



        # def conv(i, batchNormalization=False, dropout=None, activ="relu"):
        #     nIn = n_in_channel if i == 0 else nb_filters[i - 1]
        #     nOut = nb_filters[i]
        #     cnn.add_module('conv{0}'.format(i),
        #                    nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]))
        #     if batchNormalization:
        #         cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99))
        #     if activ.lower() == "leakyrelu":
        #         cnn.add_module('relu{0}'.format(i),
        #                        nn.LeakyReLU(0.2))
        #     elif activ.lower() == "relu":
        #         cnn.add_module('relu{0}'.format(i), nn.ReLU())
        #     elif activ.lower() == "glu":
        #         cnn.add_module('glu{0}'.format(i), GLU(nOut))
        #     elif activ.lower() == "cg":
        #         cnn.add_module('cg{0}'.format(i), ContextGating(nOut))
        #     if dropout is not None:
        #         cnn.add_module('dropout{0}'.format(i),
        #                        nn.Dropout(dropout))

        # batch_norm = True
        # # 128x862x64
        # for i in range(len(nb_filters)):
        #     conv(i, batch_norm, conv_dropout, activ=activation)
        #     cnn.add_module('pooling{0}'.format(i), nn.AvgPool2d(pooling[i]))  # bs x tframe x mels

        # self.cnn = cnn

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
        x1=self.cnn.conv0(x)        # x1: (batch_size, 16, 628, 128)
        x2=self.cnn.batchnorm0(x1)  # x2: (batch_size, 16, 628, 128)
        x3=self.cnn.glu0(x2)        # x3: (batch_size, 16, 628, 128)
        x4=self.cnn.dropout0(x3)    # x4: (batch_size, 16, 628, 128)
        #x4_avg=self.cnn.avg_pooling0(x4)    # x4_avg: (batch_size, 16, 314, 64)
        x4_max=self.cnn.max_pooling0(x4)    # x4_max: (batch_size, 16, 314, 64)
        # x4_min=-self.cnn.max_pooling0(-x4)  # x4_min: (batch_size, 16, 314, 64)
        #x5 = torch.concat((x4_avg,x4_max), dim=1)   # x5: (batch_size, 48, 314, 64)
        #x5 = self.cnn.pconv0(x4_avg)

        # Layer2
        x6=self.cnn.conv1(x4_max)   # x6: (batch_size, 96, 314, 64)
        x7=self.cnn.batchnorm1(x6)  # x7: (batch_size, 96, 314, 64)
        x8=self.cnn.glu1(x7)        # x8: (batch_size, 96, 314, 64)
        x9=self.cnn.dropout1(x8)    # x9: (batch_size, 96, 314, 64)
        x9_avg=self.cnn.avg_pooling1(x9)    # x9_avg: (batch_size, 96, 157, 32)
        #x9_max=self.cnn.max_pooling0(x9)    # x4_max: (batch_size, 96, 157, 32)
        # x9_min=-self.cnn.max_pooling0(-x9)  # x4_min: (batch_size, 96, 157, 32)
        #x10 = torch.concat((x9_avg,x9_max), dim=1)   # x5: (batch_size, 288, 157, 32)
        #x10 = self.cnn.pconv1(x9_avg)

        # Layer3
        x11=self.cnn.conv2(x9_avg)    # x11: (batch_size, 192, 157, 32)
        x12=self.cnn.batchnorm2(x11)  # x12: (batch_size, 192, 157, 32)
        x13=self.cnn.glu2(x12)        # x13: (batch_size, 192, 157, 32)
        x14=self.cnn.dropout2(x13)    # x14: (batch_size, 192, 157, 32)
        x14_avg=self.cnn.pooling2(x14)    # x14_avg: (batch_size, 192, 157, 16)

        # Layer4
        x16=self.cnn.conv3(x14_avg)   # x16: (batch_size, 384, 157, 16)
        x17=self.cnn.batchnorm3(x16)  # x17: (batch_size, 384, 157, 16)
        x18=self.cnn.glu3(x17)        # x18: (batch_size, 384, 157, 16)
        x19=self.cnn.dropout3(x18)    # x19: (batch_size, 384, 157, 16)
        x19_avg=self.cnn.pooling3(x19)    # x19_avg: (batch_size, 16, 157, 8)

        # Layer5
        x21=self.cnn.conv4(x19_avg)   # x21: (batch_size, 384, 157, 8)
        x22=self.cnn.batchnorm4(x21)  # x22: (batch_size, 384, 157, 8)
        x23=self.cnn.glu4(x22)        # x23: (batch_size, 384, 157, 8)
        x24=self.cnn.dropout4(x23)    # x24: (batch_size, 384, 157, 8)
        x24_avg=self.cnn.pooling4(x24)    # x24_avg: (batch_size, 384, 157, 4)

        # Layer6
        x26=self.cnn.conv5(x24_avg)   # x26: (batch_size, 384, 628, 4)
        x27=self.cnn.batchnorm5(x26)  # x27: (batch_size, 384, 628, 4)
        x28=self.cnn.glu5(x27)        # x28: (batch_size, 384, 628, 4)
        x29=self.cnn.dropout5(x28)    # x29: (batch_size, 384, 628, 4)
        x29_avg=self.cnn.pooling5(x29)    # x29_avg: (batch_size, 384, 157, 2)

        # Layer7
        x31=self.cnn.conv6(x29_avg)   # x31: (batch_size, 384, 157, 2)
        x32=self.cnn.batchnorm6(x31)  # x32: (batch_size, 384, 157, 2)
        x33=self.cnn.glu6(x32)        # x33: (batch_size, 384, 157, 2)
        x34=self.cnn.dropout6(x33)    # x34: (batch_size, 384, 157, 2)
        x34_avg=self.cnn.pooling6(x34)    # x34_avg: (batch_size, 384, 157, 1)

        ''''''''''''
        # x=self.cnn.conv0(x)        # x1: (batch_size, 16, 628, 128)
        # x=self.cnn.batchnorm0(x)  # x2: (batch_size, 16, 628, 128)
        # x=self.cnn.glu0(x)        # x3: (batch_size, 16, 628, 128)
        # x=self.cnn.dropout0(x)    # x4: (batch_size, 16, 628, 128)
        # x=self.cnn.avg_pooling0(x)    # x4_avg: (batch_size, 16, 314, 64)
        # # x4_max=self.cnn.max_pooling0(x4)    # x4_max: (batch_size, 16, 314, 64)
        # # x4_min=-self.cnn.max_pooling0(-x4)  # x4_min: (batch_size, 16, 314, 64)
        # # x5 = torch.concat(x4_avg,x4_max,x4_min, dim=1)   # x5: (batch_size, 48, 314, 64)

        # x=self.cnn.conv1(x)   # x6: (batch_size, 32, 314, 64)
        # x=self.cnn.batchnorm1(x)  # x7: (batch_size, 32, 314, 64)
        # x=self.cnn.glu1(x)        # x8: (batch_size, 32, 314, 64)
        # x=self.cnn.dropout1(x)    # x9: (batch_size, 32, 314, 64)
        # x=self.cnn.avg_pooling1(x)    # x9_avg: (batch_size, 32, 157, 32)
        # # x9_max=self.cnn.max_pooling0(x9)    # x4_max: (batch_size, 32, 157, 32)
        # # x9_min=-self.cnn.max_pooling0(-x9)  # x4_min: (batch_size, 32, 157, 32)
        # # x10 = torch.concat(x9_avg,x9_max,x9_min, dim=1)   # x5: (batch_size, 96, 157, 32)

        # x=self.cnn.conv2(x)    # x11: (batch_size, 64, 157, 16)
        # x=self.cnn.batchnorm2(x)  # x12: (batch_size, 64, 157, 16)
        # x=self.cnn.glu2(x)        # x13: (batch_size, 64, 157, 16)
        # x=self.cnn.dropout2(x)    # x14: (batch_size, 64, 157, 16)
        # x=self.cnn.avg_pooling2(x)    # x14_avg: (batch_size, 64, 157, 8)

        # x=self.cnn.conv3(x)   # x16: (batch_size, 128, 157, 8)
        # x=self.cnn.batchnorm3(x)  # x17: (batch_size, 128, 157, 8)
        # x=self.cnn.glu3(x)        # x18: (batch_size, 128, 157, 8)
        # x=self.cnn.dropout3(x)    # x19: (batch_size, 128, 157, 8)
        # x=self.cnn.avg_pooling3(x)    # x19_avg: (batch_size, 16, 157, 4)

        # x=self.cnn.conv4(x)   # x21: (batch_size, 128, 157, 4)
        # x=self.cnn.batchnorm4(x)  # x22: (batch_size, 128, 157, 4)
        # x=self.cnn.glu4(x)        # x23: (batch_size, 128, 157, 4)
        # x=self.cnn.dropout4(x)    # x24: (batch_size, 128, 157, 4)
        # x=self.cnn.avg_pooling4(x)    # x24_avg: (batch_size, 16, 157, 2)

        # x=self.cnn.conv5(x)   # x26: (batch_size, 128, 628, 2)
        # x=self.cnn.batchnorm5(x)  # x27: (batch_size, 128, 628, 2)
        # x=self.cnn.glu5(x)        # x28: (batch_size, 128, 628, 2)
        # x=self.cnn.dropout5(x)    # x29: (batch_size, 128, 628, 2)
        # x=self.cnn.avg_pooling5(x)    # x29_avg: (batch_size, 16, 157, 1)

        # x=self.cnn.conv6(x)   # x31: (batch_size, 128, 628, 128)
        # x=self.cnn.batchnorm6(x)  # x32: (batch_size, 128, 628, 128)
        # x=self.cnn.glu6(x)        # x33: (batch_size, 128, 628, 128)
        # x=self.cnn.dropout6(x)    # x34: (batch_size, 128, 628, 128)
        # x=self.cnn.avg_pooling6(x)    # x34_avg: (batch_size, 16, 157, 64)

        return x34_avg


