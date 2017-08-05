import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ResBlock(nn.Module):
    def __init__(self, n, size, isInstanceNorm=True):
        super(ResBlock, self).__init__()
        if isInstanceNorm == True:
            normLayer = nn.InstanceNorm2d
        else:
            normLayer = nn.BatchNorm2d
        self.norm1   = normLayer(n)
        self.norm2   = normLayer(n)
        self.conv1 = nn.Conv2d(n, n, size, padding=1, bias=False)
        self.conv2 = nn.Conv2d(n, n, size, padding=1, bias=True)

    def forward(self, x):
        y = x
        y = F.relu(self.norm1(self.conv1(y)), True)
        y = self.norm2(self.conv2(y))
        return x+y



class Generator(nn.Module):
    def __init__(self, isInstanceNorm=True):
        super(Generator, self).__init__()

        if isInstanceNorm:
            normLayer = nn.InstanceNorm2d
        else:
            normLayer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                kernel_size=7, stride=1, padding=3, bias=False)
        self.norm1 = normLayer(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = normLayer(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                kernel_size=3, stride=2, padding=1, bias=False)
        self.norm3 = normLayer(128)

        self.res1 = ResBlock(128, 3, isInstanceNorm)
        self.res2 = ResBlock(128, 3, isInstanceNorm)
        self.res3 = ResBlock(128, 3, isInstanceNorm)
        self.res4 = ResBlock(128, 3, isInstanceNorm)
        self.res5 = ResBlock(128, 3, isInstanceNorm)
        self.res6 = ResBlock(128, 3, isInstanceNorm)
        self.res7 = ResBlock(128, 3, isInstanceNorm)
        self.res8 = ResBlock(128, 3, isInstanceNorm)
        self.res9 = ResBlock(128, 3, isInstanceNorm)

        self.dconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.dnorm1 = normLayer(64)
        self.dconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.dnorm2 = normLayer(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=3,
                kernel_size=7, stride=1, padding=3, bias=True)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)), True)
        x = F.relu(self.norm2(self.conv2(x)), True)
        x = F.relu(self.norm3(self.conv3(x)), True)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.res9(x)

        x = F.relu(self.dnorm1(self.dconv1(x)), True)
        x = F.relu(self.dnorm2(self.dconv2(x)), True)
        x = F.tanh(self.conv4(x))
        return x


class Discriminator70(nn.Module):
    def __init__(self):
        super(Discriminator70, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.lre1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.lre2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.lre3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.lre4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv5 = nn.Conv2d(512, 1, 1, 1, 0, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.lre1(self.conv1(x))
        x = self.lre2(self.bn2(self.conv2(x)))
        x = self.lre3(self.bn3(self.conv3(x)))
        x = self.lre4(self.bn4(self.conv4(x)))
        x = self.sig(self.conv5(x))
        return x


class vggBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(vggBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU(inplace = True)
    def forward(self, x):
        x = self.relu(self.conv(x))
        return x

class vgg19_54(nn.Module):
    def __init__(self):
        super(vgg19_54, self).__init__()
        # The first convolutional layer
        self.vb1_1 = vggBlock(3, 64)
        self.vb1_2 = vggBlock(64, 64)
        # The second convolutional layer
        self.vb2_1 = vggBlock(64, 128)
        self.vb2_2 = vggBlock(128, 128)
        # The thrid convolutional layer
        self.vb3_1 = vggBlock(128, 256)
        self.vb3_2 = vggBlock(256, 256)
        self.vb3_3 = vggBlock(256, 256)
        self.vb3_4 = vggBlock(256, 256)
        # The fourth convolutional layer
        self.vb4_1 = vggBlock(256, 512)
        self.vb4_2 = vggBlock(512, 512)
        self.vb4_3 = vggBlock(512, 512)
        self.vb4_4 = vggBlock(512, 512)
        # The fifth convolutional layer
        self.vb5_1 = vggBlock(512, 512)
        self.vb5_2 = vggBlock(512, 512)
        self.vb5_3 = vggBlock(512, 512)
        self.vb5_4 = vggBlock(512, 512)

    def forward(self, x):
        x = self.vb1_2(self.vb1_1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.vb2_2(self.vb2_1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.vb3_4(self.vb3_3(self.vb3_2(self.vb3_1(x) ) ) )
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.vb4_4(self.vb4_3(self.vb4_2(self.vb4_1(x) ) ) )
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.vb5_4(self.vb5_3(self.vb5_2(self.vb5_1(x) ) ) )
        return x


def weights_loader(network, weights):
    witer = iter(weights)
    for param in network.parameters():
        paramName = witer.next()
        param.data = weights[paramName]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

