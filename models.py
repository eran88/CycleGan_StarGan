import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
padding_mode='reflect'

class Cyclegan_Discriminator(nn.Module):
    def __init__(self, output_nc=1, scale=64):
        super(Cyclegan_Discriminator, self).__init__()
        self.output_nc = output_nc
        self.lrelu=nn.LeakyReLU(0.2, inplace=True)
        self.norm2=nn.InstanceNorm2d(scale*2)
        self.norm3=nn.InstanceNorm2d(scale*4)
        self.norm4=nn.InstanceNorm2d(scale*8)
        self.conv1=nn.Conv2d(in_channels =3, out_channels=scale, kernel_size =4,stride=2,padding=1)         
        self.conv2=nn.Conv2d(in_channels =scale, out_channels=scale*2, kernel_size =4,stride=2,padding=1)         
        self.conv3=nn.Conv2d(in_channels =scale*2, out_channels=scale*4, kernel_size =4,stride=2,padding=1)         
        self.conv4=nn.Conv2d(in_channels =scale*4, out_channels=scale*8, kernel_size =4,stride=1,padding=1)         
        self.conv5=nn.Conv2d(in_channels =scale*8, out_channels=self.output_nc, kernel_size =4,stride=1,padding=1)

    def forward(self, x):
        x=self.lrelu(self.conv1(x))
        x=self.lrelu(self.norm2(self.conv2(x)))
        x=self.lrelu(self.norm3(self.conv3(x)))
        x=self.lrelu(self.norm4(self.conv4(x)))
        x=self.conv5(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        return x


class Cyclegan_Generator(nn.Module):
    def __init__(self, input_nc=3, scale=64,resnet_layers=9):
        super(Cyclegan_Generator, self).__init__()
        self.relu=nn.ReLU(inplace=True)
        self.norm1=nn.InstanceNorm2d(scale)
        self.norm2=nn.InstanceNorm2d(scale*2)
        self.norm3=nn.InstanceNorm2d(scale*4)
        self.norm4=nn.InstanceNorm2d(scale*2)
        self.norm5=nn.InstanceNorm2d(scale)
        self.conv1=nn.Conv2d(in_channels =input_nc, out_channels=scale, kernel_size =7,padding=3,stride=1,padding_mode="reflect")
        self.down1=nn.Conv2d(in_channels =scale, out_channels=scale*2, kernel_size =3, stride=2, padding=1,padding_mode=padding_mode)
        self.down2=nn.Conv2d(in_channels =scale*2, out_channels=scale*4, kernel_size =3, stride=2, padding=1,padding_mode=padding_mode)
        self.up1=nn.ConvTranspose2d(in_channels =scale*4, out_channels=scale*2, kernel_size =3, stride=2, padding=1,output_padding=1)
        self.up2=nn.ConvTranspose2d(in_channels =scale*2, out_channels=scale, kernel_size =3, stride=2, padding=1,output_padding=1)
        self.last_conv=nn.Conv2d(in_channels = scale, out_channels=3, kernel_size =7,padding=3,stride=1,padding_mode="reflect")
        self.tanh=nn.Tanh()    
        res_layers=[]
        self.resnet_layers=resnet_layers
        for i in range(resnet_layers):
            res_layers.append(ResBlock(channels=scale*4))
        self.res_layers=nn.ModuleList(res_layers)
    def forward(self, x):
        x=self.relu(self.norm1(self.conv1(x)))
        x=self.relu(self.norm2(self.down1(x)))
        x=self.relu(self.norm3(self.down2(x)))
        for i in range(self.resnet_layers):
            x=self.res_layers[i](x)+x
        x=self.relu(self.norm4(self.up1(x)))
        x=self.relu(self.norm5(self.up2(x))) 
        x=self.last_conv(x)
        x=self.tanh(x)
        return x
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1=nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3,padding=1,stride=1,padding_mode="reflect")
        self.conv2=nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3,padding=1,stride=1,padding_mode="reflect")
        self.relu=nn.ReLU(inplace=True)
        self.norm1=nn.InstanceNorm2d(channels)
        self.norm2=nn.InstanceNorm2d(channels)

    def forward(self, x):
        x=self.relu(self.norm1(self.conv1(x))) 
        x=self.norm2(self.conv2(x))
        return x


