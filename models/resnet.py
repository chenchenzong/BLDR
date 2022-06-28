'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, planes, stride=1):
    return nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

def branchBottleNeck(channel_in, channel_out, kernel_size):
    middle_channel = channel_out//4
    return nn.Sequential(
        nn.Conv2d(channel_in, middle_channel, kernel_size=1, stride=1),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),
        
        nn.Conv2d(middle_channel, middle_channel, kernel_size=kernel_size, stride=kernel_size),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),
        
        nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        )
        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear1 = nn.Linear(512*block.expansion, num_classes)
        self.linear2 = nn.Linear(512*block.expansion, num_classes)

        self.downsample1_1 = nn.Sequential(
                            conv1x1(64 * block.expansion, 512 * block.expansion, stride=8),
                            nn.BatchNorm2d(512 * block.expansion),
        )
        self.bottleneck1_1 = branchBottleNeck(64 * block.expansion, 512 * block.expansion, kernel_size=8)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc1 = nn.Linear(512 * block.expansion, num_classes)


        self.downsample2_1 = nn.Sequential(
                            conv1x1(128 * block.expansion, 512 * block.expansion, stride=4),
                            nn.BatchNorm2d(512 * block.expansion),
            )
        self.bottleneck2_1 = branchBottleNeck(128 * block.expansion, 512 * block.expansion, kernel_size=4)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc2 = nn.Linear(512 * block.expansion, num_classes)


        self.downsample3_1 = nn.Sequential(
                            conv1x1(256 * block.expansion, 512 * block.expansion, stride=2),
                            nn.BatchNorm2d(512 * block.expansion),
        )
        self.bottleneck3_1 = branchBottleNeck(256 * block.expansion, 512 * block.expansion, kernel_size=2)
        self.avgpool3 = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc3 = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        middle_output1 = self.bottleneck1_1(out)
        middle_output1 = self.avgpool1(middle_output1)
        middle1_fea = middle_output1
        middle_output1 = torch.flatten(middle_output1, 1)
        middle_output1 = self.middle_fc1(middle_output1)
        out = self.layer2(out)
        middle_output2 = self.bottleneck2_1(out)
        middle_output2 = self.avgpool2(middle_output2)
        middle2_fea = middle_output2
        middle_output2 = torch.flatten(middle_output2, 1)
        middle_output2 = self.middle_fc2(middle_output2)
        out = self.layer3(out)
        middle_output3 = self.bottleneck3_1(out)
        middle_output3 = self.avgpool3(middle_output3)
        middle3_fea = middle_output3
        middle_output3 = torch.flatten(middle_output3, 1)
        middle_output3 = self.middle_fc3(middle_output3)
        out = self.layer4(out)
        final_fea = self.avgpool(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out1 = self.linear1(out)
        out2 = self.linear2(out)
        return out1, out2, middle_output1, middle_output2, middle_output3, final_fea, middle1_fea, middle2_fea, middle3_fea

def PreResNet18(num_classes):
    return ResNet(PreActBlock, [2,2,2,2],num_classes=num_classes)

def ResNet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2],num_classes=num_classes)

def ResNet34(num_classes):
    return ResNet(BasicBlock, [3,4,6,3],num_classes=num_classes)

def ResNet50(num_classes):
    return ResNet(Bottleneck, [3,4,6,3],num_classes=num_classes)

def ResNet101(num_classes):
    return ResNet(Bottleneck, [3,4,23,3],num_classes=num_classes)

def ResNet152(num_classes):
    return ResNet(Bottleneck, [3,8,36,3],num_classes=num_classes)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
