import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity  # residual, skip connection 
        out = self.relu(out)
        
        return out 
        

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # default initialization 
        
        # weight initialization 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        
    def _make_layer(self, block, planes, blocks, stride):
        downsample = None 
        
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes*block.expansion, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(planes * block.expansion)
            )
            
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        
        self.in_planes = planes * block.expansion 
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_planes, planes
                )
            )
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)  # input: 1x64x32x32, output: 1x64x32x32
        x = self.layer2(x)  # input: 1x64x32x32, output: 1x128x16x16
        x = self.layer3(x)  # input: 1x128x16x16, output: 1x258x8x8
        x = self.layer4(x)  # input: 1x258x8x8,   output: 1x512x4x4
        
        x = self.avgpool(x)      # output: 1x512x1x1
        x = torch.flatten(x, 1)  # output: 1x512
        x = self.fc(x)           # output: 1x10
        
        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


# def ResNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


if __name__ == '__main__':
    test()
