from ResidualSEBasicBlock import ResidualSEBasicBlock
from Resnet import ResNet

def ResSENet18(num_classes = 10):
    return ResNet(ResidualSEBasicBlock, [2, 2, 2, 2], num_classes)

if __name__ == '__main__':
    import torch
    net = ResSENet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
    