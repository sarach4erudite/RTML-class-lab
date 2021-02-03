from ResidualSEBasicBlock import ResidualSEBasicBlock
from Resnet import ResNet

def ResSENet18(num_classes = 10):
    return ResNet(ResidualSEBasicBlock, [2, 2, 2, 2], num_classes)

def ResSENet50(num_classes = 10):
    '''
    First conv layer: 1
    4 residual blocks with [3, 4, 6, 3] sets of three convolutions each: 3*3 + 4*3 + 6*3 + 3*3 = 48
    last FC layer: 1
    Total layers: 1+48+1 = 50
    '''
    return ResNet(ResidualSEBasicBlock, [3, 4, 6, 3], num_classes)


def ResSENet101(num_classes = 10):
    '''
    First conv layer: 1
    4 residual blocks with [3, 4, 23, 3] sets of three convolutions each: 3*3 + 4*3 + 23*3 + 3*3 = 99
    last FC layer: 1
    Total layers: 1+99+1 = 101
    '''
    return ResNet(ResidualSEBasicBlock, [3, 4, 23, 3], num_classes)


def ResSENet152(num_classes = 10):
    '''
    First conv layer: 1
    4 residual blocks with [3, 8, 36, 3] sets of three convolutions each: 3*3 + 8*3 + 36*3 + 3*3 = 150
    last FC layer: 1
    Total layers: 1+150+1 = 152
    '''
    return ResNet(ResidualSEBasicBlock, [3, 8, 36, 3], num_classes)

if __name__ == '__main__':
    import torch
    net = ResSENet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
    