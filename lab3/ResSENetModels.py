from ResidualSEBasicBlock import ResidualSEBasicBlock
from Resnet import ResNet

def ResSENet18(num_classes = 10):
    return ResNet(ResidualSEBasicBlock, [2, 2, 2, 2], num_classes)