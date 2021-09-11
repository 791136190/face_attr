from itertools import count
import torch
import torch.nn as nn

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c_in, c_out, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DepthwiseConv(nn.Module):
    # Standard convolution
    def __init__(self, c_in, c_out, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(DepthwiseConv, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_in, k, s, autopad(k, p), groups=c_in, bias=False)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.act1 = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        self.conv2 = nn.Conv2d(c_in, c_out, 1, 1, 0, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act2 = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        out1 = self.act1(self.bn1(self.conv1(x)))
        out2 = self.act2(self.bn2(self.conv2(out1)))
        return out2

class DecConvBn(nn.Module):
    # dec convolution
    def __init__(self, c_in, c_out, k=4, s=2):  # ch_in, ch_out, kernel, stride
        super(DecConvBn, self).__init__()
        self.conv = nn.ConvTranspose2d(c_in, c_out, k, s, 1)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        return self.bn(self.conv(x))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class Add(nn.Module):
    # add a list of tensors along dimension
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x):
        return torch.add(*x)

class MaxPool(nn.Module):
    # Standard MaxPool
    def __init__(self, size, stride):  # kernel_size, stride
        super(MaxPool, self).__init__()
        self.pool = nn.MaxPool2d(size, stride)

    def forward(self, x):
        return self.pool(x)

class FC(nn.Module):
    # Standard Linear
    def __init__(self, c_in, c_out):  # ch_in, ch_out
        super(FC, self).__init__()
        self.fc = nn.Linear(c_in, c_out)
    def forward(self, x):
        return self.fc(x)

class FCRelu(nn.Module):
    # Standard Linear
    def __init__(self, c_in, c_out):  # ch_in, ch_out
        super(FCRelu, self).__init__()
        self.fc = nn.Linear(c_in, c_out)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.fc(x))

class FCSigmoid(nn.Module):
    # Standard Linear
    def __init__(self, c_in, c_out):  # ch_in, ch_out
        super(FCSigmoid, self).__init__()
        self.fc = nn.Linear(c_in, c_out)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.fc(x))


if __name__ == "__main__":
    print("end all common symbols !!!")