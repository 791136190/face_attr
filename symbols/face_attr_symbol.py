import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import yaml

import torch.nn

from symbols.common_symbols import Conv, MaxPool, DecConvBn, FC, FCRelu, FCSigmoid, Add

class FaceAttr(torch.nn.Module):
    def __init__(self, training=True, w=1.0, use_dw=False):
        super(FaceAttr, self).__init__()
        self.is_train = training
        self.w = w
        self.use_dw = use_dw

        self.conv1 = Conv(c_in=3, c_out=int(16 * w), k=5, s=2)
        self.conv2 = Conv(c_in=int(16 * w), c_out=int(32 * w), k=3, s=1)
        self.conv3 = Conv(c_in=int(32 * w), c_out=int(32 * w), k=3, s=2)

        self.conv4 = Conv(c_in=int(32 * w), c_out=int(32 * w), k=3, s=1)
        self.conv5 = Conv(c_in=int(32 * w), c_out=int(32 * w), k=3, s=2)

        self.pool1 = MaxPool(size=2, stride=2)

        self.add1 = Add()

        self.conv6 = Conv(c_in=int(32 * w), c_out=int(32 * w), k=3, s=1)
        self.conv7 = Conv(c_in=int(32 * w), c_out=int(32 * w), k=3, s=2, p=0)

        self.conv8 = Conv(c_in=int(32 * w), c_out=int(32 * w), k=3, s=1, p=0)
        self.dec1 = DecConvBn(c_in=int(32 * w), c_out=int(32 * w))

        self.add2 = Add()

        self.conv9 = Conv(c_in=int(32 * w), c_out=int(64 * w), k=3, s=1)
        self.conv10 = Conv(c_in=int(64 * w), c_out=int(64 * w), k=3, s=2, p=0)

        self.conv11 = Conv(c_in=int(32 * w), c_out=int(32 * w), k=3, s=1)
        self.conv12 = Conv(c_in=int(32 * w), c_out=int(64 * w), k=3, s=2)
        self.conv13 = Conv(c_in=int(64 * w), c_out=int(64 * w), k=3, s=1, p=0)

        self.fc1 = FCRelu(c_in=int(64 * w), c_out=int(64 * w))
        self.fl1 = torch.nn.Flatten()
        self.fc2 = FCSigmoid(c_in=int(64 * w), c_out=1)        # score
        self.fc3 = FCRelu(c_in=int(64 * w), c_out=int(32 * w))
        self.fc4 = FCSigmoid(c_in=int(32 * w), c_out=1)        # gender
        self.fc5 = FCRelu(c_in=int(64 * w), c_out=int(128 * w))
        self.fc6 = FCSigmoid(c_in=int(128 * w), c_out=1)       # age

        self.fc7 = FCRelu(c_in=int(64 * w), c_out=int(64 * w))
        self.fl2 = torch.nn.Flatten()
        self.fc8 = FCRelu(c_in=int(64 * w), c_out=int(64 * w))
        self.fc9 = FC(c_in=int(64 * w), c_out=10)               # land
        self.fc10 = FCRelu(c_in=int(64 * w), c_out=int(32 * w))
        self.fc11 = FCSigmoid(c_in=int(32 * w), c_out=1)        # glass
        self.fc12 = FCRelu(c_in=int(64 * w), c_out=int(32 * w))
        self.fc13 = FCSigmoid(c_in=int(32 * w), c_out=1)        # smile
        self.fc14 = FCRelu(c_in=int(64 * w), c_out=int(32 * w))
        self.fc15 = FCSigmoid(c_in=int(32 * w), c_out=1)        # hat
        self.fc16 = FCRelu(c_in=int(64 * w), c_out=int(32 * w))
        self.fc17 = FCSigmoid(c_in=int(32 * w), c_out=1)        # mask

    def forward(self, x):
        # x = x * 0.0078125 # 0 ~ 2
        x = x * 0.003921 # 0 ~ 1

        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)

        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        p1 = self.pool1(c3)

        add1 = self.add1((c5, p1))

        lc1 = self.conv6(c5)
        lc2 = self.conv7(lc1)

        rc1 = self.conv8(add1)

        d1 = self.dec1(lc2)
        
        add2 = self.add2((d1, rc1))

        lc3 = self.conv9(lc2)
        lc4 = self.conv10(lc3)
        lc4 = self.fl1(lc4)
        lfc = self.fc1(lc4)

        rc2 = self.conv11(add2)
        rc3 = self.conv12(rc2)
        rc4 = self.conv13(rc3)
        rc4 = self.fl2(rc4)
        rfc = self.fc7(rc4)

        score = self.fc2(lfc)
        gender = self.fc4(self.fc3(lfc))
        age = self.fc6(self.fc5(lfc))

        land = self.fc9(self.fc8(rfc))
        glass = self.fc11(self.fc10(rfc))
        smile = self.fc13(self.fc12(rfc))
        hat = self.fc15(self.fc14(rfc))
        mask = self.fc17(self.fc16(rfc))

        # out = torch.cat((score, gender, age, land, glass, smile, hat, mask), dim=1)
        # return out
        return score, gender, age, land, glass, smile, hat, mask

if __name__ == "__main__":
    print("end all get face attr model !!!")