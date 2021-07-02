from misc.utils import *


class LC_Net(nn.Module):
    def __init__(self):
        super(LC_Net, self).__init__()

        self.backbone = nn.Sequential(
            Conv(3, 32, k=3, s=1),
            Conv(32, 32, k=3, s=1),
            nn.MaxPool2d(2),
            LCM(32, 64, n=1, shortcut=True),
            nn.MaxPool2d(2),
            LCM(64, 128, n=2, shortcut=True),
            nn.MaxPool2d(2),
            LCM(128, 256, n=4, shortcut=True),
            LCM(256, 256, n=4, shortcut=True, d=2)
        )

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2),
            LCM(256, 128, n=1, shortcut=True),
            Conv(128, 128, k=3, s=1, d=2),
            Conv(128, 128, k=3, s=1, d=2),
            nn.Upsample(scale_factor=2),
            LCM(128, 64, n=1, shortcut=True),
            Conv(64, 64, k=3, s=1, d=2),
            Conv(64, 64, k=3, s=1, d=2),
            nn.Upsample(scale_factor=2),
            LCM(64, 32, n=1, shortcut=True),
            LCM(32, 32, n=1, shortcut=False, d=2),
            SM(k=3, s=1),
            LCM(32, 32, n=1, shortcut=False),
            Conv(32, 2, k=1, s=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        
        return x


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        if d == 1:
            self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=autopad(k))
        else:
            self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=d, dilation=d)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, d=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, d=d)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class LCM(nn.Module):
    # Localized Counting Module
    def __init__(self, c1, c2, n=1, shortcut=True, d=1):
        super(LCM, self).__init__()
        self.cv = Conv(c1, c2, 3, 1, d=d)
        self.m = nn.Sequential(*[Bottleneck(c2, c2, shortcut, d, e=0.5) for _ in range(n)])

    def forward(self, x):
        return self.m(self.cv(x))


class SM(nn.Module):
    # Sharpening Module
    def __init__(self, k=3, s=1):
        super(SM, self).__init__()
        self.avg = nn.AvgPool2d(k, stride=s, padding=autopad(k))
        self.max = nn.MaxPool2d(k, stride=s, padding=autopad(k))

    def forward(self, x):
        x = self.max(self.avg(x))
        return x
