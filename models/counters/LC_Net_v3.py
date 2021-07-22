from misc.utils import *


class LC_Net_v2(nn.Module):
    def __init__(self):
        super(LC_Net_v2, self).__init__()

        self.backbone = nn.Sequential(
            Conv(3, 32, k=3, s=1),
            Conv(32, 64, k=3, s=2),
            C3H(64, 64, n=1, shortcut=True),
            Conv(64, 128, k=3, s=2),
            C3H(128, 128, n=3, shortcut=True),
            Conv(128, 256, k=3, s=2),
            C3H(256, 256, n=3, shortcut=True),
            C3H(256, 256, n=1, shortcut=False, d=2)
        )

        self.head = nn.Sequential(
            Conv(256, 128, k=1, s=1),
            nn.Upsample(scale_factor=2),
            C3H(128, 128, n=1, shortcut=False, d=2),
            Conv(128, 64, k=1, s=1),
            nn.Upsample(scale_factor=2),
            C3H(64, 64, n=1, shortcut=False, d=2),
            Conv(64, 32, k=1, s=1),
            nn.Upsample(scale_factor=2),
            C3H(32, 32, n=1, shortcut=False, d=2),
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
            self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=autopad(k), bias=False)
        else:
            self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=d, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, d=1, e=0.5):  # ch_in, ch_out, shortcut, dilation, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, d=d)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3H(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, d=1, e=1.0):  # ch_in, ch_out, number, shortcut, dilation, expansion
        super(C3H, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, d, e=0.5) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1):  # ch_in, ch_out, kernel
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k=k)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))
