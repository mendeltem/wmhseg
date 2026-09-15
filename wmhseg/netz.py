"""Das Netz: U-Net 2D mit Bottleneck Attention Module (BAM) nach jedem Faltungsblock.

Nachbau von AQUA (Lee et al. 2023) nach BAM (Park, Woo, Lee, Kweon 2018,
arXiv:1807.06514). Die Architektur ist zeichengleich mit der des Trainings, sonst
passten die Gewichte nicht -- sie wird hier nur aus dem Trainingswerkzeug
herausgeloest, damit dieses Paket ohne dessen Umgebung laeuft.

Kanaele (32, 64, 128, 256), drei Downsampling-Stufen, Ausgang nativ 256x256.
Vier BAM im Encoder einschliesslich Bottleneck, drei im Decoder.
"""
import torch
import torch.nn as nn


class BAM(nn.Module):
    """Kanal- und Raumaufmerksamkeit, additiv verknuepft: F' = F + F * sigmoid(Mc + Ms)."""

    def __init__(self, C, r=16, d=4):
        super().__init__()
        r_c = max(1, C // r)
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(C, r_c), nn.BatchNorm1d(r_c), nn.ReLU(inplace=True),
            nn.Linear(r_c, C), nn.Unflatten(1, (C, 1, 1)),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(C, r_c, 1), nn.BatchNorm2d(r_c), nn.ReLU(inplace=True),
            nn.Conv2d(r_c, r_c, 3, padding=d, dilation=d),
            nn.BatchNorm2d(r_c), nn.ReLU(inplace=True),
            nn.Conv2d(r_c, r_c, 3, padding=d, dilation=d),
            nn.BatchNorm2d(r_c), nn.ReLU(inplace=True),
            nn.Conv2d(r_c, 1, 1),
        )

    def forward(self, F):
        m = torch.sigmoid(self.mlp(F) + self.spatial(F))
        return F + F * m


def _conv_block_2d(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class _BlockBam(nn.Module):
    def __init__(self, cin, cout, r=16, d=4):
        super().__init__()
        self.conv = _conv_block_2d(cin, cout)
        self.bam = BAM(cout, r=r, d=d)

    def forward(self, x):
        return self.bam(self.conv(x))


class UNetBam2d(nn.Module):
    def __init__(self, in_channels, out_channels=1, channels=(32, 64, 128, 256), r=16, d=4):
        super().__init__()
        c0, c1, c2, c3 = channels
        self.enc0 = _BlockBam(in_channels, c0, r, d)
        self.enc1 = _BlockBam(c0, c1, r, d)
        self.enc2 = _BlockBam(c1, c2, r, d)
        self.bottle = _BlockBam(c2, c3, r, d)
        self.dec2 = _BlockBam(c3 + c2, c2, r, d)
        self.dec1 = _BlockBam(c2 + c1, c1, r, d)
        self.dec0 = _BlockBam(c1 + c0, c0, r, d)
        self.head = nn.Conv2d(c0, out_channels, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        b = self.bottle(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up(d1), e0], dim=1))
        return self.head(d0)


def bau(kanaele):
    return UNetBam2d(in_channels=kanaele, out_channels=1, channels=(32, 64, 128, 256), r=16, d=4)
