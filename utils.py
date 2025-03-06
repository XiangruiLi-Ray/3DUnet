import torch
import torch.nn as nn

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res = False) -> None:
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res

        # first convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # use residual connection
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)

            if self.same_channels:
                out = x + x2
            else:
                # apply 1x1 convolutional layer to match dimensions before adding residual connection
                shortcut = nn.Conv3d(x.shape[1], x2.shape[1], 1, 1, 0).to(x.device)
                out = shortcut(x) + x2

            return out / 1.414  # normalize output by deviding sqrt 2 to keep the variance similar
        
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2
    

class UnetUp3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UnetUp3D, self).__init__()

        layers = [
            nn.ConvTranspose3d(in_channels, out_channels, 2, 2),
            DoubleConv3D(out_channels, out_channels),
            DoubleConv3D(out_channels, out_channels)
        ]

        self.model =nn.Sequential(*layers)

    def forward(self, x, skip) -> torch.Tensor:
        # concatenate x with the skip connection tensor
        x = torch.cat((x, skip), 1)
        return self.model(x)


class UnetDown3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UnetDown3D, self).__init__()

        layers = [DoubleConv3D(in_channels, out_channels),
                  DoubleConv3D(out_channels, out_channels),
                  nn.MaxPool3d(2)]
        
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    

class EmbedFC(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int):
        super(EmbedFC, self).__init__()

        self.in_dim = in_dim

        layers = [
            nn.Linear(in_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten input tensor
        x = x.view(-1, self.in_dim)
        return self.model(x)
    

