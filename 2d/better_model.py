import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class DoubleConvRes(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvRes, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        save_x = x
        x = self.conv2(x)
        return x + save_x
    
class AttentionGate(nn.Module):
    def __init__(self, in_c):
        super(AttentionGate, self).__init__()

        self.skip_conv = nn.Conv2d(in_c, in_c, kernel_size=1, padding=0)
        self.gating_signal_conv = nn.Conv2d(in_c, in_c, kernel_size=1, padding=0)

        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, gating_signal, skip_connection):
        # print(gating_signal.shape)
        # print(skip_connection.shape)
        input = self.skip_conv(skip_connection)
        gs = self.gating_signal_conv(gating_signal)
        # print(input.shape)
        # print(gs.shape)
        out = self.relu(input + gs)
        out = self.output(out)
        return out * gating_signal


class Res_AttentionUnet2d(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[64, 128, 256],
    ):
        super(Res_AttentionUnet2d, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
            self.attention_gates.append(AttentionGate(feature)) 

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        save_input = x

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
 
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):

            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
               x = TF.resize(x, size=skip_connection.shape[2:])

            # print(self.attention_gates)
            # print(x.shape)
            # print(skip_connection.shape)

            attention_out = self.attention_gates[idx//2](x, skip_connection)

            concat_skip = torch.cat((attention_out, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        x = self.final_conv(x)
        m = nn.Tanh()
        x = m(x)

        #ritorno input sommato all'output
        return x + save_input

class ResUnet2d(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[64, 128, 256],
    ):
        super(ResUnet2d, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        save_input = x

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
 
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):

            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
               x = TF.resize(x, size=skip_connection.shape[2:])

            # print(self.attention_gates)
            # print(x.shape)
            # print(skip_connection.shape)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)


        x = self.final_conv(x)
        m = nn.Tanh()
        x = m(x)

        #ritorno input sommato all'output
        return x + save_input

class FullResUnet2d(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[64, 128, 256],
    ):
        super(FullResUnet2d, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConvRes(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConvRes(feature*2, feature))

        self.bottleneck = DoubleConvRes(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        #save_input = x

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
 
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):

            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
               x = TF.resize(x, size=skip_connection.shape[2:])

            # print(self.attention_gates)
            # print(x.shape)
            # print(skip_connection.shape)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)


        x = self.final_conv(x)
        m = nn.ReLU()
        x = m(x)

        #ritorno input sommato all'output
        #return x + save_input
        return x

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def test():
    x = torch.randn((1, 1, 836, 836))
    model = FullResUnet2d(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)  
    
    print(get_n_params(model))

    assert preds.shape == x.shape

if __name__ == "__main__":
    test()
