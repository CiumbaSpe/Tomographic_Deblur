import torch
import torch.nn as nn
# import torchvision.transforms.functional as TF
# from torch_receptive_field import receptive_field
# from torchscan import summary
#from torchsummary import summary


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class ResUnet3d(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[64, 128],
    ):
        super(ResUnet3d, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

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

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        x = self.final_conv(x)
        
        m = nn.Tanh()
        x = m(x)

        x = x+save_input
        #ritorno input sommato all'output
        return x 

class prova(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1,
    ):
        super(prova, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.up = nn.ConvTranspose3d(
            128, 64, kernel_size=2, stride=2,
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv3d(64, 1, kernel_size=1)

        
    def forward(self, x):
        print("pre conv1")
        conv1 = self.conv1(x)
        x = self.pool(conv1)
        
        x = self.conv2(x)
        x = self.conv(x)


        x = self.up(x)        
        # x = torch.cat([x, conv1], dim=1)
        
        x = self.conv3(x)
        # x = self.conv3(x)
        # x = self.conv3(x)


        print("pre conv1")

        out = self.final_conv(x)

        return out


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def test():
    # x = torch.randn((1, 1, 4, 836, 836))
    model = ResUnet3d(in_channels=1, out_channels=1)
    # preds = model(x)
    summary(model, (1, 4, 836, 836))
    # model.eval()
    # receptive_field_dict = receptive_field(model, (1, 4, 836, 836))
    # summary(model,  (1, 4, 836, 836), receptive_field=True, max_depth=0)
    #receptive_field_for_unit(receptive_field_dict, "2", (1,1))
    #print(preds.shape)
    #print(x.shape)
    
    #print(get_n_params(model))

    #assert preds.shape == x.shape

if __name__ == "__main__":
    test()
