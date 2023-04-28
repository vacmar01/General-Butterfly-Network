import torch
import torch.nn as nn

def crop(input1, input2):
    assert input1.shape[0] == input2.shape[0]
    assert input1.shape[2] - input2.shape[2] in (0, 1)
    assert input1.shape[3] - input2.shape[3] in (0, 1)

    return (input1[:, :, :input2.shape[2], :input2.shape[3]], input2)

class conv_blk(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, drop_out):
        super(conv_blk, self).__init__()
        self.blk = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5 if drop_out else 0),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self, input):
        output = self.blk(input)
        return output

class deconv_blk(nn.Module):
    def __init__(self, in_channels):
        super(deconv_blk, self).__init__()
        self.blk = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=in_channels,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        output = self.blk(input)
        return output

class green(nn.Module):
    def __init__(self, channels, pos):
        super(green, self).__init__()
        self.conv =conv_blk(in_channels=channels[pos],
                            out_channels=channels[pos+1],
                            kernel_size=1 if pos == 12 else 3,
                            stride=1,
                            padding=0 if pos == 12 else 1,
                            drop_out=True if pos in (4, 5) else False,
                            )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        output_side = self.conv(input)
        output_side_pad = nn.functional.pad(output_side, (0, (output_side.shape[3] % 2), 0, (output_side.shape[2] % 2), 0, 0, 0, 0))
        output_main = self.pool(output_side_pad)
        return output_main, output_side

class purple(nn.Module):
    def __init__(self, channels, pos):
        super(purple, self).__init__()
        self.deconv = deconv_blk(in_channels=channels[pos])
        self.conv = conv_blk(in_channels=channels[pos] + (channels[13 - pos] if pos < 9 else channels[12 - pos]),
                             out_channels=channels[pos+1],
                             kernel_size=3, stride=1, padding=1, drop_out=False,
                             )

    def forward(self, input_main, input_side):
        output = self.deconv(input_main)

        output = torch.cat(crop(output, input_side), dim=1)
        output = self.conv(output)
        return output

class red(nn.Module):
    def __init__(self, channels, pos):
        super(red, self).__init__()
        self.blk = nn.Conv2d(
            in_channels=channels[pos],
            out_channels=channels[pos+1],
            kernel_size=1,stride=1, padding=0,
        )

    def forward(self, input):
        output = self.blk(input)
        return output

class in_arm(nn.Module):
    def __init__(self, channels):
        super(in_arm, self).__init__()
        self.green0 = green(channels, pos=0)
        self.green1 = green(channels, pos=1)
        self.green2 = green(channels, pos=2)

    def forward(self, input):
        output_main, output_side_0 = self.green0(input)
        output_main, output_side_1 = self.green1(output_main)
        output_main, output_side_2 = self.green2(output_main)
        return output_main, output_side_0, output_side_1, output_side_2


class out_arm(nn.Module):
    def __init__(self, channels):
        super(out_arm, self).__init__()
        self.purple0 = purple(channels, pos=9)
        self.purple1 = purple(channels, pos=10)
        self.purple2 = purple(channels, pos=11)
        self.red = red(channels, pos=12)

    def forward(self, input_main, input_side_2, input_side_1, input_side_0):
        output = self.purple0(input_main, input_side_2)
        output = self.purple1(output, input_side_1)
        output = self.purple2(output, input_side_0)
        output = self.red(output)
        return output
    
class bodyXV(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.green0 = green(channels, pos=4)
        self.green1 = green(channels, pos=5)
        self.conv = conv_blk(in_channels=channels[6],
                             out_channels=channels[7],
                             kernel_size=3, stride=1, padding=1, drop_out=True,
                             )
        self.purple0 = purple(channels, pos=7)
        self.purple1 = purple(channels, pos=8)

    def forward(self, inp):
        #output = torch.cat(crop(input_sag, input_cor, input_ax), dim=1)
        output = torch.cat(tuple(inp), dim=1)
        output, side_0 = self.green0(output)
        output, side_1 = self.green1(output)
        output = self.conv(output)
        output = self.purple0(output, side_1)
        output = self.purple1(output, side_0)
        return output

class BtrflyNetXV(nn.Module):
    def __init__(self, channels, n_inp=3):
        super().__init__()
        channels = list(channels)
        channels.insert(4, n_inp*channels[3])
        channels.insert(5, n_inp*channels[3])
        self.n_inp = n_inp
        self.input_arms = nn.ModuleList([in_arm(channels)]*self.n_inp)
        self.input_arm_cor = in_arm(channels)
        self.input_arm_ax = in_arm(channels)
        self.body = bodyXV(channels)
        self.output_arms = nn.ModuleList([out_arm(channels)]*self.n_inp)
        self.output_arm_cor = out_arm(channels)
        self.output_arm_ax = out_arm(channels)

    def forward(self, inp):
        assert len(inp) == self.n_inp, "Number of inputs specified while initiating the model and number of inputs to `forward` do not match"
        inp_features = []
        for i in range(self.n_inp):
            body, side0, side1, side2 = self.input_arms[i](inp[i])
            inp_features.append((body, side0, side1, side2))
        body_out = self.body([ft[0] for ft in inp_features])
        outp = []
        for i in range(self.n_inp):
            single_outp = self.output_arms[i](body_out, *inp_features[i][1:][::-1])
            outp.append(single_outp)
        return outp

channels = (1, 32, 64, 128, 512, 1024, 512, 512, 256, 128, 64, 1)

net = BtrflyNetXV(channels, 5)

# inp = [torch.randn([32,1,128,128])]*5

# outp = net.forward(inp)

# assert len(outp) == 5
# assert outp[0].shape == (32,1,128,128)




