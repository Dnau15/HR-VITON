import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


class ConditionGenerator(nn.Module):
    def __init__(self, opt, input1_nc, input2_nc, output_nc, ngf=64, 
norm_layer=nn.BatchNorm2d):
        super(ConditionGenerator, self).__init__()
        self.warp_feature = opt.warp_feature
        self.out_layer_opt = opt.out_layer

        self.ClothEncoder = nn.Sequential(
            ResBlock(input1_nc, ngf, norm_layer=norm_layer, scale='down'),  
# 128
            ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'),  
# 64
            ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, 
scale='down'),  # 32
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, 
scale='down'),  # 16
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, 
scale='down')  # 8
        )

        self.PoseEncoder = nn.Sequential(
            ResBlock(input2_nc, ngf, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, 
scale='down'),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, 
scale='down'),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, 
scale='down')
        )

        self.conv = ResBlock(ngf * 4, ngf * 8, norm_layer=norm_layer, 
scale='same')

        if opt.warp_feature == 'T1':
            # in_nc -> skip connection + T1, T2 channel
            self.SegDecoder = nn.Sequential(
                ResBlock(ngf * 8, ngf * 4, norm_layer=norm_layer, 
scale='up'),  # 16
                ResBlock(ngf * 4 * 2 + ngf * 4, ngf * 4, 
norm_layer=norm_layer, scale='up'),  # 32
                ResBlock(ngf * 4 * 2 + ngf * 4, ngf * 2, 
norm_layer=norm_layer, scale='up'),  # 64
                ResBlock(ngf * 2 * 2 + ngf * 4, ngf, 
norm_layer=norm_layer, scale='up'),  # 128
                ResBlock(ngf * 1 * 2 + ngf * 4, ngf, 
norm_layer=norm_layer, scale='up')  # 256
            )
        if opt.warp_feature == 'encoder':
            # in_nc -> [x, skip_connection, 
warped_cloth_encoder_feature(E1)]
            self.SegDecoder = nn.Sequential(
                ResBlock(ngf * 8, ngf * 4, norm_layer=norm_layer, 
scale='up'),  # 16
                ResBlock(ngf * 4 * 3, ngf * 4, norm_layer=norm_layer, 
scale='up'),  # 32
                ResBlock(ngf * 4 * 3, ngf * 2, norm_layer=norm_layer, 
scale='up'),  # 64
                ResBlock(ngf * 2 * 3, ngf, norm_layer=norm_layer, 
scale='up'),  # 128
                ResBlock(ngf * 1 * 3, ngf, norm_layer=norm_layer, 
scale='up')  # 256
            )
        if opt.out_layer == 'relu':
            self.out_layer = ResBlock(ngf + input1_nc + input2_nc, 
output_nc, norm_layer=norm_layer, scale='same')
        if opt.out_layer == 'conv':
            self.out_layer = nn.Sequential(
                ResBlock(ngf + input1_nc + input2_nc, ngf, 
norm_layer=norm_layer, scale='same'),
                nn.Conv2d(ngf, output_nc, kernel_size=1, bias=True)
            )

        # Cloth Conv 1x1
        self.conv1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
        )

        # Person Conv 1x1
        self.conv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
        )

        self.flow_conv = nn.ModuleList([
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, 
bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, 
bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, 
bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, 
bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, 
bias=True),
        ]
        )

        self.bottleneck = nn.Sequential(
            nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, 
stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, 
stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, 
stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, 
padding=1, bias=True), nn.ReLU()),
        )

    def normalize(self, x):
        return x

    def forward(self, opt, input1, input2, upsample='bilinear'):
        E1_list = []
        E2_list = []
        flow_list = []
        # warped_grid_list = []

        # Feature Pyramid Network
        for i in range(5):
            if i == 0:
                E1_list.append(self.ClothEncoder[i](input1))
                E2_list.append(self.PoseEncoder[i](input2))
            else:
                E1_list.append(self.ClothEncoder[i](E1_list[i - 1]))
                E2_list.append(self.PoseEncoder[i](E2_list[i - 1]))

        # Compute Clothflow
        for i in range(5):
            N, _, iH, iW = E1_list[4 - i].size()
            grid = make_grid(N, iH, iW, opt)

            if i == 0:
                T1 = E1_list[4 - i]  # (ngf * 4) x 8 x 6
                T2 = E2_list[4 - i]
                E4 = torch.cat([T1, T2], 1)

                flow = self.flow_conv[i](self.normalize(E4)).permute(0, 2, 
3, 1)
                flow_list.append(flow)

                x = self.conv(T2)
                x = self.SegDecoder[i](x)

            else:
                T1 = F.interpolate(T1, scale_factor=2, mode=upsample) + 
self.conv1[4 - i](E1_list[4 - i])
                T2 = F.interpolate(T2, scale_factor=2, mode=upsample) + 
self.conv2[4 - i](E2_list[4 - i])

                flow = F.interpolate(flow_list[i - 1].permute(0, 3, 1, 2), 
scale_factor=2, mode=upsample).permute(0, 2,
                                                                                                                  
3,
                                                                                                                  
1)  # upsample n-1 flow
                flow_norm = torch.cat(
                    [flow[:, :, :, 0:1] / ((iW / 2 - 1.0) / 2.0), flow[:, 
:, :, 1:2] / ((iH / 2 - 1.0) / 2.0)], 3)
                warped_T1 = F.grid_sample(T1, flow_norm + grid, 
padding_mode='border')

                flow = flow + self.flow_conv[i](
                    self.normalize(torch.cat([warped_T1, self.bottleneck[i 
- 1](x)], 1))).permute(0, 2, 3, 1)  # F(n)
                flow_list.append(flow)

                if self.warp_feature == 'T1':
                    x = self.SegDecoder[i](torch.cat([x, E2_list[4 - i], 
warped_T1], 1))
                if self.warp_feature == 'encoder':
                    warped_E1 = F.grid_sample(E1_list[4 - i], flow_norm + 
grid, padding_mode='border')
                    x = self.SegDecoder[i](torch.cat([x, E2_list[4 - i], 
warped_E1], 1))

        N, _, iH, iW = input1.size()
        grid = make_grid(N, iH, iW, opt)

        flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), 
scale_factor=2, mode=upsample).permute(0, 2, 3, 1)
        flow_norm = torch.cat(
            [flow[:, :, :, 0:1] / ((iW / 2 - 1.0) / 2.0), flow[:, :, :, 
1:2] / ((iH / 2 - 1.0) / 2.0)], 3)
        warped_input1 = F.grid_sample(input1, flow_norm + grid, 
padding_mode='border')

        x = self.out_layer(torch.cat([x, input2, warped_input1], 1))

        warped_c = warped_input1[:, :-1, :, :]
        warped_cm = warped_input1[:, -1:, :, :]

        return flow_list, x, warped_c, warped_cm


def make_grid(N, iH, iW, opt):
    grid_x = torch.linspace(-1.0, 1.0, iW).view(1, 1, iW, 1).expand(N, iH, 
-1, -1)
    grid_y = torch.linspace(-1.0, 1.0, iH).view(1, iH, 1, 1).expand(N, -1, 
iW, -1)
    if opt.cuda:
        grid = torch.cat([grid_x, grid_y], 3).cuda()
    else:
        grid = torch.cat([grid_x, grid_y], 3)
    return grid


class ResBlock(nn.Module):
    def __init__(self, in_nc, out_nc, scale='down', 
norm_layer=nn.BatchNorm2d):
        super(ResBlock, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        assert scale in ['up', 'down', 'same'], "ResBlock scale must be in 
'up' 'down' 'same'"

        if scale == 'same':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=1, 
bias=True)
        if scale == 'up':
            self.scale = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_nc, out_nc, kernel_size=1, bias=True)
            )
        if scale == 'down':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=2, 
padding=1, bias=use_bias)

        self.block = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, 
bias=use_bias),
            norm_layer(out_nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, 
bias=use_bias),
            norm_layer(out_nc)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.scale(x)
        return self.relu(residual + self.block(residual))


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, opt, layids=None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        if opt.cuda:
            self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], 
y_vgg[i].detach())
        return loss


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, 
target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != 
input.numel()))
            if create_label:
                real_tensor = 
self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, 
requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != 
input.numel()))
            if create_label:
                fake_tensor = 
self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, 
requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, 
target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], 
target_is_real)
            return self.loss(input[-1], target_tensor)

