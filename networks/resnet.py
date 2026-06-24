import sys
sys.path.append('..')


import torch
import torch.nn as nn

# from utils.sig_env import batch_signal_autocorrelation
# from utils.emd_fn import envelope_signal
from utils.ts_transform import transform_value


def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def get_layer1(model):
    for child in model.children():
        if isinstance(child, nn.Conv1d):
            return child
        elif isinstance(child, nn.Sequential):
            for sub_child in child.children():
                if isinstance(sub_child, nn.Conv1d):
                    return sub_child


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x1(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, input_channels=1, name='resnet', transform_in_model: bool=True, zero_mean: bool=True):
        super(ResNet, self).__init__()
        self.name = name
        self.in_planes = 64
        self.transform_in_model = transform_in_model
        self.zero_mean = zero_mean
        self.squeeze = False
        self.clamp_range = (-1, 1)

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = 512 * block.expansion
        self.fc = nn.Linear(self.feature_dim, num_classes)

        self.weights_init()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _forward(self, x, return_feats=False):
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        features.append(x.detach().cpu().numpy())  # avoid GPU memory leak

        x = self.layer1(x)
        features.append(x.detach().cpu().numpy())
        x = self.layer2(x)
        features.append(x.detach().cpu().numpy())
        x = self.layer3(x)
        features.append(x.detach().cpu().numpy())
        x = self.layer4(x)
        features.append(x.detach().cpu().numpy())

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features.append(x.detach().cpu().numpy())
        x = self.fc(x)

        if return_feats:
            return x, features
        return x

    def features(self, x):
        x = self.transform_fn(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def classify_features(self, features):
        return self.fc(features)
    
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def transform_fn(self, x):
        if self.transform_in_model:
            x = transform_value(x, zero_mean=self.zero_mean)
        
        # if self.squeeze:
        #     x = batch_signal_autocorrelation(x)
            # x = envelope_signal(x, method='peak')
            # x = quantize_signal(x, self.squeeze_level)  # works well
            # x = fuzzy_signal(x, self.fuzzy_factor)
            # x = median_filter(x, kernel_size=3)
            # x = apply_filter(x, 'lowpass', 3_000, 12_000, 5)
            # x = wavelet_denoising_tensor(x, wavelet='db4', level=1)  # bad performance
        
        x = x - x.mean(dim=-1, keepdim=True)
        x = torch.clamp(x, *self.clamp_range)
        return x
    
    def unfreeze_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    
    def forward(self, x, return_feats=False, use_dropout=False):
        x = self.transform_fn(x)
        if use_dropout:
            self.unfreeze_dropout()
        return self._forward(x, return_feats)
    
    def predict(self, x, return_feats=False, use_dropout=False):
        self.eval()
        return self.forward(x, return_feats, use_dropout)  #.softmax(dim=1)


def resnet18(num_classes=10, input_channels=1, name='resnet18', **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, input_channels=input_channels, name=name, **kwargs)


def resnet34(num_classes=10, input_channels=1, name='resnet34', **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, input_channels=input_channels, name=name, **kwargs)


def resnet50(num_classes=10, input_channels=1, name='resnet50', **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, input_channels=input_channels, name=name, **kwargs)


def resnet101(num_classes=10, input_channels=1, name='resnet101', **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, input_channels=input_channels, name=name, **kwargs)


if __name__ == "__main__":
    # arr = torch.randn(32, 1, 2048)
    # models = [resnet18(), resnet34(), resnet50(), resnet101()]
    # outs = [m(arr) for m in models]
    # for out in outs:
    #     print(out.shape)
    # calculate the number of conv layers in each model
    models = [resnet18(), resnet34(), resnet50(), resnet101()]
    for model in models:
        print(model.name, sum(1 for m in model.modules() if isinstance(m, nn.Conv1d)))

    print(get_layer1(resnet18()))
