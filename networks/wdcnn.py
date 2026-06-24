import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from utils.ts_transform import transform_value, wavelet_denoising_tensor, apply_filter
# from utils.fn_tools import softmax, quantize_signal, fuzzy_signal, median_filter, drop_signal
# from utils.emd_fn import envelope_signal
# from utils.sig_env import batch_signal_autocorrelation


class WDCNN(nn.Module):
    # Paper: A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation
    # Ability on Raw Vibration Signals
    def __init__(self, input_length, num_classes, transform_in_model: bool=False, zero_mean: bool=True, 
                 in_channels=1, clamp_range=(-1, 1), track_running_stats: bool=True):
        super().__init__()
        self.transform_in_model = transform_in_model
        self.clamp_range = clamp_range
        self.zero_mean = zero_mean
        self.name = f'wdcnn_zm_{zero_mean}'
        self.track_running_stats = track_running_stats

        # 第一层：宽卷积
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(16, track_running_stats=self.track_running_stats),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第二层
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, track_running_stats=self.track_running_stats),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第三层
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64, track_running_stats=self.track_running_stats),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第四层
        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64, track_running_stats=self.track_running_stats),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第五层
        self.layer5 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64, track_running_stats=self.track_running_stats),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 自适应平均池化，将特征长度缩放到 1
        # self.global_pool = nn.AdaptiveAvgPool1d(1)
        # calculate the output size
        feat_dim = input_length // 16 // 2 ** 5 * 64
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 100),  # 使用 LazyLinear 代替 Linear(64, 100)
            nn.BatchNorm1d(100, track_running_stats=self.track_running_stats),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(100, num_classes)
        self._layer_feats = None
        self.squeeze = False
        self.squeeze_level = 10
        self.fuzzy_factor = 0.05

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # He 初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier 初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def transform_fn(self, x):
        # if self.squeeze:
            # x = envelope_signal(x)
            # x = quantize_signal(x, self.squeeze_level)  # works well
            # x = fuzzy_signal(x, self.fuzzy_factor)
            # x = median_filter(x, kernel_size=3)
            # x = apply_filter(x, 'lowpass', 3_000, 12_000, 5)
            # x = wavelet_denoising_tensor(x, wavelet='db4', level=1)  # bad performance
            # x = drop_signal(x, winsize=10, drop_rate=0.1)
        
        if self.transform_in_model:
            x = transform_value(x, zero_mean=self.zero_mean)

        x = x - x.mean(dim=-1, keepdim=True)
        x = torch.clamp(x, *self.clamp_range)
        return x

    def unfreeze_dropout(self):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # 启用 Dropout

    def _encode(self, x):
        x = self.transform_fn(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        self._layer_feats = [out1, out2, out3, out4, out5]
        out = out5.view(out5.size(0), -1)
        return self.fc(out)

    def features(self, x):
        return self._encode(x)

    def classify_features(self, features):
        return self.classifier(features)

    def forward(self, x, return_feats=False):
        out = self._encode(x)
        logits = self.classifier(out)

        if return_feats:
            return logits, self._layer_feats

        return logits

    def predict(self, x, return_feats=False, use_dropout=False):
        self.eval()
        if use_dropout:
            self.unfreeze_dropout()
        return self.forward(x, return_feats)
