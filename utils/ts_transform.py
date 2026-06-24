import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_wavelets import DWT1D, IDWT1D
# from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image
from sklearn.preprocessing import normalize, maxabs_scale, minmax_scale
from scipy.signal import butter, lfilter, filtfilt
from scipy import signal
import pywt


def transform_array(x, transform_type):
    if x.ndim == 1:
        trans_dim = 0
    elif x.ndim == 2:
        trans_dim = 1
    else:
        raise ValueError(f"Invalid input dimension: {x.ndim}")

    if transform_type == 'normalize':
        x = normalize(x, axis=trans_dim)
    elif transform_type == 'maxabs':
        x = maxabs_scale(x, axis=trans_dim)
    elif transform_type == 'minmax':
        x = minmax_scale(x, axis=trans_dim, feature_range=(-1, 1))
    elif transform_type == 'zscore':
        x = (x - x.mean(axis=trans_dim, keepdims=True)) / x.std(axis=trans_dim, keepdims=True)
    else:
        raise ValueError(f"Invalid transform type: {transform_type}")
    
    return x


def transform_tensor(x, transform_type):
    if x.dim() <= 3:
        trans_dim = x.dim() - 1
    else:
        raise ValueError(f"无效的输入维度: {x.dim()}")

    if transform_type == 'normalize':
        x = x / torch.norm(x, dim=trans_dim, keepdim=True, p=2)
    elif transform_type == 'maxabs':
        x = x / torch.max(torch.abs(x), dim=trans_dim, keepdim=True)[0]
    elif transform_type == 'minmax':
        x_min = torch.min(x, dim=trans_dim, keepdim=True)[0]
        x_max = torch.max(x, dim=trans_dim, keepdim=True)[0]
        x = 2 * (x - x_min) / (x_max - x_min) - 1
    elif transform_type == 'zscore':
        x = (x - torch.mean(x, dim=trans_dim, keepdim=True)) / torch.std(x, dim=trans_dim, keepdim=True)
    else:
        raise ValueError(f"无效的转换类型: {transform_type}")
    
    return x


def transform_value(x, transform_type='maxabs', zero_mean=True):
    # 如果信号不首先进行去均值，那么及时面对微弱的直流信号攻击，也能产生显著的攻击效果。
    if isinstance(x, np.ndarray):
        if zero_mean:
            x = x - x.mean(-1, keepdims=True)
        return transform_array(x, transform_type)
    elif isinstance(x, torch.Tensor):
        if zero_mean:
            x = x - x.mean(-1, keepdim=True)
        return transform_tensor(x, transform_type)
    else:
        raise TypeError(f"输入类型无效: {type(x)}。应为 numpy.ndarray 或 torch.Tensor。")


# def transform_denoise(x, filter_type='lowpass', **kwargs):
#     if filter_type == 'lowpass':
#         x = lowpass_filter(x, **kwargs)
#     elif filter_type == 'highpass':
#         x = highpass_filter(x, **kwargs)
#     elif filter_type == 'bandpass':
#         x = bandpass_filter(x, **kwargs)
#     elif filter_type == 'median':
#         x = median_filter(x, **kwargs)
#     elif filter_type == 'wavelet':
#         x = wavelet_denoising(x, **kwargs)
#     else:
#         raise ValueError(f"无效的滤波器类型: {filter_type}")
#     return x


def design_filter(filter_type, cutoff_freq, sample_rate, order=5):
    nyquist = 0.5 * sample_rate  # 奈奎斯特频率
    normalized_cutoff = np.array(cutoff_freq) / nyquist  # 归一化截止频率

    if filter_type == 'lowpass':
        b, a = butter(order, normalized_cutoff, btype='low')
    elif filter_type == 'highpass':
        b, a = butter(order, normalized_cutoff, btype='high')
    elif filter_type == 'bandpass':
        b, a = butter(order, normalized_cutoff, btype='band')
    else:
        raise ValueError("Filter type must be 'lowpass', 'highpass', or 'bandpass'")

    return b, a


# 定义滤波器应用函数
def apply_filter(x, filter_type, cutoff_freq, sample_rate, order=2):
    flag = False
    if not isinstance(x, np.ndarray):
        flag = True
        device = x.device
        x = x.detach().cpu().numpy()

    b, a = design_filter(filter_type, cutoff_freq, sample_rate, order)
    filtered_signal = filtfilt(b, a, x)
    if flag:
        filtered_signal = torch.from_numpy(filtered_signal.copy()).float().to(device)

    return filtered_signal


def median_filter(x, kernel_size=3):
    return signal.medfilt(x, kernel_size=kernel_size)

def wavelet_denoising(x, wavelet='db4', level=1, mode='soft'):
    coeff = pywt.wavedec(x, wavelet, level=level)
    sigma = (1/0.6745) * np.median(np.abs(coeff[-level]))
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = [pywt.threshold(i, value=uthresh, mode=mode) for i in coeff[1:]]
    return pywt.waverec(coeff, wavelet)


def wavelet_denoising_tensor(x, wavelet='db4', level=1, mode='soft'):
    # 确保输入是三维张量 (batch_size, channels, signal_length)
    original_shape = x.shape
    original_dtype = x.dtype
    x_dim = x.dim()
    if x_dim == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x_dim == 2:
        x = x.unsqueeze(1)
    elif x_dim > 3:
        raise ValueError(f"输入张量维度应为1、2或3,但得到了{x_dim}维")

    # 初始化DWT和IDWT
    dwt = DWT1D(wave=wavelet, J=level, mode='reflect').to(x.device)  # 'zero', 'symmetric', 'periodic', 'reflect'
    idwt = IDWT1D(wave=wavelet, mode='reflect').to(x.device)

    # 执行DWT
    yl, yh = dwt(x)

    # 计算阈值
    sigma = torch.median(torch.abs(yh[-1])) / 0.6745
    uthresh = sigma * torch.sqrt(2 * torch.log(torch.tensor(x.shape[-1])))

    # 对系数进行阈值处理
    if mode == 'soft':
        yh = [torch.sign(c) * torch.maximum(torch.abs(c) - uthresh, torch.zeros_like(c)) for c in yh]
    elif mode == 'hard':
        yh = [c * (torch.abs(c) > uthresh) for c in yh]
    else:
        raise ValueError(f"无效的阈值模式: {mode}")

    # 执行IDWT
    denoised = idwt((yl, yh))

    # 返回与输入相同维度的张量，并转换回原始数据类型
    return denoised.view(original_shape).to(original_dtype)


class WaveletDenoising:
    def __init__(self, wavelet='db4', mode='soft', level=1):
        """
        :param wavelet: 使用的小波类型
        :param mode: 'soft' or 'hard'，用于阈值化方法选择
        :param level: 小波分解的级别
        """
        self.dwt = DWT1D(wave=wavelet, J=level)
        self.idwt = IDWT1D(wave=wavelet)
        self.mode = mode
        self.level = level

    def calculate_threshold(self, detail_coeffs):
        """
        基于分解的细节系数自适应计算阈值。
        常用VisuShrink方法，根据细节系数的中位绝对偏差（MAD）计算标准差，再计算阈值。
        
        :param detail_coeffs: 小波分解后的细节系数列表
        :return: 对应的自适应阈值
        """
        
        # 将所有细节系数连接成一个张量
        all_coeffs = torch.cat([coeff.flatten() for coeff in detail_coeffs])
        
        median = torch.median(torch.abs(all_coeffs))
        sigma = median / 0.6745  # 基于中位绝对偏差的估计
        threshold = sigma * torch.sqrt(2 * torch.log(torch.tensor(all_coeffs.numel(), dtype=all_coeffs.dtype, device=all_coeffs.device)))
        return threshold

    def thresholding(self, x, thresh):
        """对输入张量 x 应用软阈值或硬阈值"""
        if self.mode == 'soft':
            return torch.sign(x) * torch.maximum(torch.abs(x) - thresh, torch.tensor(0.0))
        elif self.mode == 'hard':
            return x * (torch.abs(x) > thresh).float()

    def denoise(self, signal):
        """
        对输入的时间序列信号进行小波降噪。
        
        :param signal: 输入信号, torch.Tensor, 可以是任意维度
        :return: 降噪后的信号，与输入信号具有相同的形状
        """
        original_shape = signal.shape
        original_dtype = signal.dtype

        # 将信号重塑为 (batch_size, channels, length) 的形式
        if signal.dim() == 1:
            signal = signal.unsqueeze(0).unsqueeze(0)
        elif signal.dim() == 2:
            signal = signal.unsqueeze(1)
        elif signal.dim() > 3:
            signal = signal.view(-1, signal.shape[-2], signal.shape[-1])

        # 小波分解
        cA, cD = self.dwt(signal)
        cD_denoised = []
        for detail in cD:
            threshold = self.calculate_threshold(detail)
            cD_denoised.append(self.thresholding(detail, threshold))
        
        # 使用阈值后的系数进行小波重构
        denoised_signal = self.idwt((cA, cD_denoised))
        
        # 将信号重塑为原始形状
        denoised_signal = denoised_signal.view(original_shape)

        return denoised_signal.to(original_dtype)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils.fft_utils import fft_1d
    
    _path = r"F:\Datasets\CWdata_12k\npy_data\cwru_len[2048]_ratio[0.6]_ol_train[0.6]_num_train[200]_ol_test[0]_num_test[100].npy"
    data_dict = np.load(_path, allow_pickle=True).item()
    xx = data_dict['samples_train']
    print(xx.shape)
    # _x = xx[3, 0]
    _x = xx[0, 0]

    cutoff_freq = 1500  # 低通的截止频率
    sample_rate = 12_000
    filtered_signal_lowpass = apply_filter(_x, 'lowpass', cutoff_freq, sample_rate, order=5)

    # time domain
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(_x, label='original')
    plt.plot(filtered_signal_lowpass, label='filtered')
    plt.legend()
    plt.title('Time Domain')
    plt.grid()

    plt.subplot(2, 1, 2)
    fft_values, freqs = fft_1d(_x, sample_rate)
    fft_values_filtered, _ = fft_1d(filtered_signal_lowpass, sample_rate)
    plt.plot(freqs, fft_values, label='original')
    plt.plot(freqs, fft_values_filtered, label='filtered')
    plt.legend()
    plt.title('Frequency Domain')
    plt.grid()
    plt.show()



