import numpy as np
import pandas as pd
# from scipy.io import loadmat
# import os


def get_csv_timeseries(data_path, num, header=20, chn=8, for_train=True):
    """
    https://blog.csdn.net/weixin_44122191/article/details/110098481
    :param for_train:
    :param data_path:
    :param num:
    :param header:
    :param chn:
    :return:
    """
    data_path = data_path if isinstance(data_path, list) else [data_path]
    data_ = []
    for p in data_path:
        # chunks = pd.read_csv(p, delim_whitespace=True, header=header, chunksize=1024, iterator=True)
        chunks = pd.read_csv(p, sep="\s+|,", header=header, chunksize=1024, iterator=True, engine='python')
        if for_train:
            data = chunks.get_chunk(num)
            data_.append(data.values[:num, :chn])  # (N_points, N_chn)
        else:
            data = chunks.get_chunk(num+num)
            data_.append(data.values[num:, :chn])  # (N_points, N_chn)
    data_ = np.concatenate(data_, axis=0)

    return data_


def get_csv_timeseriesV2(data_path, num_train=20, num_test=200, header=20, chn=8, for_train=True):
    """
    https://blog.csdn.net/weixin_44122191/article/details/110098481
    """
    data_path = data_path if isinstance(data_path, list) else [data_path]
    data_ = []
    for p in data_path:
        # chunks = pd.read_csv(p, delim_whitespace=True, header=header, chunksize=1024, iterator=True)
        chunks = pd.read_csv(p, sep="\s+|,", header=header, chunksize=1024, iterator=True, engine='python')
        if for_train:
            data = chunks.get_chunk(num_train)
            data_.append(data.values[:num_train, :chn])  # (N_points, N_chn)
        else:
            data = chunks.get_chunk(num_train+num_test)
            data_.append(data.values[num_train:, :chn])  # (N_points, N_chn)
    data_ = np.concatenate(data_, axis=0)

    return data_


def data_split(data, length, sample_n, win_size: int = 500, downsampling=False):
    """
    :param downsampling:
    :param sample_n:
    :param data: (num, (, (nc)))
    :param length: sample length
    :param win_size: sliding window size, (len(data)-length)//win_size+1 >= sample_n
    :return:(sample_n, length)
    """
    if downsampling:
        ids = np.arange(len(data)//2)*2  # 0, 2, 4, 6, ...
        data = data[ids]

    if win_size == 0 and int(length * sample_n) <= len(data):
        ret_data = np.reshape(data[:length * sample_n], [sample_n, length, data.shape[-1]])
    else:
        # print(f"Overlapped samples, with win_size={win_size}")
        assert cal_sample_num(len(data), length, win_size) >= sample_n

        ret_data = []
        start = 0
        for i in range(sample_n):
            ret_data.append(data[start:start + length])
            start += win_size
        ret_data = np.asarray(ret_data, dtype=np.float32)
    return ret_data


def cal_sample_num(data_length, sample_length, win_size):
    n = (data_length - sample_length) // win_size + 1  # 向下取整, floor
    return n


if __name__ == "__main__":
    # pt = r"F:\dataset\东南大学齿轮箱数据集\gearbox\bearingset\ball_20_0.csv"
    pt = r"F:\dataset\东南大学齿轮箱数据集\gearbox\gearset\Chipped_20_0.csv"
    data = pd.read_csv(pt, sep="\s+|,", header=10, engine='python')[:10]
    # print(data.shape)
    print(data[:2])
    exit()

    pt = r"F:\dataset\东南大学齿轮箱数据集\gearbox\bearingset\ball_30_2.csv"
    data = pd.read_csv(pt, sep="\s+|,", header=20, engine='python')  # [:10]
    print(data.shape)
    # print(data[:2])

    # get_csv_timeseries(pt, num=1024*2)
