import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # 将日期列和目标变量列移到数据框的前后，方便后续处理
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # 设置数据集的划分比例
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        # 定义当前数据集的范围
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]

        border2 = border2s[self.set_type]
        # print('border1 : ',border1)
        # print('border2 : ',border2)
        # 根据 features 参数选择数据集中的特征列
        if self.features == 'M' or self.features == 'MS':
            # 如果使用多变量特征（'M' 或 'MS'），则选择所有非目标变量的列；
            cols_data = df_raw.columns[1:]
            # 排除lat 和 lon 这两列
            cols_data = cols_data.difference(['location'])

            df_data = df_raw[cols_data]
        elif self.features == 'S':
            # 如果使用单变量特征（'S'），则仅选择目标变量列
            df_data = df_raw[[self.target]]

        # 判断是否对数据进行标准化
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # 提取时间特征
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            # 如果 timeenc 为 0，则手动提取时间特征（如月份、日期、星期几、小时等）
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            # 使用 time_features 函数根据给定频率提取时间特征
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # 假设 location 列已经被处理为从 0 到 182 的整数值，表示不同的站点
        self.location_data = df_raw['location'].astype(int).values
        self.location_data = torch.tensor(self.location_data, dtype=torch.long)  # 转换为 PyTorch 张量

        self.data_x = data[border1:border2]
        # print('data_x : ', self.data_x)
        self.data_y = data[border1:border2]
        # print('data_y : ', self.data_y)
        self.data_stamp = data_stamp
        # print('data_stamp : ', data_stamp)

    # 获取数据
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # 提取 location 特征，并转换为 one-hot 编码
        location_indices = self.location_data[s_begin:s_end]  # 获取当前序列的站点编号
        seq_location = torch.zeros((len(location_indices), 182))  # 创建一个全零的张量
        # seq_location[torch.arange(len(location_indices)), location_indices] = 1  # 设置对应位置为1
        #print('seq_location : ', seq_location)

        seq_x = self.data_x[s_begin:s_end]
        #print('seq_x : ', seq_x)
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_location

    # 获取数据集长度
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    # 反向标准化方法
    def inverse_transform(self, data):
        # print(f'data : {data.shape}')
        return self.scaler.inverse_transform(data)


