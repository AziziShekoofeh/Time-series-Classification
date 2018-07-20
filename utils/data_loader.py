import numpy as np
from utils import settings as S
import scipy.io as spio
from numpy.random import permutation
from sklearn.model_selection import train_test_split
import h5py


class DataLoader:
    def __init__(self, sl, validation_split=0.2, downsample=False, downsample_rate=2):
        self.sl = sl
        self.data_dir = S.intermediate_folder
        self.validation_split = validation_split
        self.downsample = downsample
        self.downsample_rate = downsample_rate

    @staticmethod
    def reshaper(data):
        # reshape input to be [samples, time steps, features]
        reshaped_data = np.reshape(data, (data.shape[0], data.shape[1], 1))
        return reshaped_data

    def load_train_test(self, bmode=False):
        d_sliding = spio.loadmat(self.data_dir + '/' + 'D_Sliding.mat')
        d_fixed = spio.loadmat(self.data_dir + '/' + 'D_Fixed.mat')

        # Matrix reformatting to access the cells
        d_train = d_sliding['Ds_train'][0:, 0:self.sl]  # B-mode+RF Train
        l_train = d_sliding['Ls_train'][0:, 0]  # B-mode+RF Train Labels
        d_test = d_fixed['Df_test'][0:, 0:self.sl]  # B-mode+RF Test
        l_test = d_fixed['Lf_test'][0:, 0]  # b-mode+RF Test Labels

        # Select the first half including only Bmode data [Bmode Data, RF Data]
        d_train_r = d_train[d_train.shape[0] / 2:, 0:self.sl]  # RF Train
        l_train_r = l_train[l_train.shape[0] / 2:, ]  # RF Train Labels
        d_test_r = d_test[d_test.shape[0] / 2:, 0:self.sl]  # RF Test
        l_test_r = l_test[l_test.shape[0] / 2:, ]  # RF Test Labels
        train_seq = d_train_r
        test_seq = d_test_r
        train_label = l_train_r
        test_label = l_test_r

        if bmode:
            d_train_b = d_train[0:d_train.shape[0] / 2, 0:self.sl]  # B-mode Train
            l_train_b = l_train[0:l_train.shape[0] / 2, ]  # B-mode Train Labels
            d_test_b = d_test[0:d_test.shape[0] / 2, 0:self.sl]  # B-mode Test
            l_test_b = l_test[0:l_test.shape[0] / 2, ]  # B-mode Test Labels
            train_seq = d_train_b
            test_seq = d_test_b
            train_label = l_train_b
            test_label = l_test_b

        if self.downsample:
            idx = np.floor(np.linspace(start=0, stop=self.sl-1, num=(self.sl/self.downsample_rate)))
            train_seq = train_seq[:, idx.astype(int)]
            test_seq = test_seq[:, idx.astype(int)]

        train_seq = self.reshaper(train_seq)
        test_seq = self.reshaper(test_seq)

        return train_seq, train_label, test_seq, test_label

    def load_data(self, bmode=False):
        train_data, train_label, test_seq, test_label = self.load_train_test(bmode)
        train_seq, train_label, validation_seq, validation_label = self.split_data(train_data, train_label)
        return train_seq, train_label, validation_seq, validation_label, test_seq, test_label

    def split_data(self, train_data, train_label):

        perm_idx = permutation(train_data.shape[0])
        train_data_perm = train_data[perm_idx, :]
        train_label_perm = train_label[perm_idx, ]

        validation_idx = int(round(self.validation_split * train_data.shape[0],0))

        train_seq = train_data_perm[validation_idx:, :]
        train_label = train_label_perm[validation_idx:, ]
        validation_seq = train_data_perm[0:validation_idx, :]
        validation_label = train_label_perm[0:validation_idx, ]

        return train_seq, train_label, validation_seq, validation_label

    def load_test(self, bmode=False):
        _, _, test_seq, test_label = self.load_train_test(bmode)
        return test_seq, test_label

    def load_data_split(self, bmode=False):   # shuffle and do the division based on the split size
        train_data, train_label, test_seq, test_label = self.load_train_test(bmode)
        data_seq = np.concatenate([train_data, test_seq])
        data_label = np.concatenate([train_label, test_label])
        train_data, test_seq, train_label, test_label = train_test_split(data_seq, data_label, test_size=0.2,
                                                                         random_state=40)
        if bmode:
            train_data = np.concatenate([train_data, test_seq])
            train_label = np.concatenate([train_label, test_label])

        train_seq, train_label, validation_seq, validation_label = self.split_data(train_data, train_label)
        return train_seq, train_label, validation_seq, validation_label, test_seq, test_label

    def load_whole_test(self, bmode=False):
        if bmode:
            d_test = h5py.File(self.data_dir + '/' + 'D_Whole_Bmode.mat')
            d_test = d_test['Dw_bmode'].value
        else:
            d_test = h5py.File(self.data_dir + '/' + 'D_Whole_RF.mat')
            d_test = d_test['Dw_rf'].value

        test_seq = d_test[0:, 0:self.sl]
        test_seq = self.reshaper(test_seq)

        return test_seq

