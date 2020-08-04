import numpy as np
import torch
from torch.utils.data import TensorDataset
from batchgenerators.utilities.file_and_folder_operations import *

from scipy.io import loadmat
import numpy as np
import tqdm

def _zscore(x):
    assert x.ndim ==2
    std = x.std(1)
    mean = x.mean(1)
    x_norm = (x-mean[:,None])/std[:,None]

    return x_norm

def load_data():
    path = '/workspace/Dy/dataset/PSD'

    files = subfiles(path, join=True,suffix='.mat')

    train_data = []
    val_data = []
    print('loading data')

    for file in files:
        if file.find('(1)')!= -1 or  file.find('(2)')!=-1:
            eeg_data = loadmat(file)
            eeg_data = eeg_data['Pxx']
            train_data.append(eeg_data.reshape(-1,40))

        if file.find('(3)')!= -1:
            eeg_data = loadmat(file)
            eeg_data = eeg_data['Pxx']
            val_data.append(eeg_data.reshape(-1,40))

    train_data = np.array(train_data)
    val_data = np.array(val_data)
    # print(train_data.shape)

    train_data = _zscore(train_data.reshape(-1,40))
    val_data = _zscore(val_data.reshape(-1,40))

    train_data = torch.Tensor(train_data)
    val_data = torch.Tensor(val_data)

    train_label = np.array([*range(1, 11)])
    train_label = train_label[None, :].T.repeat(16200, axis=1)
    train_label = train_label.reshape(-1,1)
    train_label = torch.Tensor(train_label)

    # print(train_data.shape, train_label.shape)
    val_label = np.array([*range(1, 11)])
    val_label = val_label[None, :].T.repeat(8100, axis=1)
    val_label = val_label.reshape(-1,1)
    val_label = torch.Tensor(val_label)


    train_dataset = TensorDataset(train_data, train_label)
    val_dataset = TensorDataset(val_data, val_label)

    print('done')

    return train_dataset, val_dataset



if __name__ == '__main__':
    dstTr, dstV = load_data()

