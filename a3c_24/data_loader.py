import os
from typing import Optional
import glob

import torch
from torch.utils.data import Dataset
from scipy import io


class UWADataset(Dataset):
    __version__ = '24.0'  # Dataset from '24 research. Do not change 24.

    def __init__(self, snr: Optional[int] = 10,
                 mode: str = 'train',
                 ratio: float = 0.8,
                 verbose: bool = False,
                 data_dir: str = 'data24_v21'):
        """
        Returns the UWA dataset.
        Applying DataLoader to this dataset will return the data from a single environment. Make sure to slice the data
        according to the sensing model size before feeding it to the model.
        Arguments:
            snr: int or None. The signal to noise ratio. If None, all data will be loaded.
            mode: str. The mode to load the dataset. 'train', 'test' or 'all'.
            ratio: float. The ratio of the train dataset. If mode is 'test', load (1-ratio) of the data.
            verbose: bool. Whether to print the information.
            data_dir: str. The directory of the data.
        """
        self.snr = snr
        if self.snr is not None:
            self.data_dir = os.path.join(os.getcwd(), '../UWA_MATLAB', data_dir, str(self.snr))  # pwd: a3c_24
        else:
            self.data_dir = os.path.join(os.getcwd(), '../UWA_MATLAB', data_dir)
        # if not os.path.isdir(self.data_dir):
        #     raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if verbose:
            print(f"Loading data from {self.data_dir}")

        data = []
        paths = glob.glob(os.path.join(self.data_dir, '**', 'data_*.mat'), recursive=True)
        files_list = [path for path in paths if os.path.isfile(path)]

        # File check.
        """
        for file in files_list:
            if io.loadmat(file)['freq_data'].shape[1] != 110:
                print(f"Data shape error: {io.loadmat(file)['freq_data'].shape}, {file}")
                return
        """

        [data.append(io.loadmat(file)) for file in files_list]
        # Check if the data shape is correct.
        # print("Checking data shape...")
        # [print(io.loadmat(file)['freq_data'].shape, file) for file in files_list]

        # Each data consists of sub-batches of data. The train/test data size is rather discrete, as the sub-batches
        # aren't divided.
        # Concatenate data.
        self.sensing_data = torch.from_numpy(data[0]['freq_data'])
        self.cqi_data = torch.from_numpy(data[0]['CQI_data'])
        self.label_data = torch.from_numpy(data[0]['label_data'])
        self.rate_data = torch.from_numpy(data[0]['rate_data'])
        for d in data[1:]:
            try:
                self.sensing_data = torch.cat((self.sensing_data, torch.from_numpy(d['freq_data'])), dim=0)
                self.cqi_data = torch.cat((self.cqi_data, torch.from_numpy(d['CQI_data'])), dim=0)
                self.label_data = torch.cat((self.label_data, torch.from_numpy(d['label_data'])), dim=0)
                self.rate_data = torch.cat((self.rate_data, torch.from_numpy(d['rate_data'])), dim=0)
            except RuntimeError as e:
                print(f"{e}")
                [print(f"{k}: {d[k].shape}") for k in d.keys() if not k.startswith('__')]

        if mode == 'train':
            self.sensing_data = self.sensing_data[:int(ratio * len(self.sensing_data))]
            self.cqi_data = self.cqi_data[:int(ratio * len(self.cqi_data))]
            self.label_data = self.label_data[:int(ratio * len(self.label_data))]
            self.rate_data = self.rate_data[:int(ratio * len(self.rate_data))]
        elif mode == 'test':
            self.sensing_data = self.sensing_data[int(ratio * len(self.sensing_data)):]
            self.cqi_data = self.cqi_data[int(ratio * len(self.cqi_data)):]
            self.label_data = self.label_data[int(ratio * len(self.label_data)):]
            self.rate_data = self.rate_data[int(ratio * len(self.rate_data)):]
        elif mode == 'all':
            pass
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if verbose:
            print(f"Data loaded.\n"
                  f"CQI data shape:     {self.cqi_data.shape}\n"
                  f"Sensing data shape: {self.sensing_data.shape}\n"
                  f"Rate data shape:    {self.rate_data.shape}\n"
                  f"Label data shape:   {self.label_data.shape}\n")

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        cqi_slice = self.cqi_data[idx].squeeze().to(torch.float32)
        sensing_slice = self.sensing_data[idx].squeeze().to(torch.float32)
        label_slice = self.label_data[idx].squeeze()
        rate_slice = self.rate_data[idx].squeeze().to(torch.float32)

        return (cqi_slice, sensing_slice), (rate_slice, label_slice)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    snr_list = (-4, -2, 0, 2, 4, 6, 8, 10, 12)
    snr_list = (10,)
    for snr in snr_list:
        uwadataset = UWADataset(mode='all',
                                # ratio=0.1,
                                snr=snr,
                                data_dir='data24_v18',
                                verbose=True)
    print(f"Testing UWADataset...")
    data_loader = DataLoader(uwadataset, batch_size=1, shuffle=True)
    print(f"Length of data loader: {len(data_loader)}\n"
          f"Length of dataset:     {len(uwadataset)}")

    num_windows = 8  # use cfg.num_windows in actual code.
    print(f"If the number of impulses is {num_windows}, the corresponding data sizes are:")
    for i, (x, y) in enumerate(data_loader):
        print(f"Batch {i}:")
        # Here, from x[0]["0", start_idx : start_idx + num_windows] , "0" is the batch index which is fixed to 0 since
        # the batch size is always 1.
        cqi_data = x[0][0]
        sensing_data = x[1][0]
        rate_data = y[0][0]
        rb_data = y[1][0]
        print(f"    CQI data:                {cqi_data[0:num_windows].shape, cqi_data[0:num_windows].dtype}\n"
              f"    Sensing data:            {sensing_data[0:num_windows].shape, sensing_data[0:num_windows].dtype}\n"
              f"    Corresponding rate data: {rate_data[num_windows].shape, rate_data[num_windows].dtype}\n"
              f"    Corresponding RB data:   {rate_data[num_windows].shape, rate_data[num_windows].dtype}\n")
        if i == 0:
            print(f"\nData size of single time slot:\n"
                  f"    CQI data:     {cqi_data[0].shape, cqi_data[0].dtype}\n"
                  f"    Sensing data: {sensing_data[0].shape, sensing_data[0].dtype}\n"
                  f"    Rate data:    {rate_data[0].shape, rate_data[0].dtype}\n"
                  f"    RB data:      {rb_data[0].shape, rb_data[0].dtype}")
            print("cf. Size: [Number of batches (envs), time length, data size]")
            break

