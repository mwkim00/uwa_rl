import torch
import torch.nn as nn

from utils import normalized_columns_initializer, weights_init


class ActorCritic_v0(nn.Module):
    """
    Actor Critic model.
    A processing unit preprocesses the sensing data into a vector of shape (num_windows, 1, 5). This is concatenated
    with the CQI data and is fed to the main unit of the model.
    sensing module --> reshape & concatenate  --> main module --> critic & actor
    """

    def __init__(self, cfg):
        super(ActorCritic_v0, self).__init__()
        self.version = cfg.ac_version

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.2)
        self.sensing_module_v0 = nn.Sequential(
            # See size: nn.Conv3d(8, 8, 2, 1, 1)(torch.rand((8, 2, 13, 10))).shape = torch.Size([8, 3, 14, 11])
            # Original size: torch.Size([8, 2, 13, 10])
            nn.Conv3d(in_channels=cfg.num_windows, out_channels=cfg.num_windows, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),  # torch.Size([8, 3, 14, 11])
            nn.Conv3d(in_channels=cfg.num_windows, out_channels=cfg.num_windows, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),  # torch.Size([8, 4, 15, 12])
            nn.Conv3d(in_channels=cfg.num_windows, out_channels=cfg.num_windows, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # torch.Size([8, 4, 15, 12])
            nn.Conv3d(in_channels=cfg.num_windows, out_channels=cfg.num_windows, kernel_size=3, stride=1,
                      padding=(0, 1, 1)),
            nn.ReLU(),  # torch.Size([8, 2, 15, 12])
        )
        self.sensing_module_v1 = nn.Sequential(
            # See size: nn.Conv3d(8, 8, 2, 1, 1)(torch.rand((8, 2, 13, 10))).shape = torch.Size([8, 3, 14, 11])
            # Original size: torch.Size([8, 2, 13, 10])
            nn.Conv3d(in_channels=cfg.num_windows, out_channels=cfg.num_windows, kernel_size=2, stride=1, padding=1),
            nn.ELU(),  # torch.Size([8, 3, 14, 11])
            nn.Dropout(),
            nn.Conv3d(in_channels=cfg.num_windows, out_channels=cfg.num_windows, kernel_size=2, stride=1, padding=1),
            nn.ELU(),  # torch.Size([8, 4, 15, 12])
            nn.Conv3d(in_channels=cfg.num_windows, out_channels=cfg.num_windows, kernel_size=3, stride=1, padding=1),
            nn.ELU(),  # torch.Size([8, 4, 15, 12])
            nn.Dropout(),
            nn.Conv3d(in_channels=cfg.num_windows, out_channels=cfg.num_windows, kernel_size=3, stride=1, padding=1),
            nn.ELU(),  # torch.Size([8, 4, 15, 12])
            nn.Conv3d(in_channels=cfg.num_windows, out_channels=cfg.num_windows, kernel_size=3, stride=1,
                      padding=(0, 1, 1)),
            nn.ELU(),  # torch.Size([8, 2, 15, 12])
        )

        self.linear_sensing_v0 = nn.Sequential(
            nn.Linear(2 * 15 * 12, cfg.nrb),
        )
        self.linear_sensing_v1 = nn.Sequential(
            nn.Linear(2 * 15 * 12, 2 * cfg.nrb),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(2 * cfg.nrb, cfg.nrb),
            nn.ELU(),
            nn.Linear(cfg.nrb, cfg.nrb),
            nn.ELU(),
        )

        self.main_module_v0 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.num_windows, out_channels=cfg.num_windows, kernel_size=(3, 2), stride=1,
                      padding=(1, 0)),
            nn.ReLU(),  # torch.Size([8, 5, 1])
            nn.Flatten(),
            nn.Linear(5, cfg.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size // 2, cfg.hidden_size)
        )
        self.main_module_v1 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.num_windows, out_channels=cfg.num_windows, kernel_size=(3, 2), stride=1,
                      padding=(1, 0)),
            nn.ELU(),  # torch.Size([8, 5, 1])
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(5, cfg.hidden_size // 2),
            nn.ELU(),
            nn.Linear(cfg.hidden_size // 2, cfg.hidden_size // 2),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(cfg.hidden_size // 2, cfg.hidden_size),
            nn.ELU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ELU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )
        self.lstm = nn.LSTM(input_size=cfg.hidden_size, hidden_size=cfg.hidden_size, num_layers=cfg.num_lstm_layers,
                            batch_first=True)

        self.critic_linear = nn.Linear(cfg.hidden_size, 1)
        self.actor_linear = nn.Linear(cfg.hidden_size, cfg.action_space)

        # Initialize.
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        for i in range(cfg.num_lstm_layers):
            self.lstm.bias_ih_l0.data.fill_(0)
            self.lstm.bias_hh_l0.data.fill_(0)

        self.train()

    def forward(self, inputs):
        if self.version == 0:
            (cqi, sensing), (hx, cx) = inputs
            x_sensing = self.sensing_module_v0(sensing)
            x_sensing = self.linear_sensing_v0(x_sensing.view(-1, 2 * 15 * 12))
            x = torch.cat((x_sensing.unsqueeze(2), cqi.unsqueeze(2)), dim=2)
            x = self.main_module_v0(x)
            x = x.unsqueeze(0)  # torch.Size([1, 8, hidden size]) = (batch, num_windows, hidden_size)
            x, (hx, cx) = self.lstm(x, (hx, cx))
            # x = x.squeeze()

            # Previous codes.
            ###################################################################
            """
            x = F.elu(self.conv1(inputs))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))
            
            x = x.view(-1, 32 * 3 * 3)
            hx, cx = self.lstm(x, (hx, cx))
            x = hx
            """

            # Take only the last time step output.
            # Could squeeze dim 0 (batch dimension) as below, but leave space for batch.
            # return self.critic_linear(x[:, -1, :]).squeeze(), self.actor_linear(x[:, -1, :]).squeeze(), (hx, cx)
            return self.critic_linear(x[:, -1, :]), self.actor_linear(x[:, -1, :]), (hx, cx)

        elif self.version == 1:
            (cqi, sensing), (hx, cx) = inputs
            x_sensing = self.sensing_module_v1(sensing)
            x_sensing = self.linear_sensing_v1(x_sensing.view(-1, 2 * 15 * 12))

            x = torch.cat((x_sensing.unsqueeze(2), cqi.unsqueeze(2)), dim=2)
            x = self.main_module_v1(x)
            x = x.unsqueeze(0)  # torch.Size([1, 8, hidden size]) = (batch, num_windows, hidden_size)

            x, (hx, cx) = self.lstm(x, (hx, cx))
            return self.critic_linear(x[:, -1, :]), self.actor_linear(x[:, -1, :]), (hx, cx)
        else:
            raise ValueError(f"Invalid version {self.version}.")


class Convolutional_v0(nn.Module):
    def __init__(self, cfg):
        super(Convolutional_v0, self).__init__()
        self.version = cfg.cnn_version
        self.sensing_module_v0 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),  # torch.Size([8, 3, 14, 11])
            nn.BatchNorm3d(1),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),  # torch.Size([8, 4, 15, 12])
            nn.BatchNorm3d(1),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # torch.Size([8, 4, 15, 12])
            nn.BatchNorm3d(1),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=(0, 1, 1)),
            nn.ReLU(),  # torch.Size([8, 2, 15, 12])
            nn.BatchNorm3d(1),
        )

        self.sensing_module_v1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),  # torch.Size([110, 1, 3, 14, 11])
            nn.BatchNorm3d(1),
            nn.Conv3d(in_channels=1, out_channels=2, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),  # torch.Size([110, 2, 4, 15, 12])
            nn.Conv3d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # torch.Size([110, 2, 4, 15, 12])
            nn.Conv3d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # torch.Size([110, 2, 4, 15, 12])
            nn.Conv3d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # torch.Size([110, 2, 4, 15, 12])
            nn.BatchNorm3d(2),
            nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=(0, 1, 1)),
            nn.ReLU(),  # torch.Size([110, 1, 2, 15, 12])
        )

        self.linear_sensing_v0 = nn.Linear(2 * 15 * 12, cfg.nrb)

        self.linear_sensing_v1 = nn.Sequential(
            nn.Linear(2 * 15 * 12, cfg.nrb),
            nn.Linear(cfg.nrb, cfg.nrb),
        )

        self.main_module_v0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 2), stride=1, padding=(1, 0)),
            nn.ReLU(),  # torch.Size([8, 5, 1])
            nn.Flatten(),
            nn.Linear(5, 32),
            nn.ELU(),
            nn.BatchNorm1d(num_features=32),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 32),
            nn.ELU(),
            nn.BatchNorm1d(num_features=32),
            nn.Linear(32, 10)
        )

        self.main_module_v1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 2), stride=1, padding=(1, 0)),
            nn.ReLU(),  # torch.Size([8, 5, 1])
            nn.Flatten(),
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, inputs):
        if self.version == 0:
            x_cqi = inputs[0]
            x_sensing = inputs[1]

            if len(x_cqi.shape) == 3:  # Batched data
                x_cqi = x_cqi.flatten(0, 1)

            if len(x_sensing.shape) == 5:  # Batched data. Ex) [batch_size, 110, 2, 13, 10]
                x_sensing = x_sensing.flatten(0, 1).unsqueeze(1)
            elif len(x_sensing.shape) == 4:  # Not batched but with time axis. Ex) [110, 2, 13, 10]
                x_sensing = x_sensing.unsqueeze(1)
            elif len(x_sensing.shape) == 3:  # Not batched with single time slice. Ex) [2, 13, 10]
                x_sensing = x_sensing.unsqueeze(0)
            else:
                raise ValueError(f"Sensing data has wrong shape: {x_sensing.shape}")

            x_sensing = self.sensing_module_v0(x_sensing)
            x_sensing = self.linear_sensing_v0(x_sensing.view(-1, 2 * 15 * 12))
            x = torch.cat((x_sensing.unsqueeze(2), x_cqi.unsqueeze(2)), dim=2)
            x = self.main_module_v0(x.unsqueeze(1))

            return x

        elif self.version == 1:
            x_cqi = inputs[0]
            x_sensing = inputs[1]

            if len(x_cqi.shape) == 3:  # Batched data
                x_cqi = x_cqi.flatten(0, 1)

            if len(x_sensing.shape) == 5:  # Batched data. Ex) [batch_size, 110, 2, 13, 10]
                x_sensing = x_sensing.flatten(0, 1).unsqueeze(1)
            elif len(x_sensing.shape) == 4:  # Not batched but with time axis. Ex) [110, 2, 13, 10]
                x_sensing = x_sensing.unsqueeze(1)
            elif len(x_sensing.shape) == 3:  # Not batched with single time slice. Ex) [2, 13, 10]
                x_sensing = x_sensing.unsqueeze(0)
            else:
                raise ValueError(f"Sensing data has wrong shape: {x_sensing.shape}")

            x_sensing = self.sensing_module_v1(x_sensing)
            x_sensing = self.linear_sensing_v1(x_sensing.view(-1, 2 * 15 * 12))
            x = torch.cat((x_sensing.unsqueeze(2), x_cqi.unsqueeze(2)), dim=2)
            x = self.main_module_v1(x.unsqueeze(1))

            return x

        else:
            raise ValueError(f"Invalid version {self.version}.")


class FullyConnected_v0(nn.Module):
    def __init__(self, cfg):
        super(FullyConnected_v0, self).__init__()
        self.version = cfg.fcn_version

        action_space = cfg.action_space

        self.elu = nn.ELU()

        # CQI data branch
        self.fc1_1 = nn.Linear(action_space // 2, action_space)
        self.bn1_1 = nn.BatchNorm1d(num_features=action_space)
        self.fc1_2 = nn.Linear(action_space, action_space)
        self.bn1_2 = nn.BatchNorm1d(num_features=action_space)
        self.fc1_3 = nn.Linear(action_space, 64)
        self.bn1_3 = nn.BatchNorm1d(num_features=64)
        self.dropout1_1 = nn.Dropout(0.25)

        # Sensing data branch
        self.fc2_1 = nn.Linear(2 * 13 * 10, 256)
        self.bn2_1 = nn.BatchNorm1d(num_features=256)
        self.fc2_2 = nn.Linear(256, 512)
        self.bn2_2 = nn.BatchNorm1d(num_features=512)
        self.fc2_3 = nn.Linear(512, 256)
        self.bn2_3 = nn.BatchNorm1d(num_features=256)
        self.fc2_4 = nn.Linear(256, 128)
        self.bn2_4 = nn.BatchNorm1d(num_features=128)
        self.fc2_5 = nn.Linear(128, 64)
        self.bn2_5 = nn.BatchNorm1d(num_features=64)
        self.dropout2_1 = nn.Dropout(0.25)

        # Main branch
        self.fc3_1 = nn.Linear(128, 64)
        self.bn3_1 = nn.BatchNorm1d(num_features=64)
        self.fc3_2 = nn.Linear(64, 32)
        self.bn3_2 = nn.BatchNorm1d(num_features=32)
        self.fc3_3 = nn.Linear(32, action_space)
        self.bn3_3 = nn.BatchNorm1d(num_features=action_space)
        self.dropout3_1 = nn.Dropout(0.25)

    def forward(self, inputs):
        if self.version == 0:
            x_cqi = inputs[0]
            x_sensing = inputs[1]

            if len(x_cqi.shape) == 3:  # Batched data
                x_cqi = x_cqi.flatten(0, 1)

            if len(x_sensing.shape) == 5:  # Batched data. Ex) [batch_size, 110, 2, 13, 10]
                x_sensing = x_sensing.flatten(0, 1).flatten(1)
            elif len(x_sensing.shape) == 4:  # Not batched but with time axis. Ex) [110, 2, 13, 10]
                x_sensing = x_sensing.flatten(1)
            elif len(x_sensing.shape) == 3:  # Not batched with single time slice. Ex) [2, 13, 10]
                x_sensing = x_sensing.flatten()
            else:
                raise ValueError

            x_cqi = self.fc1_1(x_cqi)
            x_cqi = self.bn1_1(x_cqi)
            x_cqi = self.elu(x_cqi)

            x_cqi = self.fc1_2(x_cqi)
            x_cqi = self.bn1_2(x_cqi)
            x_cqi = self.elu(x_cqi)

            x_cqi = self.fc1_3(x_cqi)
            x_cqi = self.bn1_3(x_cqi)
            x_cqi = self.elu(x_cqi)

            x_cqi = self.dropout1_1(x_cqi)

            """
            if len(x_sensing.shape) == 5:  # Batched data. Ex) [batch_size, 110, 2, 13, 10]
                x_sensing = self.fc2_1(x_sensing.flatten(0, 1).flatten(1))
            elif len(x_sensing.shape) == 4:  # Not batched but with time axis. Ex) [110, 2, 13, 10]
                x_sensing = self.fc2_1(x_sensing.flatten(1))
            elif len(x_sensing.shape) == 3:  # Not batched with single time slice. Ex) [2, 13, 10]
                x_sensing = self.fc2_1(x_sensing.flatten())
            else:
                raise ValueError
            """
            x_sensing = self.fc2_1(x_sensing)
            x_sensing = self.bn2_1(x_sensing)
            x_sensing = self.elu(x_sensing)

            x_sensing = self.fc2_2(x_sensing)
            x_sensing = self.bn2_2(x_sensing)
            x_sensing = self.elu(x_sensing)

            x_sensing = self.fc2_3(x_sensing)
            x_sensing = self.bn2_3(x_sensing)
            x_sensing = self.elu(x_sensing)

            x_sensing = self.fc2_4(x_sensing)
            x_sensing = self.bn2_4(x_sensing)
            x_sensing = self.elu(x_sensing)

            x_sensing = self.fc2_5(x_sensing)
            x_sensing = self.bn2_5(x_sensing)
            x_sensing = self.elu(x_sensing)

            x_sensing = self.dropout2_1(x_sensing)

            x_cat = torch.cat((x_cqi, x_sensing), dim=-1)
            x_cat = self.fc3_1(x_cat)
            x_cat = self.bn3_1(x_cat)
            x_cat = self.elu(x_cat)

            x_cat = self.dropout3_1(x_cat)

            x_cat = self.fc3_2(x_cat)
            x_cat = self.bn3_2(x_cat)
            x_cat = self.fc3_3(x_cat)
            output = self.bn3_3(x_cat)

            return output

        elif self.version == 1:
            pass

        else:
            raise ValueError(f"Invalid version {self.version}.")


class RandomCQI(nn.Module):
    """
    Comparison scheme which selects RB randomly and determines rate based on CQI (rate = W * log(1 + SINR))
    Takes one data per env.
    """

    def __init__(self, cfg):
        super(RandomCQI, self).__init__()
        self.epsilon = cfg.epsilon

    def forward(self, inputs):
        x_cqi = inputs[0]  # (batch size, 110, 5)
        x_cqi = x_cqi[:, -1, :]  # See only the last time slot. (batch size, 5)
        batch_size = x_cqi.shape[0]
        rand_idx = torch.randint(0, 5, (batch_size,))  # Set to 5 instead of NRB or some shit because it is fixed.
        selected = x_cqi[torch.arange(batch_size).unsqueeze(1), rand_idx.unsqueeze(1)].squeeze()
        # Check if selected correctly
        # print(selected[0:5])
        # print(rand_idx[0:5])
        # print(x_cqi[0:5, :])
        rate = (1 - self.epsilon) * torch.log2(1 + selected)
        return (rate, rand_idx)


class kthBestCQI(nn.Module):
    """
    Comparison scheme which selects RB with k-th best CQI and determines rate based on CQI (rate = W * log(1 + SINR))
    Takes one data per env.
    """

    def __init__(self, cfg):
        super(kthBestCQI, self).__init__()
        self.epsilon = cfg.epsilon
        self.k = cfg.k

    def forward(self, inputs):
        x_cqi = inputs[0]  # (batch size, 110, 5)
        x_cqi = x_cqi[:, -1, :]  # See only the last time slot. (batch size, 5)
        batch_size = x_cqi.shape[0]

        selected, selected_idx = torch.kthvalue(x_cqi, 5 - self.k)

        # Check if selected correctly
        # print(self.k)
        # print(selected[0:5])
        #print(selected_idx[0:5])
        # p rint(x_cqi[0:5, :])
        rate = (1 - self.epsilon) * torch.log2(1 + selected)
        return (rate, selected_idx)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    from configs import Config
    from data_loader import UWADataset

    cfg = Config()
    test_ac = False
    test_cnn = False
    test_fcn = False
    test_randomcqi = True
    test_kthbestcqi = True

    try:  # Check if torchsummaryX is installed
        from torchsummaryX import summary

        '''
        Used torchsummary==1.3.0 & pandas==1.5.2
        Higher version of pandas lead to TypeError
        (https://github.com/nmhkahn/torchsummaryX/issues/28)
        
        Modified torchsummaryX.py from the package.
        Original code line 101:
            df_sum = df.sum()
        Modified code line 101~110:
            """
            mwkim
            1. Change value of numeric_only to True regarding the following warning
            FutureWarning: The default value of numeric_only in DataFrame.sum is deprecated. In a future version, it will default to False. In addition, specifying 'numeric_only=None' is deprecated. Select only valid columns or specify the value of numeric_only to silence this warning.
            2. Change dtype of df['params_nt'] to int64 to enable sum operation.
            """
            # Original code
            # df_sum = df.sum()
            # Modified code
            df_sum = df.astype({'params_nt': 'int64'}).sum(numeric_only=True)
        '''
        data_cqi = torch.zeros(cfg.num_windows, 5)
        data_sensing = torch.zeros(cfg.num_windows, 2, 13, 10)
        cx = torch.zeros(cfg.num_lstm_layers, 1, cfg.hidden_size)
        hx = torch.zeros(cfg.num_lstm_layers, 1, cfg.hidden_size)
        print(f"\n{11 * '='}")
        print("A3C Summary")
        print(f"{11 * '='}")
        summary(ActorCritic_v0(cfg), ((data_cqi, data_sensing), (hx, cx)))

        data_cqi = torch.zeros(1, 110, 5)
        data_sensing = torch.zeros(1, 110, 2, 13, 10)

        print(f"\n{11 * '='}")
        print("CNN Summary")
        print(f"{11 * '='}")
        summary(Convolutional_v0(cfg), (data_cqi, data_sensing))

        print(f"\n{11 * '='}")
        print("FCN Summary")
        print(f"{11 * '='}")
        summary(FullyConnected_v0(cfg), (data_cqi, data_sensing))

    except Exception as e:
        print(e)

    if any([test_ac, test_cnn, test_fcn, test_randomcqi, test_kthbestcqi]):
        dataset = UWADataset(ratio=0.1, snr=10, mode='train', data_dir='data24_v18')

    if test_ac:
        model = ActorCritic_v0(cfg)
        print(model)

        # Initialize hidden states.
        cx = torch.zeros(cfg.num_lstm_layers, 1, cfg.hidden_size)
        hx = torch.zeros(cfg.num_lstm_layers, 1, cfg.hidden_size)

        # Use real data
        data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        it = iter(data_loader)
        (x, y) = next(it)
        cqi_data = x[0][0]  # Shape: [time_length, # RB]
        sensing_data = x[1][0]  # Shape: [time_length, 2, 13, 10]
        rate_data = y[0][0]  # Shape: [time_length, # RB]
        rb_data = y[1][0]  # Shape: [time_length, # RB]

        # Use random data for faster loading
        """
        cqi_data = torch.rand(110, 5)
        sensing_data = torch.rand(110, 2, 13, 10)
        """

        model_estimates = model(((cqi_data[0:cfg.num_windows], sensing_data[0:cfg.num_windows]), (hx, cx)))
        critic_value, actor_value, (hx, cx) = model_estimates

        print(f"Overviews:\n"
              f"Input shapes & dtypes:\n"
              f"    CQI data: {cqi_data[0:cfg.num_windows].shape, cqi_data[0:cfg.num_windows].dtype}\n"
              f"    Sensing data: {sensing_data[0:cfg.num_windows].shape, sensing_data[0:cfg.num_windows].dtype}\n")
        print(f"Output shapes & dtypes:\n"
              f"    Critic: {critic_value.shape, critic_value.dtype}\n"
              f"    Actor: {actor_value.shape, actor_value.dtype}\n"
              f"    Hidden states: {hx.shape, hx.dtype}\n"
              f"                   {cx.shape, cx.dtype}\n")

    if test_cnn:
        model = Convolutional_v0(cfg)
        print(model)

        data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        x, y = next(iter(data_loader))

        # Data with batch removed.
        data_cqi = x[0][0]
        data_sensing = x[1][0]
        y_rate = y[0][0]
        y_rb = y[1][0]

        # Data with batch.
        data_cqi = x[0]
        data_sensing = x[1]
        y_rate = y[0]
        y_rate = y_rate.flatten(0, 1)
        y_rb = y[1]
        y_rb = y_rb.flatten(0, 1)

        print(f"x, y shape: {data_cqi.shape=}, {data_sensing.shape=}, {y_rate.shape=}, {y_rb.shape=}")
        output = model((data_cqi, data_sensing))
        print(f"The output shape of the fully connected model: {output.shape}")

    if test_fcn:
        model = FullyConnected_v0(cfg)
        print(model)

        data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        x, y = next(iter(data_loader))

        # Data with batch removed.
        data_cqi = x[0][0]
        data_sensing = x[1][0]
        y_rate = y[0][0]
        y_rb = y[1][0]

        # Data with batch.
        data_cqi = x[0]
        data_sensing = x[1]
        y_rate = y[0]
        y_rate = y_rate.flatten(0, 1)
        y_rb = y[1]
        y_rb = y_rb.flatten(0, 1)

        print(f"x, y shape: {data_cqi.shape=}, {data_sensing.shape=}, {y_rate.shape=}, {y_rb.shape=}")
        output = model((data_cqi, data_sensing))
        print(f"The output shape of the fully connected model: {output.shape}")

    if test_randomcqi:
        model = RandomCQI(cfg)
        print(model)

        data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        x, y = next(iter(data_loader))

        # Data with batch.
        data_cqi = x[0]
        data_sensing = x[1]
        y_rate = y[0]
        # y_rate = y_rate.flatten(0, 1)
        y_rb = y[1]
        # y_rb = y_rb.flatten(0, 1)

        print(f"x, y shape: {data_cqi.shape=}, {data_sensing.shape=}, {y_rate.shape=}, {y_rb.shape=}")
        output = model((data_cqi, data_sensing))
        print(f"The output shape of the random CQI model: {output[0].shape}, {output[1].shape}")

    if test_kthbestcqi:
        model = kthBestCQI(cfg)
        print(model)

        data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        x, y = next(iter(data_loader))

        # Data with batch.
        data_cqi = x[0]
        data_sensing = x[1]
        y_rate = y[0]
        # y_rate = y_rate.flatten(0, 1)
        y_rb = y[1]
        # y_rb = y_rb.flatten(0, 1)

        print(f"x, y shape: {data_cqi.shape=}, {data_sensing.shape=}, {y_rate.shape=}, {y_rb.shape=}")
        output = model((data_cqi, data_sensing))
        print(f"The output shape of the k-th best CQI model: {output[0].shape}, {output[1].shape}")
