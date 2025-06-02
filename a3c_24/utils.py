import os
from typing import Any, Optional, Union
import random
import re
import json
import pprint
import subprocess

import torch.nn as nn
import einops
import numpy as np
import torch
import matplotlib.pyplot as plt


def encode_datetime(datetime: str) -> str:
    """
    Encode datetime to hex string
    """
    return hex(int(datetime))[2:].upper()


def decode_datetime(datetime_hex: str) -> str:
    """
    Decode hex string to datetime
    """
    return str(int(datetime_hex, base=16))


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val=0, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def choose_device(gpu_idx: Optional[int] = None, criterion: str = 'memory') -> str:
    """
    Choose GPU according to certain criterion (memory usage, temperature, etc.)
    Used torch.cuda
    Arguments:
        gpu_idx: int or None. The index of the GPU to use. To use CPU, set to -1.
            If None, choose according to criterion.
        criterion: str. The criterion to choose the GPU. 'memory' or 'temperature'.
    """
    if type(gpu_idx) is str:
        gpu_idx = None  # Reset to None
    if not torch.cuda.is_available():
        print('CUDA is not available. Using CPU.')
        return 'cpu'
    if gpu_idx is not None:
        if gpu_idx == -1:
            print('Using CPU')
            return 'cpu'
        print(f'Using GPU {gpu_idx}')
        return f'cuda:{gpu_idx}'

    # DEBUG: Temperature measurement is not available.
    if criterion == 'temperature':
        print('Temperature-measuring code is currently not available. Changing criterion to memory.')
        criterion = 'memory'

    gpu_num = torch.cuda.device_count()
    if criterion == 'temperature':
        try:
            temp = [torch.cuda.temperature(i) for i in range(gpu_num)]
            gpu_idx = temp.index(min(temp))
            print(f'Using GPU {gpu_idx} with temperature {temp[gpu_idx]}°C')
            return f'cuda:{gpu_idx}'
        except ModuleNotFoundError:
            print('torch.cuda.temperature not found. Changing criterion to memory.')
            criterion = 'memory'
    if criterion == 'memory':  # See torch.cuda.memory_summary() for summarized information.
        # Buggy code
        # mem = [torch.cuda.memory_allocated(i) for i in range(gpu_num)]
        # gpu_idx = mem.index(min(mem))

        # New code
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
                                stdout=subprocess.PIPE)
        mem = [int(x) for x in result.stdout.decode('utf-8').strip().split('\n')]
        gpu_idx = mem.index(max(mem))

        print(f'Using GPU {gpu_idx} with memory {mem[gpu_idx]} bytes')
        return f'cuda:{gpu_idx}'


def set_seed(seed: int = 1234, benchmark: bool = True) -> None:
    """
    Set seed for reproducibility
    Ref) https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    if benchmark:
        torch.backends.cudnn.benchmark = False  # Setting to True may increase the performance.


def seed_worker(worker_id: int) -> None:
    """
    Seed worker for reproducibility
    Ref) https://pytorch.org/docs/stable/notes/randomness.html
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ensure_shared_grads(model: Any, shared_model: Any) -> None:
    """
    Ensure that the shared model has the gradients from the model
    """
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def normalized_columns_initializer(weights: torch.tensor, std: float = 1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m: Any) -> None:
    """
    Initialize weights
    Ref) https://pytorch.org/docs/stable/nn.init.html
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def map_tensor_to_range(rb_estimate: torch.Tensor, options: str = 'linear', range: () = (0.1, 1)) -> torch.Tensor:
    """
    Map RB availability to a certain range, or softmax it.
    Arguments:
        rb_estimate: torch.Tensor of shape (NRB). The raw RB availability from the model.
        options: str. The mapping options. 'linear' or 'softmax'.
        range: tuple. The range to map the RB availability, if options is 'linear'.
    """
    if options == 'linear':
        return range[0] + (range[1] - range[0]) * (rb_estimate - torch.min(rb_estimate)) / (
                torch.max(rb_estimate) - torch.min(rb_estimate))
    elif options == 'softmax':
        return torch.nn.functional.softmax(rb_estimate, dim=-1)
    else:
        raise ValueError('Invalid options. Choose either linear or softmax.')


def eval_action(model_estimates, rate_true, rb_true, cfg, is_test=False, meters=(), meters_verbose=None) -> (float, bool):
    """
    Evaluate action, calculate reward and done.

    Rate reward: Sum of two rewards; distribution reward, and maximum rate reward.
    cf. From '23, the variance loss that used variance is deprecated as MSE = Variance + Bias ** 2.
    torch.var(torch.ones(1, 5)) = 0.0, nn.MSELoss()(torch.ones(1, 5), torch.zeros(1, 5)) = 1.0
    Variance shapes the estimate distrubution similar to the true distribution, whereas MSE decreases the difference
    between the estimate and the true value, which decreases the variance.

    Distribution reward (reward_rate_dist): Uses -MSELoss. Here, the estimated rate values exceeding the true rate
    values are penalized. (The estimated rate is replaced by zero.)
    Maximum rate reward (reward_rate_max): -MSELoss between the true rate and the estimated rate of the ideal RB. Though
    this reward is included in the distribution reward, it is separated for clarity.

    RB reward: Sum of two rewards; ideal reward, and availability reward. reward between chosen RB - ideal RB, reward between chosen RB - available RB.
    Ideal RB reward (reward_RB_ideal): Cross entropy loss between the estimated RB availability and the ideal RB.
    Available RB reward (reward_RB_avail): MSE loss between the estimated RB availability and the available RB.
    MSE loss is used.
    cf. CELoss: -torch.log(torch.exp(rb_estimate) / torch.sum(torch.exp(rb_estimate)))

    Arguments:
        model_estimates: torch.Tensor of shape (1, 2 * NRB).
            Model estimates of available rate and RB availability probability.
        rate_true: torch.Tensor of shape (NRB). The true achievable rates of each RB.
        rb_true: torch.Tensor of shape (NRB). The true RB usage information.
            (If available 0, else non-zero. The number indicates the node index in use)
        cfg: Configurations class. The configurations.
        is_test: bool. Whether the model is in test mode. If Trhe, done is calculated more strictly.
        meters: tuple. Tuple of AverageMeter instances to update. Updated only in test mode.
            (rate success rate, RB success rate, success rate, average throughput)
        meters_verbose: tuple or None. If None, do not save additional data. If not None, save reward data.
    Returns:
        reward: torch.Tensor if shape () with float value. The reward.
        done: bool. Whether the episode is done.
    """
    if model_estimates.isnan().any():  # Check if NaN exists in model output. Debug.
        print("Model estimates contain NaN values.")
        print(f"{model_estimates=}")
    model_estimates = model_estimates.squeeze()

    rate_estimate = model_estimates[..., 0:cfg.nrb]  # Renewed version
    # rate_estimate = model_estimates[0:cfg.nrb]

    rate_estimate = rate_estimate * (rate_estimate < rate_true)  # If estimated rate exceeds true rate, set to 0.

    rb_estimate = model_estimates[..., cfg.nrb:]  # Estimated RB usage score. Higher means higher availability.  # Renewed version
    # rb_estimate = model_estimates[cfg.nrb:]  # Estimated RB usage score. Higher means higher availability.

    if cfg.rb_estimate_only:
        rb_chosen = torch.argmax(rb_estimate)  # Use the estimated RB availability as the availability score.
    else:
        rb_chosen = torch.argmax(map_tensor_to_range(rb_estimate) * rate_estimate)  # Use throughput as avail score.

    # If the RB is not available, set the rate to 0.
    # rate_estimate is trained to learn the zero-ed out rate true. Two losses for RB?
    rate_true = rate_true * (rb_true == 0)  # rate_true being all zero could be a problem, if all RBs are being used.
    rate_true_max = torch.max(rate_true)
    if rate_true_max == 0:  # rate_true_max could be zero.
        rate_true_max = 1

    rb_ideal = torch.argmax((rate_true * (rb_true == 0)), dim=-1)  # Renewed version
    # rb_ideal = torch.argmax((rate_true * (rb_true == 0)), dim=0)

    # Rate reward.
    if cfg.normalize_rate:
        # Maximum rate reward.
        if len(rate_estimate.shape) == 1:
            reward_rate_max = -nn.MSELoss()(rate_estimate[rb_ideal], rate_true[rb_ideal]) / rate_true[rb_ideal] ** 2
        else:
            reward_rate_max = -nn.MSELoss()(rate_estimate[torch.arange(rate_estimate.size(0)), rb_ideal], rate_true[torch.arange(rate_true.size(0)), rb_ideal]) / rate_true[torch.arange(rate_true.size(0)), rb_ideal] ** 2  # Renewed version

        # Distribution reward.
        reward_rate_dist = -nn.MSELoss()(rate_estimate, rate_true) / rate_true_max ** 2  # Normalize by maximum rate.

    else:
        if len(rate_estimate.shape) == 1:
            reward_rate_max = -nn.MSELoss()(rate_estimate[rb_ideal], rate_true[rb_ideal])
        else:
            reward_rate_max = -nn.MSELoss()(rate_estimate[torch.arange(rate_estimate.size(0)), rb_ideal], rate_true[torch.arange(rate_true.size(0)), rb_ideal])  # Renewed version
        reward_rate_dist = -nn.MSELoss()(rate_estimate, rate_true)

    reward_rate = cfg.rate_weight[0] * reward_rate_dist + cfg.rate_weight[1] * reward_rate_max

    # RB reward.
    # Ideal RB reward.
    reward_RB_ideal = -nn.CrossEntropyLoss()(rb_estimate, rb_ideal)

    # Available RB reward.
    # This could somewhat contradict with reward_RB_ideal. Omit ideal RB?
    reward_RB_avail = -nn.MSELoss()(rb_estimate, (rb_true == 0).float())
    """
    # Reward calculated without ideal RB.
    rb_estimate_sliced = torch.cat([rb_estimate[:rb_ideal], rb_estimate[rb_ideal+1:]])
    rb_true_sliced = torch.cat([rb_true[:rb_ideal], rb_true[rb_ideal+1:]])
    reward_RB_avail = -nn.MSELoss()(rb_estimate_sliced, (rb_true_sliced == 0).float())
    """

    reward_RB = cfg.rb_weight[0] * reward_RB_ideal + cfg.rb_weight[1] * reward_RB_avail

    reward = (cfg.reward_weight[0] * reward_rate + cfg.reward_weight[1] * reward_RB
              + cfg.reward_weight[2] * rate_estimate[rb_chosen])
    """
    if cfg.debug_verbose and torch.rand(1).item() < 0.0003:  # Debug: compare the rewards.
        print(f"    DEBUGGING: {cfg.rate_weight[0]*reward_rate_dist=:.2f} | {cfg.rate_weight[1]*reward_rate_max=:.2f} |"
              f" {cfg.rb_weight[0]*reward_RB_ideal=:.2f} | {cfg.rb_weight[1]*reward_RB_avail=:.2f} |"
              f" {cfg.reward_weight[0]*reward_rate=:.2f} | {cfg.reward_weight[1]*reward_RB=:.2f}")
    """
    # Debug: compare the rewards.
    # if torch.rand(1) < 0.01:
    #     print(f"Rate reward: {reward_rate:.4f} | RB reward: {reward_RB:.4f}\n")

    # If selected RB is not available or estimated rate of chosen RB is higher than the available rate, done.
    done = (rb_true[rb_chosen] != 0) or (rate_estimate[rb_chosen] > rate_true[rb_chosen])
    if is_test:
        meters[0].update(rate_estimate[rb_chosen] <= rate_true[rb_chosen])
        meters[1].update(rb_true[rb_chosen] == 0)  # Update only if the chosen RB is available.
        meters[2].update(not done)  # Update depending on rate and RB success.
        meters[3].update(rate_estimate[rb_chosen])

    if meters_verbose is not None:
        meters_verbose[0].update(reward_rate_dist)
        meters_verbose[1].update(reward_rate_max)
        meters_verbose[2].update(reward_RB_ideal)
        meters_verbose[3].update(reward_RB_avail)
        meters_verbose[4].update(reward)

    if reward == torch.inf or reward == -torch.inf or torch.isnan(reward):  # Debug
        raise ValueError(f'Abnormal reward. {reward=}, {reward_rate=}, {reward_RB=}, {rb_true=}')
    return reward, done


def sync_cfg(cfg: Any, file_name='MW_RL_input.m') -> None:
    """
    Unused.
    Synchronize configurations. Open .m file and change the default values according to the configurations.
    Arguments:
        cfg: Config. The configurations.
        file_name: str. The name of the .m file. Should be located in UWA_MATLAB.
    Returns:
        None
    """
    path = os.path.join(os.getcwd(), '../UWA_MATLAB', file_name)
    with open(path, 'r+') as f:
        content = f.read()
        keys = ['NCR', 'NUE', 'sensing_window', 'num_windows', 'SNR', 't_lamda', 't_min']
        values = [cfg.nrb, cfg.nue, cfg.sensing_window, cfg.num_windows, cfg.SNR, cfg.t_lamda, cfg.t_min]

        content = _update_file(content, keys, values)

    # Erase and rewrite the file.
    with open(path, 'w') as f:
        f.write(content)
    return


def _update_file(content: str, keys: list[str], values: list[Union[tuple, str]]) -> str:
    """
    Unused.
    Update the file.
    Arguments:
        content: str. The content of the file.
        keys: str. The key to change, in MATLAB.
        values: float. The value of the key.
    Returns:
        content: str. The updated content.
    """
    for key, value in zip(keys, values):
        if type(value) == tuple:
            value = f"{value}".replace('(', '[').replace(')', ']')
            pattern = rf"\n{key}=\[\d+, \d+\]"
            replacement = f"\n{key}={value}"
        else:
            pattern = rf"\n{key}=\d+"
            replacement = f"\n{key}={value}"
        content = re.sub(pattern, replacement, content)
    return content


def reshape_sensing_data(data_sensing: np.ndarray, window: tuple = (13, 10)) -> np.ndarray:
    """
    Deprecated (Sensing data is now reshaped in MATLAB).
    Reshape the sensing data to fit the model input.
    Arguments:
        data_sensing: np.ndarray of shape (num_windows, 2816, 1). The sensing data.
        window: tuple. Size of the sensing window.
    Returns:
        reshaped_data: np.ndarray of shape (num_windows, 2, window[0], window[1]). The reshaped sensing data.
    """
    window_size = window[0] * window[1]
    start_idx = 1536  # data_sensing.shape[0] * 12 // 22
    end_idx = 1664  # data_sensing.shape[0] * 13 // 22
    leftover = window_size - (end_idx - start_idx)
    start_idx = start_idx - int(leftover / 2)
    end_idx = start_idx + window_size
    if leftover < 0:
        raise ValueError('Window size is too small.')
    # print(data_sensing[:, start_idx:end_idx, :].shape, leftover, start_idx, end_idx)
    reshaped_data = einops.rearrange(data_sensing[:, start_idx:end_idx, :], 'b (h w) 1 -> b h w', h=window[0],
                                     w=window[1])
    real_part = reshaped_data.real
    imag_part = reshaped_data.imag
    reshaped_data = np.stack([real_part, imag_part], axis=1)
    return reshaped_data


def visualize(dir_name: str, show: bool = False) -> None:
    """
    Visualize the training results.
    Arguments:
        dir_name: str. The name of the file to visualize
        show: bool. If True, show image. Else, save..
    """
    file_path = os.path.join(dir_name, 'log.csv')
    csv_header = ['Time', 'Num episodes', 'Episode reward', 'Episode length', 'Rate success rate', 'RB success rate',
                  'Success rate', 'Average throughput']
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    data = data.T
    time = data[0]
    num_episodes = data[1]
    episode_reward = data[2]
    episode_length = data[3]
    rate_success_rate = data[4]
    rb_success_rate = data[5]
    success_rate = data[6]
    avg_throughput = data[7]

    fig, axs = plt.subplots(2, 3, figsize=(30, 10))

    it = iter(axs.flat)

    ax = next(it)
    ax.plot(num_episodes, episode_reward, label='Episode reward')
    ax.set_title('Episode reward')
    ax.set_xlabel('Num episodes')
    ax.set_ylabel('Episode reward')
    ax.legend()

    ax = next(it)
    ax.plot(num_episodes, episode_length, label='Episode length')
    ax.set_title('Episode length')
    ax.set_xlabel('Num episodes')
    ax.set_ylabel('Episode length')
    ax.legend()

    ax = next(it)
    ax.plot(num_episodes, rate_success_rate, label='Rate success rate')
    ax.set_title('Rate success rate')
    ax.set_xlabel('Num episodes')
    ax.set_ylabel('Rate success rate')
    ax.legend()

    ax = next(it)
    ax.plot(num_episodes, rb_success_rate, label='RB success rate')
    ax.set_title('RB success rate')
    ax.set_xlabel('Num episodes')
    ax.set_ylabel('RB success rate')
    ax.legend()

    ax = next(it)
    ax.plot(num_episodes, success_rate, label='Success rate')
    ax.set_title('Success rate')
    ax.set_xlabel('Num episodes')
    ax.set_ylabel('Success rate')
    ax.legend()

    ax = next(it)
    ax.plot(num_episodes, avg_throughput, label='Average throughput')
    ax.set_title('Average throughput')
    ax.set_xlabel('Num episodes')
    ax.set_ylabel('Average throughput')
    ax.legend()

    if show:
        plt.show()

        plt.cla()  # Clear axis
        plt.clf()  # Clear figure
        plt.close()  # Close a figure window
    else:
        plt.savefig(os.path.join(dir_name, 'result.png'))

        plt.cla()  # Clear axis
        plt.clf()  # Clear figure
        plt.close()  # Close a figure window


def json_pprint(dir: str) -> None:
    """
    Pretty print ugly configs.json.
    Arguments:
        dir: str. Location of configs.json, under log directory.
    """
    with open(os.path.join(dir, 'configs.json'), 'r') as f:
        data = json.load(f)
    pprint.pprint(data)


class Counter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0  # number of total values of the counter
        self.true = 0  # number of true values of the counter

    def update(self, tf):
        if len(tf.shape) == 0:
            self.count += 1
        else:
            self.count += len(tf)
        self.true += torch.sum(tf).item()

    def get(self):
        if self.count == 0:
            return 0
        return self.true / self.count


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Parameters:
            patience (int): How long to wait after the last time validation loss improved.
            verbose (bool): Print a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, key_name='net'):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)

        checkpoint = {
            key_name: model.state_dict()
        }
        torch.save(checkpoint, self.path)

        self.val_loss_min = val_loss


if __name__ == "__main__":
    test_visualize = True
    test_eval_action = False
    test_sync_cfg = False
    test_json_pprint = False

    if test_json_pprint:
        json_pprint('a3c_1-SNR_10-1F01062C')
        print('json_pprint test ended')

    if test_visualize:
        visualize('1F6BB44B-a3c_4-SNR_12', show=True)
        print('visualize test ended')

    if test_eval_action:
        # eval_action test
        print("eval_action test")
        from torch.utils.data import DataLoader

        from model import ActorCritic_v0
        from configs import Config
        from data_loader import UWADataset

        set_seed(0)

        data_loader = DataLoader(UWADataset(), batch_size=1, shuffle=True)

        cfg = Config()
        model = ActorCritic_v0(cfg)
        print(model)

        cx = torch.zeros(cfg.num_lstm_layers, 1, cfg.hidden_size)
        hx = torch.zeros(cfg.num_lstm_layers, 1, cfg.hidden_size)

        step = 0

        it = iter(data_loader)
        (x, y) = next(it)
        cqi_data = x[0][0]
        sensing_data = x[1][0]

        rate_true = y[0][0]
        rb_true = y[1][0]

        model_estimates = model(
            ((cqi_data[step:step + cfg.num_windows], sensing_data[step:step + cfg.num_windows]), (hx, cx)))
        actor_value = model_estimates[1].clone()  # Deep copy.

        rate_success_rate = AverageMeter()
        rb_success_rate = AverageMeter()
        success_rate = AverageMeter()
        avg_throughput = AverageMeter()

        meters = (rate_success_rate, rb_success_rate, success_rate, avg_throughput)

        print(f"Rate part of the model: {actor_value.squeeze()[0:cfg.nrb].detach()}\n"
              f"RB part of the model:   {actor_value.squeeze()[cfg.nrb:].detach()}\n"
              f"Actual rate:            {rate_true[step + cfg.num_windows]}\n"
              f"Actual RB:              {rb_true[step + cfg.num_windows]}\n")
        reward, done = eval_action(actor_value, rate_true[step + cfg.num_windows], rb_true[step + cfg.num_windows], cfg,
                                   is_test=True, meters=meters)
        print(f"{rate_success_rate.avg=}\n"
              f"{rb_success_rate.avg=}\n"
              f"{success_rate.avg=}\n"
              f"{avg_throughput.avg=}\n")
        print(f"Reward: {reward}\n"
              f"Done: {done}\n")

        '''
        modified_actor_value = actor_value.clone()
        modified_actor_value[0, 0:cfg.nrb] = rate_true[step+cfg.num_windows]
        modified_actor_value[0, cfg.nrb:] = 0
        modified_actor_value[0, cfg.nrb + torch.argmax((rb_true[step+cfg.num_windows] == 0) * rate_true[step+cfg.num_windows])] = 1
        reward = eval_action(modified_actor_value, rate_true[step+cfg.num_windows], rb_true[step+cfg.num_windows], cfg)[0]
        print(f"Modified actor value: {modified_actor_value}\n"
              f"Modified reward: {reward}\n")

        modified_actor_value = actor_value.clone()
        modified_actor_value[0, 0:cfg.nrb] = rate_true[step+cfg.num_windows]
        modified_actor_value[0, cfg.nrb:] = 0
        modified_actor_value[0, cfg.nrb + torch.argmax((rb_true[step+cfg.num_windows] == 0) * rate_true[step+cfg.num_windows])] = 10
        reward = eval_action(modified_actor_value, rate_true[step+cfg.num_windows], rb_true[step+cfg.num_windows], cfg)[0]
        print(f"Modified actor value: {modified_actor_value}\n"
              f"Modified reward: {reward}\n")

        modified_actor_value = actor_value.clone()
        modified_actor_value[0, 0:cfg.nrb] = rate_true[step+cfg.num_windows]
        modified_actor_value[0, cfg.nrb:] = (rb_true[step+cfg.num_windows] == 0) * rate_true[step+cfg.num_windows]
        reward = eval_action(modified_actor_value, rate_true[step+cfg.num_windows], rb_true[step+cfg.num_windows], cfg)[0]
        print(f"Modified actor value: {modified_actor_value}\n"
              f"Modified reward: {reward}\n")
        '''

        custom_actions = [torch.tensor([[3.35, 4.41, 2.16, 4.13, 2.26, 1, -1, -1, -1, -1]]),
                          torch.tensor([[-1., -1., -1., -1., -1., 1, -1, -1, -1, -1]]),
                          ]
        for i in range(len(custom_actions)):
            reward = \
                eval_action(custom_actions[i], rate_true[step + cfg.num_windows], rb_true[step + cfg.num_windows], cfg)[
                    0]
            print(f"For custom action {i + 1}, the reward is {reward}")

    if test_sync_cfg:
        # sync_cfg test
        from configs import Config
        import shutil

        shutil.copyfile('../UWA_MATLAB/MW_RL_input.m', '../UWA_MATLAB/MW_RL_input_.m')
        cfg = Config()
        cfg.nrb = 1234
        cfg.nue = 2345
        cfg.sensing_window = (3456, 4567)
        cfg.num_windows = 5678
        cfg.SNR = 6789
        cfg.t_lamda = 7890
        cfg.t_min = 8901

        sync_cfg(cfg, 'MW_RL_input_.m')

    print("End of test.")
