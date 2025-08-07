import os
import datetime
import itertools
import json

try:
    import torch
    import torch.multiprocessing as mp
except Exception as e:
    print(f"Error while importing PyTorch: {e}")
    import sys
    sys.exit(1)

from model import ActorCritic_v0, FullyConnected_v0, Convolutional_v0
from test import test
from train import train, train_NN
from utils import encode_datetime, set_seed, choose_device, visualize
from configs import Config, header_dict
from plot_figures import plot_figures


def main(run_idx=None, schemes=(0, 1, 2), snrs=(-4, -2, 0, 2, 4, 6, 8, 10, 12), cfg=Config()):
    """
    Arguments:
        run_idx: tuple(str). The index of the run. Used to specify run, preventing overlaps. When finetuning, set to None.
    """
    if run_idx is not None:
        data_idx = run_idx[0]
    run_idx = '_'.join(run_idx)
    log_dir = f"log_{run_idx}"
    os.makedirs(log_dir, exist_ok=True)

    # Iterate through different configuration settings.
    for scheme, snr in itertools.product(schemes, snrs):
        # cfg = Config()  # Use the configuration given to the function

        # Set configuration as given in the for loop.
        cfg.SNR = snr
        cfg.scheme = scheme
        cfg.data_version = data_idx

        cfg.seed = int(datetime.datetime.now().strftime('%m%d%H%M%S'))
        set_seed(cfg.seed)
        os.environ['OMP_NUM_THREADS'] = '1'  # Prevents deadlocks in multithreading.

        # CUSTOM SETTINGS HERE
        # Example)
        # if scheme = 0
        #     cfg.value_loss_coef = 0.0005
        if scheme != 0:
            cfg.max_episodes = cfg.max_episodes * cfg.num_processes

        """
        ML-based comparison schemes. (To see other comparison schemes, see comparison.py)
        0: Proposed (A3C).
        1: Actor-Critic approach
        2: CNN-based approach
        3: FCN v1
        4: FCN v2
        """
        file_name_header = header_dict[scheme]
        if scheme == 0:  # A3C
            file_name_header = file_name_header.replace(r'\d+', str(cfg.num_processes))
        elif scheme == 1:  # Actor-Critic
            cfg.num_processes = 1
        # TODO: Remove below code after train_NN()
        elif scheme == 2:  # CNN
            cfg.num_processes = 1
            cfg.num_windows = 1
        elif scheme in (3, 4):  # FCN
            cfg.num_processes = 1
            cfg.num_windows = 1  # To set the data loader batch size.
        elif scheme in header_dict.keys():
            pass
        else:
            raise ValueError(f"Invalid comparison scheme index: {scheme}")

        if run_idx is None:
            run_idx = encode_datetime(datetime.datetime.now().strftime('%m%d%H%M%S'))
        subdir_name = run_idx + '-' + file_name_header + 'SNR_' + str(cfg.SNR)
        run_dir = f"{log_dir}/{subdir_name}"

        # Prevent overlapping.
        if os.path.exists(run_dir):
            print(f"File {run_dir} already exists. Skipping.")
        else:
            print(f"File {run_dir} does not exist. Training...")
            # Save configurations.
            cfg.run_dir = run_dir
            os.makedirs(run_dir, exist_ok=True)

            if type(cfg.device) is torch.device:
                cfg.device = torch.device(cfg.device)
            else:
                cfg.device = torch.device(choose_device(cfg.device))

            with open(os.path.join(run_dir, 'configs.json'), 'w') as f:
                try:
                    json.dump(cfg.__dict__, f, indent=4)
                except Exception as e:
                    pass

            print(f"Run index: {run_idx}")
            
            if scheme in (0, 1):  # If the model is RL-based, run the RL code.
                shared_model = ActorCritic_v0(cfg).to(cfg.device)
                shared_model.share_memory()

                if cfg.no_shared:
                    optimizer = None
                else:
                    optimizer = cfg.optimizers[cfg.optimizer_key](shared_model.parameters(), lr=cfg.lr)
                    optimizer.share_memory()

                processes = []

                # Multiprocessing tools.
                # counter: Shared counter, count the number of total steps.
                # lock: Lock to prevent the processes from executing the similar code at the same time.
                # cf. acquire(): Claim lock / release(): Release lock
                counter = mp.Value('i', 0)
                early_stop = mp.Value('i', 0)  # Early stop flag. 0: Continue training, 1: Stop training.
                lock = mp.Lock()

                with open(os.path.join(run_dir, 'log.csv'), 'w') as f:
                    '''
                    Time records the actual time passed since the beginning of the training.
                    Num steps records the total number of steps taken while training.
                    Episode reward records the total reward of the test episode.
                    Episode length records the length of the test episode.
                    Rate success rate records the rate success rate of the test episode.
                    RB success rate records the RB success rate of the test episode.
                    (Rate success rate and RB success rate is recorded to compare the performance and finetune the model.)
                    Success rate records the success rate (rate success & RB success) of the test episode.
                    Average throughput records the average throughput of the test episode.
                    '''
                    f.write('Time,Num episodes,Episode reward,'
                            'Episode length,Rate success rate,'
                            'RB success rate,Success rate,Average throughput,'
                            'Rate dist. reward, Rate max reward,RB ideal reward,RB avail. reward\n')

                p = mp.Process(target=test, args=(cfg, shared_model, counter, early_stop, run_dir))
                p.start()
                processes.append(p)

                for rank in range(0, cfg.num_processes):
                    p = mp.Process(target=train, args=(rank, cfg, shared_model, counter, lock, early_stop, optimizer))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

                visualize(run_dir)

            # If the model is a regular neural network.
            else:
                with open(os.path.join(run_dir, 'log.csv'), 'w') as f:
                    '''
                    Time records the actual time passed since the beginning of the training.
                    Num steps records the total number of steps taken while training.
                    Episode reward records the total reward of the test episode.
                    Episode length records the length of the test episode.
                    Rate success rate records the rate success rate of the test episode.
                    RB success rate records the RB success rate of the test episode.
                    (Rate success rate and RB success rate is recorded to compare the performance and finetune the model.)
                    Success rate records the success rate (rate success & RB success) of the test episode.
                    Average throughput records the average throughput of the test episode.
                    '''
                    f.write('Time,Num episodes,Episode reward,'
                            'Episode length,Rate success rate,'
                            'RB success rate,Success rate,Average throughput,'
                            'Rate dist. reward, Rate max reward,RB ideal reward,RB avail. reward\n')

                train_NN(cfg, optimizer=None)


if __name__ == '__main__':
    mp.set_start_method('spawn')

    run_ids = ['123', '124', '125', '126', '127', '128', '129', '130', '131', '132']
    for run_cfgs_idx in range(len(run_ids)):
        cfg = Config()
        # cfg.lr = (run_cfgs_idx % 5 + 1) * 3e-6
        cfg.lr = (run_cfgs_idx % 5) * 1e-6 + 3e-6
        cfg.use_scheduler = False

        weights_list = [[0.3, 0.3, 0.8], [0.5, 0.5, 0.5], [0.7, 0.7, 0.3], [0.9, 0.9, 0.1], [1.0, 1.0, 0.0]]
        if run_cfgs_idx > 4:
            cfg.lr = 5e-6
            cfg.reward_weight = weights_list[run_cfgs_idx - 5]

        # run_idx = encode_datetime(datetime.datetime.now().strftime('%m%d%H%M%S'))  # Auto-index based on datetime.

        data_idx = 'v21'
        # run_idx = '26'  # Use run idx given above
        run_idx = run_ids[run_cfgs_idx]
        run_idx_joined = data_idx + '_' + run_idx

        schemes = (0, 1,)
        snrs = (-4, -2, 0, 2, 4, 6, 8, 10, 12)

        main(run_idx=(data_idx, run_idx), schemes=schemes, snrs=snrs, cfg=cfg)  # Or use cfg.snrs_list
        
        # Plot
        # CNN_RUN_IDX = ''
        # FCN_RUN_IDX = ''
        # run_idx_tuple = (run_idx_joined, run_idx_joined, CNN_RUN_IDX, FCN_RUN_IDX)
        
        # Option 1: Use run index
        try:
            plot_figures(run_idx=run_idx_joined, schemes=schemes, fig_dir=f"log_{run_idx_joined}")
        except Exception as e:
            pass
        # Option 2: Use a combination if indices
        # plot_figures(run_idx=run_idx_joined, schemes=(0,1,2,3), fig_dir=run_idx_tuple)

