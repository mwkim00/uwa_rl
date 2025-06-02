import torch

from shared_adam import SharedAdam


class Config():
    """
    Configuration settings that are not due to change are declared as a class variable.
    """
    optimizers = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
        'RMSprop': torch.optim.RMSprop,
        'Adadelta': torch.optim.Adadelta,
        'Adagrad': torch.optim.Adagrad,
        'Adamax': torch.optim.Adamax,
        'ASGD': torch.optim.ASGD,
        'LBFGS': torch.optim.LBFGS,
        'SharedAdam': SharedAdam,
    }
    criteria = {
        'MSELoss': torch.nn.MSELoss,
        'L1Loss': torch.nn.L1Loss,
        'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
        'NLLLoss': torch.nn.NLLLoss,
        'PoissonNLLLoss': torch.nn.PoissonNLLLoss,
        'KLDivLoss': torch.nn.KLDivLoss,
        'BCELoss': torch.nn.BCELoss,
        'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss,
    }

    def __init__(self):
        """
        Parameters that may change are declared as an instance variable.
        This allows saving the variables easier, using dict() or pickle.
        """

        """
        Experiment configurations.
        """
        self.seed = 1234  # Random seed.
        self.batch_size = 128  # Batch size.
        self.optimizer_key = 'SharedAdam'
        self.criterion_key = 'MSELoss'
        self.sleep_time = 2  # Time between tests.
        self.run_dir = None  # The directory to save the results.

        """
        Environment configurations. Reflected in the MATLAB code.
        """
        self.nrb = 5  # Number of RBs. NCR in MATLAB.
        self.nue = 4  # Number of UEs. NUE in MATLAB.
        self.sensing_window = (13, 10)  # Sensing window size single sensing impulse. Max: 2816
        self.num_windows = 4  # Number of sensing impulses.
        self.SNR = 10  # Signal-to-noise ratio.
        self.scheme = -1  # Comparison scheme. A non-negative integer.
        self.data_version = -1  # Dataset version.

        """
        Learning configurations.
        """
        self.data_ratio = 0.8  # Train data ratio.
        self.max_episodes = 1e8  # Maximum number of steps in an episode. Arbitrarily large number.
        self.num_steps = 10  # Maximum number of steps per episode. Not used since the time length is finite.
        self.lr = 1e-5  # Learning rate.
        self.gamma = 0.99  # Discount factor.
        self.lamda = 0.005  # See GAE.
        self.hidden_size = 64  # The number of neurons in hidden layers of the neural network.
        self.num_lstm_layers = 2  # Number of LSTM layers.
        self.no_shared = False  # Use an optimizer without shared momentum.
        self.num_processes = 4  # Number of train processes. Actual number of processes is num_processes + 1 (test).
        # Entropy coefficient. Originally 0.01. Don't use for now, since multi-label and entropy doesn't match well.
        self.entropy_coef = 0.01  # Entropy coefficient. See train.py. Encourages exploration.
        self.max_grad_norm = 50  # Maximum gradient norm.
        # Reward weight. reward = cfg.reward_weight[0] * reward_rate + cfg.reward_weight[1] * reward_RB
        self.normalize_rate = True  # Normalize the rate reward. Temporarily set to False.
        self.normalize_RB = True  # Normalize the RB reward.
        self.value_loss_coef = 0.005  # Value loss coefficient.
        # self.reward_weight = [0.1, 1.0, 1.0]  # Reward weight. [rate reward, RB reward, throughput reward]
        self.reward_weight = [0.1, 0.1, 1.0]  # Reward weight. [rate reward, RB reward, throughput reward]
        self.rate_weight = [1.5, 0.5]  # Rate reward weight. [distribution reward, max reward]
        self.rb_weight = [0.4, 1.6]  # RB reward weight. [ideal reward, availability reward]
        self.num_inputs = (self.sensing_window[0] * self.sensing_window[1] + 1) * self.num_windows  # Number of inputs.
        self.action_space = self.nrb * 2  # Number of actions.
        self.deque_maxlen = 100  # Maximum length of the action list.
        # Use RB estimate only to decide the RB. If False, use throughput to decide final RB. (train.py, utils.py)
        self.rb_estimate_only = False
        self.use_linear_policy_score = True  # Linearize policy score instead of softmax. If False, use log_prob.
        self.train_break = True  # If done, break train loop. Else, continue training.
        self.test_break = True  # If done, break test loop. Else, continue testing.
        # Early stop if the reward doesn't improve for this number of episodes (Even though there are no overfitting in
        # normal RL, we have limited data.)
        self.use_early_stop = True  # If True, use early stop.
        self.patience = 4
        # self.early_stop: Deprecated, as cfg class is not shared between processes. Use shared variable instead.
        # self.early_stop = False  # Early stop flag. If True, stop training. Always set to False on initialization.

        self.debug_verbose = False  # Debug mode. If True, print debug_verbose messages.

        self.snrs_list = (-4, -2, 0, 2, 4, 6, 8, 10, 12)  # Default SNR range for training and plotting.

        self.data_start_from_zero = False  # If True, start data index from 0. If False, start from random index.

        """
        Time distribution configurations.
        These variables are also considered in MATLAB.
        
        There are two random variable to be determined: 1. Time gap between my packet occurrences and 2. time gap 
        of other nodes' packet occurrences. The latter decides the active nodes and the RB usage.
        
        Shifted Poisson distribution and shifted exponential distribution are used. Shifted exponential random variable 
        describes the time between the events, with the time gap having a certain minimum value. Shifted Poisson random
        variable expresses the number of events in a certain time interval, with event occurrence following shifted
        exponential distribution. 
        Denoting lambda as the average number of events per unit time (1/lambda is the average time interval between the
        events), the distributions are expressed as:
        Shifted exponential random variable ~ Exp(lambda) + t_min
        Shifted Poisson random variable ~ Pois(1/(1/lambda + t_min))
        where X~Exp(lambda) is an exponential random variable with mean 1/lambda, and X~Pois(lambda) is a Poisson random
        variable with mean lambda.
        Exp(lambda) is expressed in MATLAB as exprnd(1/lambda) and Pois(lambda) is expressed as poissrnd(lambda).
        
        Caution: Do not mix up cfg.lamda and cfg.t_lamda. (cfg.lamda is the GAE coefficient. cfg.t_lamda is the packet 
        distribution-related variable)
        """
        self.t_lamda = 0.4  # Average packet per minute.
        self.t_min = 1  # Minimum time gap between packet occurrences.

        # Configurations for normal neural networks (CNN, FCN)
        self.epochs = 100
        self.eval_period = 10

        # Model version
        self.ac_version = 1
        self.cnn_version = 1
        self.fcn_version = 0
        self.epsilon = 0.37  # epsilon used for epsilon-rate control in RandomCQI model.
        self.k = 0  # Used to select the RB used in kthBestCQI. k=0 chooses the RB with best CQI.
        
        self.use_scheduler = True
        self.scheduler = 'ReduceLROnPlateau'

        self.device = 0  # Device to use. None: Choose device according to criterion. -1: CPU, 0~: GPU idx.


# header_dict: Used to distinguish between different configurations. Used in main.py. Contains scheme information only.
header_dict = {0: r"a3c_\d+-", 1: "ac-", 2: "cnn-", 3: "fcn_v0-", 4: "fcn_v1-"}
# scheme_dict: Used to label different schemes in plot_figures.py
scheme_dict = {0: "A3C", 1: "A2C", 2: "CNN-based", 3: "FCN-v0", 4: "FCN-v1"}
base_cfg = Config()


if __name__ == '__main__':
    import os
    import pprint
    import pickle
    import json

    cfg = Config()
    cfg.nrb = 10
    print(f"The class variables of Config are: ")
    pprint.pprint(Config.__dict__)
    print(f"The instance variables of Config are: ")
    pprint.pprint(cfg.__class__.__dict__)

    print(f"Save and reload cfg.")
    with open('test.pickle', 'wb') as f:
        pickle.dump(Config, f)  # Or could save an instance also.
    with open('test.json', 'w') as f:
        json.dump(cfg.__dict__, f)

    with open('test.pickle', 'rb') as f:
        loaded_cfg = pickle.load(f)
        print(f"\nRestored cfg from pickle is: ")
        pprint.pprint(loaded_cfg.__dict__)
        cfg = loaded_cfg()
        print(cfg.optimizer_key)
    with open('test.json', 'r') as f:
        loaded_cfg = json.load(f)
        print(f"\nRestored cfg from json is: ")
        pprint.pprint(loaded_cfg)

    print(os.getcwd())
    os.remove('test.pickle')
    os.remove('test.json')
