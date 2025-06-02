import time
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from model import ActorCritic_v0, FullyConnected_v0, Convolutional_v0
from data_loader import UWADataset
from utils import set_seed, ensure_shared_grads, eval_action, map_tensor_to_range, EarlyStopping, Counter, AverageMeter


def train(rank, cfg, shared_model, counter, lock, early_stop, optimizer):
    """
    Arguments:
        rank: int. the rank, or the process index of the process.
        cfg: Configuration class. The configurations.
        shared_model: ActorCritic_v0 class. The model to train.
        counter: torch.multiprocessing.Value. The shared counter.
        lock: torch.multiprocessing.Lock. The shared lock.
        early_stop: torch.multiprocessing.Value. The shared early stop flag.
        optimizer: The optimizer to use.
    Returns:
        None
    """
    set_seed(cfg.seed + rank)
    
    model = ActorCritic_v0(cfg).to(cfg.device)

    model.train()

    dataset = UWADataset(snr=cfg.SNR, mode='train', ratio=cfg.data_ratio, data_dir='data24_'+cfg.data_version)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"Rank {rank:02d} | Data loaded")
    step = 0  # Starting index of the data window.

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=cfg.lr)
    
    """
    # TODO: Added code
    # Learning rate scheduler
    if cfg.use_scheduler:
        if cfg.scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        elif cfg.scheduler == 'Lambda':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.85 ** epoch)
        elif cfg.scheduler == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        else:
            raise ValueError(f'Invalid scheduler {cfg.scheduler}')
    """


    x, y = next(iter(data_loader))
    if cfg.data_start_from_zero:
        start_idx = 0
    else:
        start_idx = torch.randint(0, len(x[0][0]) - cfg.num_windows, (1,)).item()
    data_cqi = x[0][0][start_idx:].to(cfg.device)
    data_sensing = x[1][0][start_idx:].to(cfg.device)
    y_rate = y[0][0][start_idx:].to(cfg.device)
    y_rb = y[1][0][start_idx:].to(cfg.device)

    done = True

    episode_idx = 0
    while not early_stop.value:
        # Start of an episode, sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(cfg.num_lstm_layers, 1, cfg.hidden_size).to(cfg.device)
            hx = torch.zeros(cfg.num_lstm_layers, 1, cfg.hidden_size).to(cfg.device)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        rewards = []
        entropies = []
        policies = []

        for step in range(len(data_cqi) - cfg.num_windows):  # step: episode length
            # actor_value: [rate, RB availability probabilities]
            critic_value, actor_value, (hx, cx) = model(((data_cqi[step:step + cfg.num_windows],
                                                          data_sensing[step:step + cfg.num_windows]),
                                                         (hx, cx)))  # Send state to device.

            # prob: used to calculate entropy and select RB.
            prob = F.softmax(actor_value[0, cfg.nrb:], dim=-1)
            # log_prob: used to calculate entropy and as a GAE coefficient.
            log_prob = F.log_softmax(actor_value[0, cfg.nrb:], dim=-1)

            # Entropy usage is deprecated.
            entropy = -(log_prob * prob).sum(0, keepdim=True)
            entropies.append(entropy)

            if cfg.rb_estimate_only:  # Use model's RB estimate only.
                # Choose an action, according to the probability.
                chosen_rb = prob.multinomial(num_samples=1).detach()
                # Or choose action with highest probability.
                # chosen_rb = torch.argmax(actor_value[0, cfg.nrb:]).unsqueeze(0)
            else:  # Use throughput to choose RB.
                chosen_rb = torch.argmax(actor_value[0, 0:cfg.nrb] * actor_value[0, cfg.nrb:]).detach()

            if cfg.use_linear_policy_score:
                policy_score = map_tensor_to_range(actor_value[0, cfg.nrb:], range=(0, 1))
                chosen_policy = policy_score.gather(0, chosen_rb)
            else:  # Use previous code (log_prob as policy score).
                chosen_policy = log_prob.gather(0, chosen_rb)

            reward, done = eval_action(actor_value, y_rate[step + cfg.num_windows], y_rb[step + cfg.num_windows], cfg)

            done = done and cfg.train_break  # If done and train_break, break the episode.
            # If episode length exceeds the data length, break the episode.
            done = done or (step >= len(data_cqi) - cfg.num_windows)
            # If model is not RL-based, break the episode.
            if cfg.scheme in (2, 3, 4):
                done = True

            if done:
                # with lock:  # lock.acquire(), lock.release()
                #     counter.value += 1
                # Initialize env
                x, y = next(iter(data_loader))
                if cfg.data_start_from_zero:
                    start_idx = 0
                else:
                    start_idx = torch.randint(0, len(x[0][0]) - cfg.num_windows, (1,)).item()
                data_cqi = x[0][0][start_idx:].to(cfg.device)
                data_sensing = x[1][0][start_idx:].to(cfg.device)
                y_rate = y[0][0][start_idx:].to(cfg.device)
                y_rb = y[1][0][start_idx:].to(cfg.device)

            values.append(critic_value)
            rewards.append(reward)
            policies.append(chosen_policy)

            if done:
                break

        with lock:  # lock.acquire(), lock.release()
            counter.value += 1  # counter counts the number of total episodes.

        R = torch.zeros(1, 1).to(cfg.device)
        if not done:
            critic_value, _, _ = model(
                ((data_cqi[step:step + cfg.num_windows], data_sensing[step:step + cfg.num_windows]), (hx, cx)))
            R = critic_value.detach()

        # Calculate the loss after each episode.
        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1).to(cfg.device)
        """
        rewards[i]: R_i. The reward at time i.
        values[i]: V(S_i). The value of the state at time i.
        (outside for loop) R: V(S_T) The value of the last state.
        (inside for loop) R: G_i. The return or discounted future reward at time i.        
        """
        for i in reversed(range(len(rewards))):  # At time i.
            R = cfg.gamma * R + rewards[i]  # Calculate discounted future reward at time i.
            advantage = R - values[i]

            # Generalized Advantage Estimation
            delta_t = rewards[i] + cfg.gamma * values[i + 1] - values[i]
            gae = gae * cfg.gamma * cfg.lamda + delta_t

            # See A3C algorithm.
            policy_loss = policy_loss - policies[i] * gae - cfg.entropy_coef * entropies[i]  # Entropy is deprecated.
            value_loss = value_loss + 0.5 * advantage.pow(2)

        optimizer.zero_grad()
        if type((policy_loss + cfg.value_loss_coef * value_loss)) is not torch.Tensor:
            raise ValueError(f"Loss is not torch.Tensor.")
        (policy_loss + cfg.value_loss_coef * value_loss).backward()  # Debug; see loss ratio
        # if cfg.debug_verbose and torch.rand(1).item() < 0.01:  # Debug
        #     print(f"    DEBUGGING: {policy_loss.item()=:.2f}, {cfg.value_loss_coef*value_loss.item()=:.2f}, Equal value_loss_coef = {policy_loss.item()/value_loss.item()}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)  # See if gradient clipping is necessary.

        ensure_shared_grads(model, shared_model)

        optimizer.step()
        episode_idx += 1
        if episode_idx >= cfg.max_episodes:
            print(f"Rank {rank:02d} | Training finished. Max episodes reached ({cfg.max_episodes} episodes).")
            break

        # scheduler.step(loss_avg.avg)

    if early_stop.value == 1:
        print(f"Rank {rank:02d} | Early stopping")


def train_NN(cfg, optimizer):
    """
    Arguments:
        cfg: Configuration class. The configurations.
    Returns:
        None
    """
    set_seed(cfg.seed)
    if cfg.scheme == 2:
        model = Convolutional_v0(cfg).to(cfg.device)
    elif cfg.scheme == 3:
        model = FullyConnected_v0(cfg).to(cfg.device)
    else:
        raise ValueError
    
    start_time = time.time()

    dataset = UWADataset(snr=cfg.SNR, mode='train', data_dir='data24_'+cfg.data_version)
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    print(f"Train   | Data loaded")

    dataset = UWADataset(snr=cfg.SNR, mode='test', data_dir='data24_'+cfg.data_version)
    test_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    print(f"Test    | Data loaded")

    early_stopping = EarlyStopping(patience=cfg.patience, verbose=True, path=os.path.join(cfg.run_dir, 'model.pth'))

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # Learning rate scheduler
    if cfg.use_scheduler:
        if cfg.scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        elif cfg.scheduler == 'Lambda':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.85 ** epoch)
        elif cfg.scheduler == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        else:
            raise ValueError(f'Invalid scheduler {cfg.scheduler}')

    i = 0  # Iteration counter
    with open(os.path.join(cfg.run_dir, 'log.csv'), 'a') as f:
        for epoch in range(cfg.epochs):
            optimizer.zero_grad()
            model.train()

            x, y = next(iter(train_loader))
            i += 1
            """
            data_idx = torch.randint(0, len(x[0][0]), (1,)).item()
            data_cqi = x[0][0][data_idx].to(cfg.device)
            data_sensing = x[1][0][data_idx].to(cfg.device)
            y_rate = y[0][0][data_idx].to(cfg.device)
            y_rb = y[1][0][data_idx].to(cfg.device)
            """
            """  # Original code
            data_cqi = x[0][0].to(cfg.device)
            data_sensing = x[1][0].to(cfg.device)
            y_rate = y[0][0].to(cfg.device)
            y_rb = y[1][0].to(cfg.device)
            """
            data_cqi = x[0].to(cfg.device)
            data_sensing = x[1].to(cfg.device)
            y_rate = y[0].flatten(0, 1).to(cfg.device)
            y_rb = y[1].flatten(0, 1).to(cfg.device)

            output = model((data_cqi, data_sensing))

            if len(output.shape) == 2:  # Use batched data
                reward = 0
                for data_i in range(len(output)):
                    reward_slice, _ = eval_action(output[data_i], y_rate[data_i], y_rb[data_i], cfg)
                reward += reward_slice
            else:
                reward, _ = eval_action(output, y_rate, y_rb, cfg)

            loss = -reward  # Minimize loss
            
            loss.backward()

            optimizer.step()

            if (len(train_loader) * epoch + i) % cfg.eval_period == 0:
                model.eval()
                loss_avg = AverageMeter()  # Average loss
                rate_success_rate = AverageMeter()  # Rate success rate
                rb_success_rate = AverageMeter()  # RB success rate
                success_rate = AverageMeter()  # Success rate
                throughput_avg = AverageMeter()  # Average throughput
                meters = (rate_success_rate, rb_success_rate, success_rate, throughput_avg)
                meters_verbose = [AverageMeter() for _ in range(5)]

                # Use batch of time data instead of single time slice.
                for (x, y) in test_loader:
                    """
                    data_cqi = x[0][0].to(cfg.device)
                    data_sensing = x[1][0].to(cfg.device)
                    y_rate = y[0][0].to(cfg.device)
                    y_rb = y[1][0].to(cfg.device)
                    """
                    data_cqi = x[0].to(cfg.device)
                    data_sensing = x[1].to(cfg.device)
                    y_rate = y[0].flatten(0, 1).to(cfg.device)
                    y_rb = y[1].flatten(0, 1).to(cfg.device)

                    with torch.no_grad():
                        output = model((data_cqi, data_sensing))

                    if len(output.shape) == 2:  # Use batched data
                        reward = 0
                        for data_i in range(len(output)):
                            reward_slice, _ = eval_action(output[data_i], y_rate[data_i], y_rb[data_i], cfg, True,
                                                          meters, meters_verbose)
                        reward += reward_slice
                    else:
                        reward, _ = eval_action(output, y_rate, y_rb, cfg)

                    loss_avg.update(-reward)

                t = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start_time))

                # NC: Not Considered
                print(f"Time: {t} | Num Iterations: {len(train_loader) * epoch + i} | Average Test Episode Reward: {loss_avg.avg:.2f} | "
                      f"Test Episode Length: NC | Rate Success Rate: {rate_success_rate.avg:.2f} | "
                      f"RB Success Rate: {rb_success_rate.avg:.2f} | Success Rate: {success_rate.avg:.2f} | "
                      f"Average Throughput: {throughput_avg.avg:.2f}")
                f.write(f"{t},{len(train_loader) * epoch + i},{loss_avg.avg},"
                        f"{123},{rate_success_rate.avg},"
                        f"{rb_success_rate.avg},{success_rate.avg},{throughput_avg.avg},"
                        f"{meters_verbose[0].avg},{meters_verbose[1].avg},{meters_verbose[2].avg},{meters_verbose[3].avg}\n")

                scheduler.step(loss_avg.avg)
                try:
                    print(scheduler.get_last_lr())
                except Exception as e:
                    print(e)

                early_stopping(loss_avg.avg, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            # scheduler.step(loss_avg.avg)


if __name__ == "__main__":
    from configs import Config

    cfg = Config()
    shared_model = ActorCritic_v0(cfg).to(cfg.device)
    shared_model.share_memory()
    counter = torch.multiprocessing.Value('i', 0)
    lock = torch.multiprocessing.Lock()
    optimizer = None

    train(0, cfg, shared_model, counter, lock, optimizer)
    print("Training finished.")
