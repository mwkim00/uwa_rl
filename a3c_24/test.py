import time
import os
from collections import deque

import torch
from torch.utils.data import DataLoader

from model import ActorCritic_v0, FullyConnected_v0
from data_loader import UWADataset
from utils import set_seed, eval_action, AverageMeter


def test(cfg, shared_model, counter, early_stop, run_idx):
    """
    Arguments:
        cfg: Configuration class. The configurations.
        shared_model: ActorCritic_v0 class. The model to test.
        counter: mp.Value. The shared counter.
        early_stop: mp.Value. The shared early stop flag.
        run_idx: str. The index of the run. Used to find the log file.
    Returns:
        None
    """
    # TODO: replace run_idx with cfg.run_dir
    set_seed(cfg.seed + cfg.num_processes)

    model = ActorCritic_v0(cfg).to(cfg.device)
    model.eval()

    dataset = UWADataset(snr=cfg.SNR, mode='test', ratio=cfg.data_ratio, data_dir='data24_'+cfg.data_version)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"Test    | Data loaded")

    step = 0  # Starting index of the data window.

    prev_counter_val = 0
    early_stop_counter = 0  # Used to stop training early when sufficiently converged.

    max_reward_avg = -torch.inf

    # Counter and metrics used inside only inside this function.
    reward_sum = 0
    rate_success_rate = AverageMeter()
    rb_success_rate = AverageMeter()
    success_rate = AverageMeter()
    avg_throughput = AverageMeter()
    avg_steps = AverageMeter()
    reward_meter = AverageMeter()
    meters = (rate_success_rate, rb_success_rate, success_rate, avg_throughput)

    meters_verbose = (AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter())
    """
    # Additional meters for verbose mode.
    if cfg.debug_verbose:
        # reward_rate_dist, reward_rate_max, reward_RB_ideal, reward_RB_avail, reward
        meters_verbose = (AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter())
    else:
        meters_verbose = None
    """

    start_time = time.time()

    actions = deque(maxlen=cfg.deque_maxlen)

    with open(os.path.join(run_idx, 'log.csv'), 'a') as f:
        while True:
            if counter.value == prev_counter_val and counter.value != 0:
                print(f"Test finished. Counter: {counter.value}\n"
                      f"Results saved at {os.path.join(run_idx, 'log.csv')}.\n")
                break
            prev_counter_val = counter.value

            for i, (x, y) in enumerate(data_loader):  # Test through all test data.
                if cfg.data_start_from_zero:
                    start_idx = 0
                else:
                    start_idx = torch.randint(0, len(x[0][0]) - cfg.num_windows, (1,)).item()
                data_cqi = x[0][0][start_idx:].to(cfg.device)
                data_sensing = x[1][0][start_idx:].to(cfg.device)
                y_rate = y[0][0][start_idx:].to(cfg.device)
                y_rb = y[1][0][start_idx:].to(cfg.device)
                # Start of an episode, sync with the shared model.
                model.load_state_dict(shared_model.state_dict())
                cx = torch.zeros(cfg.num_lstm_layers, 1, cfg.hidden_size).to(cfg.device)
                hx = torch.zeros(cfg.num_lstm_layers, 1, cfg.hidden_size).to(cfg.device)

                for step in range(len(data_cqi) - cfg.num_windows):  # step: episode length
                    with torch.no_grad():
                        critic_value, actor_value, (hx, cx) = model(((data_cqi[step:step + cfg.num_windows],
                                                                      data_sensing[step:step + cfg.num_windows]),
                                                                     (hx, cx)))  # Send state to device.
                    if cfg.rb_estimate_only:
                        chosen_rb = torch.argmax(actor_value[0, cfg.nrb:])
                    else:
                        chosen_rb = torch.argmax(actor_value[0, 0:cfg.nrb] * actor_value[0, cfg.nrb:])

                    reward, done = eval_action(actor_value, y_rate[step + cfg.num_windows],
                                               y_rb[step + cfg.num_windows], cfg,
                                               True, meters, meters_verbose)

                    # If model is not RL-based, break the episode.
                    if cfg.scheme in (2, 3, 4):
                        done = True

                    reward_sum += reward
                    reward_meter.update(reward)

                    actions.append(chosen_rb.item())

                    if not done:
                        step += 1

                    if actions.count(actions[0]) == actions.maxlen:
                        # if cfg.debug_verbose:
                        #     print(f"Agent stuck at {actions[0]} for {actions.maxlen} steps. Setting done = True")
                        done = True

                    if done and cfg.test_break:
                        break  # Break. Test ends here. If not, don't break and test through all test data.

                    if step >= len(data_cqi) - cfg.num_windows:
                        break  # Break if the episode length exceeds the data length.

                avg_steps.update(step)

            # Tested through all test data.
            t = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start_time))

            if cfg.debug_verbose:
                print(f"{meters_verbose[0].avg} | {meters_verbose[1].avg} | {meters_verbose[2].avg} | "
                      f"{meters_verbose[3].avg} | {meters_verbose[4].avg}")

            print(f"Time: {t} | Num Episodes: {counter.value} | Average Test Episode Reward: {reward_meter.avg:.2f} | "
                  f"Test Episode Length: {avg_steps.avg:.2f} | Rate Success Rate: {rate_success_rate.avg:.2f} | "
                  f"RB Success Rate: {rb_success_rate.avg:.2f} | Success Rate: {success_rate.avg:.2f} | "
                  f"Average Throughput: {avg_throughput.avg:.2f}")
            f.write(f"{t},{counter.value},{reward_meter.avg},"
                    f"{avg_steps.avg},{rate_success_rate.avg},"
                    f"{rb_success_rate.avg},{success_rate.avg},{avg_throughput.avg},"
                    f"{meters_verbose[0].avg},{meters_verbose[1].avg},{meters_verbose[2].avg},{meters_verbose[3].avg}\n")

            # Compare with previous result and save model
            if reward_meter.avg > max_reward_avg:
                max_reward_avg = reward_meter.avg
                torch.save(model.state_dict(), os.path.join(cfg.run_dir, 'model.pth'))
                print(f"Model saved. Max average reward: {max_reward_avg:.2f}")
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # Initialize env
            reward_sum = 0
            actions.clear()
            step = 0
            rate_success_rate.reset()
            rb_success_rate.reset()
            success_rate.reset()
            avg_throughput.reset()

            time.sleep(cfg.sleep_time)

            # if cfg.early_stop = False: Early stop is not used.
            if cfg.use_early_stop and early_stop_counter >= cfg.patience:
                print(f"No improvement for {cfg.patience} episodes.\n"
                      f"Test finished. Counter: {counter.value}\n"
                      f"Results saved at {os.path.join(run_idx, 'log.csv')}.\n")
                early_stop.value = 1
                break


if __name__ == '__main__':
    from configs import Config

    cfg = Config()
    cfg.run_dir = f"runs/test"
    # os.makedirs(cfg.run_dir)
    model = ActorCritic_v0(cfg)
    shared_model = ActorCritic_v0(cfg)
    shared_model.share_memory()
    counter = torch.multiprocessing.Value('i', 0)
    run_idx = 'test'

    test(cfg, shared_model, counter, run_idx)
