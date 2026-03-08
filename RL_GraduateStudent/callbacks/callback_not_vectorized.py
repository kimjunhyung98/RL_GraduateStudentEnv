import gymnasium as gym
import os

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback

import datetime


class RewardPlotCallback(BaseCallback):
    def __init__(self, plot_freq=1000, verbose=0):
        super().__init__(verbose)
        self.plot_freq = plot_freq
        self.episode_rewards = []
        self.episode_rewards_avg = []
        self.current_episode_reward = 0.0
        self.time_start = datetime.datetime.now()

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]  # assumes single env
        done = self.locals["dones"][0]
        self.current_episode_reward += reward
 
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_rewards_avg.append(np.mean(self.episode_rewards[-20:]))
            self.current_episode_reward = 0.0

        if self.n_calls % self.plot_freq == 0 and self.episode_rewards:
            self._plot_rewards()
            self.training_env.envs[0].render()
        return True


    def _plot_rewards(self):
        clear_output(True)
        plt.clf()
        self.time_now = datetime.datetime.now()
        time_diff = self.time_now - self.time_start
        time_diff = time_diff.total_seconds()
        hours = int(time_diff // 3600)
        minutes = int((time_diff % 3600) // 60)
        textstr = (
            f'Start Time  : {self.time_start.strftime("%Y-%m-%d, %H:%M:%S")}\n'
            f'Learning Duration  : {hours} h {minutes} min\n'
            f'current num_timesteps     : {self.num_timesteps}'                
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)

        if len(self.episode_rewards)>2500:

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns, custom size

            # Plot on the first subplot

            ax1.plot(self.episode_rewards, label="Episode reward")
            ax1.plot(self.episode_rewards_avg, label="Episode reward Average")
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Reward")
            ax1.set_title("Reward Curve")
            ax1.text(0.25, -0.15, textstr, transform=ax1.transAxes, fontsize=10, horizontalalignment='left', verticalalignment='top', bbox=props)
            ax1.grid(True)
            ax1.legend()

            # Plot on the second subplot
            ax2.plot(self.episode_rewards, label="Episode reward")
            ax2.plot(self.episode_rewards_avg, label="Episode reward Average")
            ax2.set_xlim(len(self.episode_rewards)-300, len(self.episode_rewards))
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Reward")
            ax2.set_title("Reward Curve(Zoomed In)")
            ax2.grid(True)
            ax2.legend()

            # Adjust layout and display the plot
            plt.tight_layout()

        else :
            plt.plot(self.episode_rewards, label="Episode reward")
            plt.plot(self.episode_rewards_avg, label="Episode reward Average")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.text(0.25, -0.15, textstr, transform=plt.gca().transAxes, fontsize=10, horizontalalignment='left', verticalalignment='top', bbox=props)
            plt.title("Reward Curve")
            # if np.min(self.episode_rewards[-100:])>-1000:
            #     plt.ylim(-1000, 0)
            # if np.min(self.episode_rewards[-100:])>-100:
            #     plt.ylim(-100, 0)
            plt.grid(True)
            plt.legend()
        plt.pause(0.001)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f'./LearnedModels_{timestamp}/'
os.makedirs(save_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=save_dir,
    name_prefix="rl_model",
)

class CombinedCallback(BaseCallback):
    def __init__(self, plot_freq=1000, save_freq=10000, save_path=save_dir, verbose=0):
        super().__init__(verbose)
        self.plot_freq = plot_freq
        self.save_freq = save_freq
        self.save_path = save_path
        
        # Plotting attributes
        self.episode_rewards = []
        self.episode_rewards_avg = []
        self.current_episode_reward = 0.0
        self.time_start = datetime.datetime.now()


    def _init_callback(self) -> None:
        num_envs = self.model.get_env().num_envs
        self.current_episode_reward = np.zeros(num_envs)
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        self.current_episode_reward += rewards

        for i, done in enumerate(dones):
            if done:
                self.episode_rewards.append(self.current_episode_reward[i])
                self.episode_rewards_avg.append(np.mean(self.episode_rewards[-20:]))
                self.current_episode_reward[i] = 0.0

        # Plotting logic
        if self.n_calls % self.plot_freq == 0 and self.episode_rewards:
            self._plot_rewards()
        
        # Model saving logic
        if self.n_calls % self.save_freq == 0:
            if self.num_timesteps<1_000:
                path = os.path.join(self.save_path, f'rl_model_{self.num_timesteps}_steps')
            else:
                path = os.path.join(self.save_path, f'rl_model_{round(self.num_timesteps/1_000)}k_steps')
                self.model.save(path)
            if self.verbose > 0:
                print(f"Saving model to {path}")
                
        return True
    
    def _plot_rewards(self):
        clear_output(True)
        plt.clf()
        self.time_now = datetime.datetime.now()
        time_diff = self.time_now - self.time_start
        time_diff = time_diff.total_seconds()
        hours = int(time_diff // 3600)
        minutes = int((time_diff % 3600) // 60)
        textstr = (
            f'Start Time  : {self.time_start.strftime("%Y-%m-%d, %H:%M:%S")}\n'
            f'Learning Duration  : {hours} h {minutes} min\n'
            f'current num_timesteps     : {self.num_timesteps}'                
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)

        if len(self.episode_rewards)>2500:

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns, custom size

            # Plot on the first subplot

            ax1.plot(self.episode_rewards, label="Episode reward")
            ax1.plot(self.episode_rewards_avg, label="Episode reward Average")
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Reward")
            ax1.text(0.25, -0.15, textstr, transform=ax1.transAxes, fontsize=10, horizontalalignment='left', verticalalignment='top', bbox=props)
            ax1.set_title("Reward Curve")
            ax1.grid(True)
            ax1.legend()

            # Plot on the second subplot
            ax2.plot(self.episode_rewards, label="Episode reward")
            ax2.plot(self.episode_rewards_avg, label="Episode reward Average")
            ax2.set_xlim(len(self.episode_rewards)-300, len(self.episode_rewards))
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Reward")
            ax2.set_title("Reward Curve(Zoomed In)")
            ax2.grid(True)
            ax2.legend()

            # Adjust layout and display the plot
            plt.tight_layout()

        else :
            plt.plot(self.episode_rewards, label="Episode reward")
            plt.plot(self.episode_rewards_avg, label="Episode reward Average")
            plt.text(0.25, -0.15, textstr, transform=plt.gca().transAxes, fontsize=10, horizontalalignment='left', verticalalignment='top', bbox=props)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("Reward Curve")
            # if np.min(self.episode_rewards[-100:])>-1000:
            #     plt.ylim(-1000, 0)
            # if np.min(self.episode_rewards[-100:])>-100:
            #     plt.ylim(-100, 0)
            plt.grid(True)
            plt.legend()
        plt.pause(0.001)