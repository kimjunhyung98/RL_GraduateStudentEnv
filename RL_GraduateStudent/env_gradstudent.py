# 서울대학교 김준형 제작

import os
import numpy as np
import pandas as pd
import gymnasium as gym
import time
import matplotlib.pyplot as plt

from gymnasium import spaces

class GradStudentEnv(gym.Env):

    def __init__(self, render_mode='Human', device='cpu'):
        self.device = device # 이건 torch등을 써서 DNN 등을 쓸 때 필요할 것이다
        self.render_mode = render_mode

        # DEFINE ACTION AND OBSERVATION SPACE
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict({
            "knowledge": spaces.Box(low=0.0, high=200.0, shape=(1,), dtype=np.float32),
            "stress": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            "time": spaces.Box(low=0.0, high=400, shape=(1,), dtype=np.float32)
        })

    def reset(self, seed=None, options=None):

        # INITIALIZE STATE
        self.knowledge = 10
        self.stress = 10
        self.time_elapsed = 0
        self.max_steps = 100
        self.status = 'enrolled'

        # TRACKING HISTORY DURING EPISODE
        self.history_location = np.full(3, 0)
        self.history_knowledge = []
        self.history_stress = []

        # OBSERVATION AND INFO AFTER RESET
        observation = {
            "knowledge": np.array([self.knowledge], dtype=np.float32),
            "stress": np.array([self.stress], dtype=np.float32),
            "time": np.array([self.time_elapsed], dtype=np.float32)
        }

        info = {
            "status": self.status,
            "history_location": self.history_location,
            "history_knowledge": np.array(self.history_knowledge, dtype=np.float32),
            "history_stress": np.array(self.history_stress, dtype=np.float32)
        }

        # DEBUG MODE FLAG
        self.debug_mode = False  # 만약 코드가 이상하다면, print문에 원하는 내용을 넣어서 확인해보자.

        if self.debug_mode: # 원하는 내용을 아무거나 넣어서 확인하자. 위치를 바꿀 수도 있다.
            print(observation)

        # FOR SAVING PLOT IMAGES
        if options == 'SaveVideo':
            self.save_video = True
            self.frame_id = 0
            timestamp = time.strftime("frames_%y%m%d_%H%M")  # 폴더명
            self.folder_name = f"./{timestamp}"
            # Create the folder if it doesn't exist
            os.makedirs(self.folder_name, exist_ok=True)
        else:
            self.save_video = False

        return observation, info
    
    def step(self, action):

        # ACTION
        if action == 0:  # Go to Lab
            self.knowledge  += np.random.randint(1, 4)
            self.stress     += np.random.randint(1, 3)
            self.history_location[0] += 1
        elif action == 1:  # Go Home and Sleep
            self.knowledge  += np.random.randint(0, 1)
            self.stress     += np.random.randint(-4, 0)
            self.history_location[1] += 1
        elif action == 2:  # Go to Karaoke
            self.knowledge  += np.random.randint(-2, 0)
            self.stress     += np.random.randint(-10, -5)
            self.history_location[2] += 1

        # STRESS AFFECTS PERFORMANCE
        if self.stress > 70:
            self.knowledge -=5  # Reduce knowledge by 20%
        elif self.stress > 50:
            self.knowledge -=2  # Reduce knowledge by 10%
        elif self.stress > 30:            
            self.knowledge -=np.random.randint(0, 1)  # Reduce knowledge by 5%

        # CLIP VALUES AND RECORD HISTORY
        self.knowledge = np.clip(self.knowledge, 0, 200)
        self.stress = np.clip(self.stress, 0, 100)
        self.history_knowledge.append(self.knowledge)
        self.history_stress.append(self.stress)

        # INCREASE TIME
        self.time_elapsed += 1

        # DONE ?
        ## Check if the student should be expelled
        if self.knowledge >150:
            self.max_steps = 300
        elif self.knowledge >100:
            self.max_steps = 200
        elif self.knowledge >50:
            self.max_steps = 150

        ## Check if done
        if self.time_elapsed >= self.max_steps:
            self.status = 'expelled'
        elif self.stress >= 100 :
            self.status = 'burned out'
        elif self.knowledge >= 200 :
            self.status = 'graduated'

        terminated = True if self.status == 'graduated' else False
        truncated = True if self.status in ['expelled', 'burned out'] else False

        # REWARD
        reward = 0.1*np.clip(self.knowledge - 0.5*self.stress, 0, 1000)
        if terminated:
            reward += reward*(300 - self.time_elapsed)+1000/self.time_elapsed
        elif truncated:
            reward = 0

        # OBSERVATION AND INFO
        observation = {
            "knowledge": np.array([self.knowledge], dtype=np.float32),
            "stress": np.array([self.stress], dtype=np.float32),
            "time": np.array([self.time_elapsed], dtype=np.float32)
        }

        info = {
            "status": self.status,
            "history_location": self.history_location,
            "history_knowledge": np.array(self.history_knowledge, dtype=np.float32),
            "history_stress": np.array(self.history_stress, dtype=np.float32)
        }

        # DEBUG PRINT
        if self.debug_mode: # 원하는 내용을 아무거나 넣어서 확인하자. 위치를 바꿀 수도 있다.
            print(f"Action: {action} | Time: {self.time_elapsed} | Knowledge: {self.knowledge:.2f} | Stress: {self.stress:.2f} | Status: {self.status} | Reward: {reward}")
            print(observation)
            print(self.history_location)

        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == 'human': # show graph with matplotlib
            return self._render_frame()
        
        elif self.render_mode == 'text': # show text 
            print(f"Time: {self.time_elapsed} | Knowledge: {self.knowledge:.2f} | Stress: {self.stress:.2f} | Status: {self.status}")

    def _render_frame(self):
        plt.clf()
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(2, 1)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title("Grad Student Progress")
        timesteps = np.arange(len(self.history_knowledge))
        ax1.plot(timesteps, self.history_knowledge,  label='Knowledge')
        ax1.plot(timesteps, self.history_stress,  label='Stress')
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Levels")
        ax1.legend()
        ax1.grid(True)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_title("Location History")
        locations = ['Lab', 'Home/Sleep', 'Karaoke']
        bars = ax2.bar(locations, self.history_location)
        ax2.bar_label(bars)
        ax2.set_ylabel("Visits")
        plt.subplots_adjust(hspace=0.35)
        if self.save_video:
            figure_name = f"{self.folder_name}/frame_{self.frame_id:04d}.png"
            plt.savefig(figure_name)
            print(f'figure saved to {figure_name}\n')
            self.frame_id += 1

       
        plt.pause(0.1)