
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image
from celluloid import Camera

def speed_gameplay(path,  speed_factor):
    with Image.open(path) as gif:
        frames = []
        new_duration = []

        for frame in range(gif.n_frames):
            gif.seek(frame)
            frames.append(gif.copy())
            new_duration.append(int(gif.info['duration'] / speed_factor))

        frames[0].save(path, save_all=True, append_images=frames[1:], 
                       optimize=False, duration=new_duration, loop=0)

class CartPoleAgent(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super().__init__()
        self.num_actions = num_actions
        self.linear_input = nn.Linear(num_inputs, hidden_size)
        self.hidden_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_output = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, observation):
        x = self.linear_input(observation)
        x = self.hidden_1(x)
        x = self.linear_output(x)
        return torch.softmax(x, dim=1)

    def get_action(self, observation):
        observation = torch.from_numpy(
            observation
        ).float().unsqueeze(0)
        probabilities = self.forward(observation)
        action = probabilities.multinomial(num_samples=1)
        return action.item(), probabilities[:, action.item()]
    
def time_decay_returns(rewards, gamma=0.99):
    output = []
    for index_reward in range(rewards.size):
        intermediate = gamma ** np.arange(0,rewards[index_reward:].size)
        intermediate = rewards[index_reward:] * intermediate
        output.append(intermediate.sum())
    output = torch.tensor(output)
    output = output - (output.mean()/(output.std()+1e-9))
    return output

def policy_loss(log_probabilities, returns):
    output = torch.stack(
        [-lg_p * r for lg_p, r in zip(log_probabilities, returns)]
    ).sum()
    return output

def update_policy(policy, log_probabilities, rewards, gamma=0.99):
    returns = time_decay_returns(rewards, gamma)
    policy.optimizer.zero_grad()
    loss = policy_loss(log_probabilities, returns)
    loss.backward()
    policy.optimizer.step()

def get_trajectory(env, policy, make_gif=False,gif_path=None,fps=60):
    if make_gif:
        fig, ax = plt.subplots(figsize=(6,4))
        camera = Camera(fig)


    observation, info = env.reset()
    
    
    actions = []
    rewards = []
    observations = []
    log_probabilities = []
    
    episode_over = False
    n=0
    while not episode_over:
        action, probability = policy.get_action(observation) 
        observation, reward, terminated, truncated, info = env.step(action)
            
        actions.append(action)
        rewards.append(reward)
        observations.append(observation)
        log_probabilities.append(torch.log(probability))
    
        if make_gif:
            ax.imshow(env.render())
            camera.snap()
        episode_over = terminated or truncated
        n+=1
    
    if make_gif:    
        for i in range(15):
            ax.imshow(np.zeros(env.render().shape))
            camera.snap()
        animation = camera.animate()
        animation.save(gif_path,fps=fps)
        plt.close()
        
        
    return (
        np.array(actions),
        np.array(rewards),
        np.array(observations),
        log_probabilities
    )