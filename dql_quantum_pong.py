import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from imagetranformer import transform
from rl_common import ReplayMemory, update_state, learn
from dqn_model import DQN
from gym import wrappers
import cv2
import time


MAX_EXPERIENCE = 50000
MIN_EXPERIENCE = 5000
TARGET_UPDATE_PERIOD = 50000
IM_SIZE = 84
K = 4
n_history = 4
MAX_STEPS_PER_EPSIODE = 50000

class Statistics():
    q = 0
    qt = 0
    c = np.zeros((2,2))
            

        


def play_ones(env,
              total_t,
              experience_replay_buffer,
              model,
              target_model,
              gamma,
              batch_size,
              epsilon,
              epsilon_change,
              epsilon_min,
              pathOut,
              record,
              train_idxs):
    
    t0 = datetime.now()
    obs = env.reset()
    print(obs.shape)
    obs_small = transform(obs, [IM_SIZE, IM_SIZE])
    state = np.stack([obs_small] * n_history, axis = 2)
    loss = [None, None]
    
    total_time_training = 0
    num_steps_in_episode = 0
    episode_reward = [0,0]
    quantum_button = [0,0]
    quantum_button_dual = 0
    done = False
    if record == True:
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), 20.0, (640,480))
    while not done:
        
        if total_t % TARGET_UPDATE_PERIOD == 0:
            for ii in range(2):
                target_model[ii].copy_from(model[ii])
            print("model is been copied!")
        action = []
        for ii in range(2):
            action.append(model[ii].sample_action(state, epsilon))
            if action[ii] > 2:
                quantum_button[ii] += 1
        if (action[0]==3 and action[1]==3) or (action[0]==0 and action[1]==0):
            quantum_button_dual += 1
        obs, reward, done, _ = env.step(action)
        obs_small = transform(obs,  [IM_SIZE, IM_SIZE])
        next_state = update_state(state, obs_small)
        t0_2 = datetime.now()
        
        for ii in range(2):
            episode_reward[ii] += reward[ii]
        for ii in train_idxs:
            experience_replay_buffer[ii].add_experience(action[ii], obs_small, reward[ii], done)
            loss = learn(model[ii], target_model[ii], experience_replay_buffer[ii], gamma, batch_size)
        
        
        dt = datetime.now() - t0_2
        
        total_time_training += dt.total_seconds()
        num_steps_in_episode += 1
        
        state = next_state
        total_t += 1
        epsilon = max(epsilon - epsilon_change, epsilon_min)
        if record == True:
            frame = cv2.cvtColor(obs_small, cv2.COLOR_GRAY2BGR)
            frame = cv2.resize(frame,(640,480))
            out.write(frame)
            #cv2.imshow("frame", frame)
        statistics = Statistics()    
        c, qs, qst = env.statistics()
        statistics.c = c
        statistics.q = qs
        statistics.qt = qst
    if record == True:
        out.release()
        
    quantum_button[0] = quantum_button[0]/num_steps_in_episode
    quantum_button[1] = quantum_button[1]/num_steps_in_episode
    quantum_button_dual = quantum_button_dual/num_steps_in_episode
    return total_t, episode_reward, (datetime.now()-t0), num_steps_in_episode, total_time_training/num_steps_in_episode, epsilon, quantum_button, quantum_button_dual, statistics

def smooth(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i-99)
        y[i] = float(x[start:(i+1)].sum())/(i-start+1)
    return y
        

if __name__ == '__main__':
    plot_flag = False
    gamma = 0.99
    batch_sz = 32
    num_episodes = 3000
    total_t = 0
    experience_replay_buffer = [ReplayMemory(),ReplayMemory()]
    episode_rewards = np.zeros((2,num_episodes))
    episode_lens = np.zeros(num_episodes)
    train_idxs = [0,1]
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) / 50000
    quantum_buttons = np.zeros((2,num_episodes))
    quantum_button_duals = np.zeros(num_episodes)
    env = gym.make('gym_quantum_pong:Quantum_Pong-v0')
    
    #monitor_dir = 'video'
    #env = wrappers.Monitor(env, monitor_dir)
    model = []
    stats = []
    target_model = []
    for ii in range(2):
        model.append(DQN(
                K = K,
                image_size=IM_SIZE
                ))
        
        target_model.append(DQN(
                K = K,
                image_size=IM_SIZE
                ))
    
    
    
    
    print("Initializing experience replay buffer...")
    obs = env.reset()
    
    for i in range(MIN_EXPERIENCE):
        action = [np.random.choice(K),np.random.choice(K)]
        obs, reward, done, _ = env.step(action)
        obs_small = transform(obs,  [IM_SIZE, IM_SIZE])
        experience_replay_buffer[0].add_experience(action[0], obs_small, reward[0], done)
        experience_replay_buffer[1].add_experience(action[1], obs_small, reward[1], done)
        
        if done:
            obs = env.reset()
            
    t0 = datetime.now()
    record = True
    episode_reward = [0,1]
    skip_intervel = 5
    lr = 6e-6
    for i in range(num_episodes):
        video_path = 'video/Episode_'+str(i)+'.avi'
        if i%100 == 0:
            record = True
        else:
            record = False

        if i % skip_intervel == 0:
            if train_idxs == [0]:
                train_idxs = [1]
            else:
                train_idxs = [0]
        
    
            
        total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon, quantum_button, quantum_button_dual, statistics = play_ones(
                env,
                total_t,
                experience_replay_buffer,
                model,
                target_model,
                gamma,
                batch_sz,
                epsilon,
                epsilon_change,
                epsilon_min,
                video_path,
                record,
                train_idxs)
        
        stats.append(statistics)
        for ii in range(2):
            episode_rewards[ii,i] = episode_reward[ii]
            quantum_buttons[ii,i] = quantum_button[ii]
        episode_lens[i] = num_steps_in_episode
        quantum_button_duals[i] = quantum_button_dual
        last_100_avg1 = episode_rewards[0,max(0,i-100):i+1].mean()
        last_100_avg2 = episode_rewards[1,max(0,i-100):i+1].mean()
        print("Episode:", i ,
              "training:", train_idxs, 
              "Duration:", duration,
              "Num steps:", num_steps_in_episode,
              "Reward 1:", episode_reward[0],
              "Reward 2:", episode_reward[1],
              "Training time per step:", "%.3f" %time_per_step,
              "Avg Reward 1:", "%.3f"%last_100_avg1,
              "Avg Reward 2:", "%.3f"%last_100_avg2,
              "Epsilon:", "%.3f"%epsilon)
        print(statistics.c, statistics.q, statistics.qt)
        sys.stdout.flush()
    print("Total duration:", datetime.now()-t0)
    
    
    y1 = smooth(episode_rewards[0,:i])
    y2 = smooth(episode_rewards[1,:i])
    b1 = smooth(quantum_buttons[0,:i])
    b2 = smooth(quantum_buttons[1,:i])
    qbd = smooth(quantum_button_duals[:i])
    C = np.empty((len(stats),6))    
    for ii in range(len(stats)):
        if stats[ii].c is None:
            C[ii,:] = 0
        else:
            C[ii,:4] = stats[ii].c.reshape(-1)
            C[ii,4] = stats[ii].q
            C[ii,5] = stats[ii].qt
            
    statistics_data = (y1, y2, b1, b2, qbd)
    statistics_data = np.array(statistics_data)
    np.save("stat_quantum_choice_29092019(60).npy", statistics_data)
    np.save("stat_quantum_choice_C_29092019(60).npy", C)
    env.close()
    
     
        
        
       
        
        
        
    
    
    
    
    
    
        
    
