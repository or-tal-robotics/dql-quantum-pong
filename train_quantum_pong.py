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
import pandas as pd

MAX_EXPERIENCE = 50000
MIN_EXPERIENCE = 5000
TARGET_UPDATE_PERIOD = 50000
IM_SIZE = 84
K = 5
n_history = 4

class Statistics():
    q = 0
    qt = 0
    c = np.zeros((2,2))
            
def save_weights_stat(W_left, W_right):         
    df = pd.DataFrame([W_left])
    df.to_csv("weights_stats_left.csv",sep="\t", mode='a', header=False)
    df = pd.DataFrame([W_right])
    df.to_csv("weights_stats_right.csv",sep="\t", mode='a', header=False)

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
              record):
    
    t0 = datetime.now()
    obs = env.reset()
    obs_small_right = transform(obs[0])
    obs_small_left = transform(obs[1])
    state_right = np.stack([obs_small_right] * n_history, axis = 2)
    state_left = np.stack([obs_small_left] * n_history, axis = 2)
    
    total_time_training = 0
    num_steps_in_episode = 0
    episode_reward = [0,0]
    quantum_button = [0,0]
    quantum_button_dual = 0      
    # if np.random.binomial(1,epsilon)==1:
    #     use_force = True
    # else:
    #     use_force = False
    done = False
    if record == True:
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), 20.0, (640,480))
    while not done:
        
        if total_t % TARGET_UPDATE_PERIOD == 0:
            for ii in range(2):
                target_model[ii].copy_from(model[ii])
            print("model is been copied!")
        action_left = model[1].sample_action(state_left, epsilon)
        action_right = model[0].sample_action(state_right, epsilon)
        # if use_force == True:
        #     if env.QP.right_player.theta_ent < np.pi/4:
        #         action_right = 4
        #     elif env.QP.right_player.theta_ent > np.pi/4:
        #         action_right = 5
        #     elif env.QP.right_player.theta_mes1 > np.pi/2:
        #         action_right = 1
        #     elif env.QP.right_player.theta_mes1 < np.pi/2:
        #         action_right = 0
        #     elif env.QP.right_player.theta_mes2 > 0:
        #         action_right = 3
        #     elif env.QP.right_player.theta_mes2 < 0:
        #         action_right = 2


        obs, reward, done, hit = env.step([action_right,action_left])

        obs_small_right = transform(obs[0])
        obs_small_left = transform(obs[1])
        next_state_left = update_state(state_left, obs_small_left)
        next_state_right = update_state(state_right, obs_small_right)
        t0_2 = datetime.now()
        
        for ii in range(2):
            if reward[ii]>0:
                episode_reward[ii] += reward[ii]

        experience_replay_buffer[0].add_experience(action_right, obs_small_right, reward[0], done)
        learn(model[0], target_model[0], experience_replay_buffer[0], gamma, batch_size)  
        experience_replay_buffer[1].add_experience(action_left, obs_small_right, reward[1], done)
        learn(model[1], target_model[1], experience_replay_buffer[1], gamma, batch_size)
        
        dt = datetime.now() - t0_2
        
        total_time_training += dt.total_seconds()
        num_steps_in_episode += 1
        
        state_left = next_state_left
        state_right = next_state_right
        total_t += 1
        epsilon = max(epsilon - epsilon_change, epsilon_min)
        if record == True:
            frame = cv2.cvtColor(obs_small_right, cv2.COLOR_GRAY2BGR)
            frame = cv2.resize(frame,(640,480))
            org = (50, 20) 
            font = cv2.FONT_HERSHEY_SIMPLEX
            # fontScale 
            fontScale = 1
               
            # Blue color in BGR 
            color = (255, 0, 0) 
              
            # Line thickness of 2 px 
            thickness = 2
            frame = cv2.putText(frame, str(episode_reward[0]), org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
            out.write(frame)
            #cv2.imshow("frame", frame)
    
    if record == True:
        out.release()
        
    quantum_button[0] = quantum_button[0]/num_steps_in_episode
    quantum_button[1] = quantum_button[1]/num_steps_in_episode
    quantum_button_dual = quantum_button_dual/num_steps_in_episode
    return total_t, episode_reward, (datetime.now()-t0), num_steps_in_episode, total_time_training/num_steps_in_episode, epsilon, quantum_button, quantum_button_dual

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
    num_episodes = 5000
    total_t = 0
    experience_replay_buffer = [ReplayMemory(),ReplayMemory()]
    episode_rewards = np.zeros((2,num_episodes))
    episode_lens = np.zeros(num_episodes)
    train_idxs = [0,1]
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) / 1000000
    quantum_buttons = np.zeros((2,num_episodes))
    quantum_button_duals = np.zeros(num_episodes)
    env = gym.make('gym_quantum_pong:Quantum_Pong-v0', mode = "quantum")
    

    left_player_model = DQN(
                K = 5,
                image_size=IM_SIZE
                )
    
    left_player_model_target = DQN(
                K = 5,
                image_size=IM_SIZE
                )
    
    right_player_model = DQN(
                K = 7,
                image_size=IM_SIZE
                )
    
    right_player_model_target = DQN(
                K = 7,
                image_size=IM_SIZE
                )
    
    
    
    # print(left_player_model.get_weights()[-2].shape)
    


    print("Initializing experience replay buffer...")
    obs = env.reset()
    
    for i in range(MIN_EXPERIENCE):
        action = [np.random.choice(K),np.random.choice(K)]
        obs, reward, done, _ = env.step(action)
        obs_small_right = transform(obs[0])
        obs_small_left = transform(obs[1])
        experience_replay_buffer[0].add_experience(action[0], obs_small_right, reward[0], done)
        experience_replay_buffer[1].add_experience(action[1], obs_small_left, reward[1], done)
        
        if done:
            obs = env.reset()
            
    t0 = datetime.now()
    record = True
    episode_reward = [0,1]
    for i in range(num_episodes):
        video_path = 'video/Episode_'+str(i)+'.avi'
        if i%100 == 0:
            record = True
        else:
            record = False

        
        
            
      
            
        total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon, quantum_button, quantum_button_dual = play_ones(
                env,
                total_t,
                experience_replay_buffer,
                [right_player_model, left_player_model],
                [right_player_model_target, left_player_model_target],
                gamma,
                batch_sz,
                epsilon,
                epsilon_change,
                epsilon_min,
                video_path,
                record)
        
       
        for ii in range(2):
            episode_rewards[ii,i] = episode_reward[ii]/200.0
        
        episode_lens[i] = num_steps_in_episode
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
        sys.stdout.flush()
    print("Total duration:", datetime.now()-t0)
    
    
    y1 = smooth(episode_rewards[0,:i])
    
    y2 = smooth(episode_rewards[1,:i])
    plt.plot(y1)
    plt.plot(y2)

    env.close()
    
     
    
        
       
        
        
        
    
    
    
    
    
    
        
    
