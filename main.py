import gym
import numpy as np
import random
import pandas as pd
import collections
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque

class DQN:
    def __init__(self, env, learning, gamma, decay):
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=1000)
        self.batch_size = 32
        self.gamma = gamma
        self.decay = decay
        self.learning_rate = learning
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, s, a, r, s_prime, done):
        self.memory.append([s, a, r, s_prime, done])

    def replay(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        s = np.vstack([batch[0] for batch in mini_batch])
        a = np.vstack([batch[1] for batch in mini_batch])
        r = np.vstack([batch[2] for batch in mini_batch])
        s_prime = np.vstack([batch[3] for batch in mini_batch])
        done = np.vstack([batch[4] for batch in mini_batch]).astype(np.uint8)
        not_done = np.where(done == False)[0]
        new_state_est = self.model.predict(s_prime)
        target = self.target_model.predict(s_prime)
        if len(done[0]) > 0:
            r[not_done] += np.multiply(self.gamma, target[not_done, np.argmax(new_state_est[not_done], axis=1)].reshape(len(not_done), 1))
        reward_est = self.model.predict(s)
        reward_est[np.arange(self.batch_size).reshape(self.batch_size, 1), a] = r
        self.model.fit(s, reward_est, verbose=0)

    def target_update(self):
        self.target_model.set_weights(self.model.get_weights())

# Training
random.seed(1234)
learning = [0.001]
decay = [0.995]
gamma = [0.990]

for l in learning:
    for d in decay:
        for g in gamma:
            print("Learning Rate: {}, Gamma: {}, Decay: {}".format(l, g, d))
            epsilon = 1
            env = gym.make("LunarLander-v2")
            env.reset()
            env.render()
            agent = DQN(env=env, learning=l, gamma=g, decay = d)
            total = deque(maxlen=200)
            epsilon_min = 0.01
            trials = 1000
            episodes = 2000
            avg_rewards_ = []

            for episode in range(episodes):
                s = env.reset()
                s = np.reshape(s, [1, 8])
                rewards = 0
                for trial in range(1000):
                    if np.random.random() <= epsilon:
                        a = np.random.randint(0, 4)
                    else:
                        a = np.argmax(agent.model.predict(s))
                    new_s, r, done, _ = env.step(a)
                    new_s = new_s.reshape(1, 8)
                    rewards += r
                    agent.remember(s, a, r, new_s, done)
                    if (len(agent.memory) > 64) & (trial % 2 == 0):
                        agent.replay()
                    s = new_s
                    if done:
                        agent.target_update()
                        break
                total.append(rewards)
                avg_rewards = np.average(total)
                avg_rewards_.append(avg_rewards)
                if epsilon > epsilon_min:
                    epsilon *= d
                print("Episode: {}/{}, Average Reward Score: {}".format(episode, episodes, avg_rewards))
                if episode % 100 == 0:
                    print("Average Reward Score for episodes {}: {}".format(episode, avg_rewards))
                if avg_rewards >= 200.0:
                    # model_json = agent.target_model.to_json()
                    # with open("trained_model.json", "w") as json_file:
                    #     json_file.write(model_json)
                    # agent.target_model.save_weights("model_weights.h5")
                    break
                # df_total = pd.DataFrame(list(collections.deque(total)))
                # df_total.to_csv("l_{}_g_{}_d_{}_total_v3.csv".format(l, g, d))
                # df_avg_rewards_ = pd.DataFrame(avg_rewards_)
                # df_avg_rewards_.to_csv("l_{}_g_{}_d_{}_v3.csv".format(l, g, d))

# Getting trained model
json_file = open("trained_model.json", "r")
trained_model = json_file.read()
json_file.close()
trained_model = model_from_json(trained_model)
trained_model.load_weights("model_weights.h5")
env = gym.make("LunarLander-v2")
agent = DQN(env=env, learning=0.001, gamma=0.990, decay = 0.995)
episodes = 100
epsilon = 1
min_epsilon = 0.01
total = deque(maxlen=100)
avg_rewards_ = []
random.seed(1234)
for episode in range(episodes):
    s = env.reset()
    s = np.reshape(s, [1, 8])
    rewards = 0
    for trial in range(1000):
        env.render()
        a = np.argmax(trained_model.predict(s))
        new_s, r, done, _ = env.step(a)
        new_s = np.reshape(new_s, [1, 8])
        rewards += r
        s = new_s
        if done:
            break
    total.append(rewards)
    avg_rewards = np.average(total)
    avg_rewards_.append(avg_rewards)
    df_total_trained = pd.DataFrame(list(collections.deque(total)))
    df_total_trained.to_csv("Trained Model Total Rewards.csv")
    df_avg_rewards_trained_ = pd.DataFrame(avg_rewards_)
    df_avg_rewards_trained_.to_csv("Trained Model Average Rewards.csv")
    plt.plot(range(len(total)), total)
    print("Episode: {}/{}, Average Reward Score: {}".format(episode, episodes, np.average(total)))

