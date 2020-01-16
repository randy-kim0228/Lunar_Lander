# import gym
# import numpy as np
# import random
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import Adam
#
# from collections import deque
#
# import matplotlib.pyplot as plt
#
# class DQN:
#     def __init__(self, env):
#         self.env = env
#         self.memory = deque(maxlen=1000)
#         self.gamma = 0.99
#         self.learning_rate = 0.0001
#         self.tau = .01
#         self.model = self.create_model()
#         self.target_model = self.create_model()
#
#     def create_model(self):
#         model = Sequential()
#         model.add(Dense(128, input_dim=self.env.observation_space.shape[0], activation="relu"))
#         model.add(Dense(64, activation="relu"))
#         model.add(Dense(self.env.action_space.n, activation='linear'))
#         model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
#         return model
#
#     def remember(self, state, action, reward, new_state, done):
#         self.memory.append([state, action, reward, new_state, done])
#
#     def replay(self):
#         batch_size = 64
#         samples = random.sample(self.memory, batch_size)
#         s = np.vstack([e[0] for e in samples])
#         a = np.vstack([e[1] for e in samples])
#         r = np.vstack([e[2] for e in samples])
#         n = np.vstack([e[3] for e in samples])
#         status = np.vstack([e[4] for e in samples]).astype(np.uint8)
#         statusn = np.where(status == False)[0]
#         # s = np.vstack(samples[:, 0])
#         # a = np.array(samples[:, 1], dtype=int)
#         # r = np.copy(samples[:, 2])
#         # s_p = np.vstack(samples[:, 3])
#         # status = np.where(samples[:, 4] == False)
#
#         estimate = self.model.predict(n)
#         target = self.target_model.predict(n)
#
#         if len(status[0]) > 0:
#             # c = estimate[statusn]
#             best_next_action = np.argmax(estimate[statusn], axis=1)
#             d = target[statusn, best_next_action].reshape(len(statusn), 1)
#             r[statusn] += np.multiply(self.gamma, target[statusn, best_next_action].reshape(len(statusn), 1))
#
#         expected_reward = self.model.predict(s)
#         expected_reward[np.arange(batch_size).reshape(batch_size, 1), a] = r
#
#         self.model.fit(s, expected_reward, verbose=0)
#
#         # for sample in samples:
#         #     state, action, reward, new_state, done = sample
#         #     target = self.target_model.predict(state)
#         #     if done:
#         #         target[0][action] = reward
#         #     else:
#         #         Q_future = max(self.target_model.predict(new_state)[0])
#         #         target[0][action] = reward + Q_future * self.gamma
#         #     self.model.fit(state, target, verbose=0)
#
#     def target_train(self):
#         weights = self.model.get_weights()
#         target_weights = self.target_model.get_weights()
#
#         # target_weights = weights*self.tau + target_weights * (1 - self.tau)
#
#         for i in range(len(target_weights)):
#             target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
#         self.target_model.set_weights(target_weights)
#
#     def save_model(self, fn):
#         self.model.save(fn)
#
#
# # def main():
# env = gym.make("LunarLander-v2")
# trials = 2000
# EPISODE = 500
# epsilon = 1
# dqn_agent = DQN(env=env)
# total = deque(maxlen=100)
# rolling_total = []
# min_epsilon = 0.01
# decayrate = 0.999
#
# for trial in range(trials):
#     s = env.reset()
#     s = np.reshape(s, [1, 8])
#     total_reward = 0
#
#     for step in range(EPISODE):
#         if np.random.random() > epsilon:
#             a = np.argmax(dqn_agent.model.predict(s))
#         else:
#             a = np.random.randint(0, 4)
#
#         new_s, r, done, _ = env.step(a)
#         total_reward += r
#         new_s = new_s.reshape(1, 8)
#
#         dqn_agent.remember(s, a, r, new_s, done)
#
#         if (len(dqn_agent.memory) > 64) & (step % 4 == 0):
#             dqn_agent.replay()
#             dqn_agent.target_train()
#
#         s = new_s
#         if done:
#             break
#
#     total.append(total_reward)
#     ave = np.average(total)
#     rolling_total.append(ave)
#     epsilon = max(min_epsilon, epsilon * decayrate)
#     print('\rEpisode {}\tAverage Score: {:.2f}'.format(trial, ave), end="")
#     if trial % 100 == 0:
#         print('\rEpisode {}\tAverage Score: {:.2f}'.format(trial, ave))
#     if ave >= 220.0:
#         print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(trial, ave))
#         model_json = dqn_agent.target_model.to_json()
#         with open("model.json", "w") as json_file:
#             json_file.write(model_json)
#         dqn_agent.target_model.save_weights("model.h5")
#
#         break
#
#
# #         print('episode %i, episode reward %i, rolling reward %i, epsilon %.2f' % (trial, total_reward, ave, epsilon))
#
# if __name__ == "__main__":
#     main()
#
# from keras.models import model_from_json
#
# # serialize model to JSON
# # model_json = model.to_json()
# # with open("model.json", "w") as json_file:
# #     json_file.write(model_json)
# # # serialize weights to HDF5
# # model.save_weights("model.h5")
# # print("Saved model to disk")
#
# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
#
# # # evaluate loaded model on test data
# # loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# # score = loaded_model.evaluate(X, Y, verbose=0)
# # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
#
#
#
# env = gym.make("LunarLander-v2")
# trials = 2000
# EPISODE = 500
# epsilon = 1
# dqn_agent = DQN(env=env)
# total = deque(maxlen=100)
# rolling_total = []
# min_epsilon = 0.01
# decayrate = 0.999
#
# for i in range(100):
#     # env.seed(None)
#     # print(model.get_weights()[0])
#     # prev = model.get_weights()
#
#     total_reward = 0
# #     steps = 0
#     s = env.reset()
#     s = np.reshape(s, [1, 8])
#
#     for step in range(EPISODE):
#         a = np.argmax(loaded_model.predict(s))
#         new_s, r, done, info = env.step(a)
#         new_s = np.reshape(new_s, [1, 8])
# #         env.render()
#         total_reward += r
#         s = new_s
#         if done: break
#
# #     env.close()
#     total.append(total_reward)
#     plt.plot(range(len(total)), total)
#     print('\rEpisode {}\tAverage Score: {:.2f}'.format(step, np.average(total)), end="")

import matplotlib.pyplot as plt
def exp1(total, learning, gamma, decay):
    plt.plot(range(len(total)), total)
#     plt.plot(range(len(total)), np.repeat(np.average(total), len(total)))
    plt.title("DQN learning %.3f, gamma %.3f, decay %.3f" %(learning, gamma, decay))
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig("DQN L %.3f G %.3f D %.3f.png" %(learning, gamma, decay))
    plt.show()

import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque


class DQN:
    def __init__(self, env, learning, gamma):
        self.env = env
        self.memory = deque(maxlen=1000)
        self.gamma = gamma
        self.learning_rate = learning
        self.tau = .01
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.env.observation_space.shape[0], activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 64
        samples = random.sample(self.memory, batch_size)
        s = np.vstack([e[0] for e in samples])
        a = np.vstack([e[1] for e in samples])
        r = np.vstack([e[2] for e in samples])
        n = np.vstack([e[3] for e in samples])
        status = np.vstack([e[4] for e in samples]).astype(np.uint8)
        statusn = np.where(status == False)[0]

        estimate = self.model.predict(n)
        target = self.target_model.predict(n)

        if len(status[0]) > 0:
            best_next_action = np.argmax(estimate[statusn], axis=1)
            r[statusn] += np.multiply(self.gamma, target[statusn, best_next_action].reshape(len(statusn), 1))

        expected_reward = self.model.predict(s)
        expected_reward[np.arange(batch_size).reshape(batch_size, 1), a] = r

        self.model.fit(s, expected_reward, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


# def main():
env = gym.make("LunarLander-v2")

# epsilon = 1
# dqn_agent = DQN(env=env)
# total = deque(maxlen=100)
# rolling_total = []
# min_epsilon = 0.01
decayrate = [0.999, 0.99]
learningrate = [0.0001, 0.001]
gammarate = [0.999, 0.99]

for learning in learningrate:
    for gamma in gammarate:
        for decay in decayrate:
            env = gym.make("LunarLander-v2")
            dqn_agent = DQN(env=env, learning=learning, gamma=gamma)
            epsilon = 1
            total = deque(maxlen=100)
            min_epsilon = 0.01
            trials = 1000
            EPISODE = 1000
            rolling_total = []

            for trial in range(trials):
                s = env.reset()
                s = np.reshape(s, [1, 8])
                total_reward = 0

                for step in range(EPISODE):
                    if np.random.random() > epsilon:
                        a = np.argmax(dqn_agent.model.predict(s))
                    else:
                        a = np.random.randint(0, 4)

                    new_s, r, done, _ = env.step(a)
                    total_reward += r
                    new_s = new_s.reshape(1, 8)

                    dqn_agent.remember(s, a, r, new_s, done)

                    if (len(dqn_agent.memory) > 64) & (step % 2 == 0):
                        dqn_agent.replay()
                        dqn_agent.target_train()

                    s = new_s
                    if done:
                        break

                total.append(total_reward)
                ave = np.average(total)
                rolling_total.append(ave)
                epsilon = max(min_epsilon, epsilon * decay)
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(trial, ave), end="")
                if trial % 100 == 0:
                    print('\rEpisode {}\tAverage Score: {:.2f}'.format(trial, ave))
                if ave >= 220.0:
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(trial, ave))
                    target_model_json = dqn_agent.target_model.to_json()
                    model_json = dqn_agent.model.to_json()
                    with open("target_model.json", "w") as json_file:
                        json_file.write(target_model_json)
                    with open("model.json", "w") as json_file:
                        json_file.write(model_json)

                    dqn_agent.model.save_weights("model.h5")
                    dqn_agent.target_model.save_weights("target_model.h5")
                    #                     exp1(rolling_total)

                    #                     exp1(rolling_total, learning, gamma, decay)
                    break
            print('\nlearning %.3f, gamma %.3f, decay %.3f' % (learning, gamma, decay))
            exp1(rolling_total, learning, gamma, decay)




from keras.models import model_from_json

# serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
load_model = json_file.read()
json_file.close()
model = model_from_json(load_model)
model.load_weights("model.h5")

json_file = open('target_model.json','r')
load_target = json_file.read()
json_file.close()
target_model = model_from_json(load_target)
target_model.load_weights("target_model.h5")


env = gym.make("LunarLander-v2")
trials = 2000
EPISODE = 1000
total = []
rolling_total = []
min_epsilon = 0.01
decayrate = 0.999

for i in range(100):
    total_reward = 0
    s = env.reset()
    s = np.reshape(s, [1, 8])

    for step in range(EPISODE):
        a = np.argmax(model.predict(s))
        new_s, r, done, info = env.step(a)
        new_s = np.reshape(new_s, [1, 8])
        total_reward += r
        s = new_s
        if done:
            break
    total.append(total_reward)
    print('Episode {}\tAverage Score: {:.2f}'.format(step, np.average(total)))
    rolling_total.append(np.average(total))
plt.plot(range(len(rolling_total)), rolling_total)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.savefig("DQN_Test")
plt.show()


env = gym.make("LunarLander-v2")
trials = 2000
EPISODE = 1000
epsilon = 1
total = []
rolling_total = []
min_epsilon = 0.01
decayrate = 0.999

for i in range(100):
    total_reward = 0
    s = env.reset()
    s = np.reshape(s, [1, 8])

    for step in range(EPISODE):
        a = np.argmax(target_model.predict(s))
        new_s, r, done, info = env.step(a)
        new_s = np.reshape(new_s, [1, 8])
        total_reward += r
        s = new_s
        if done:
            break
    total.append(total_reward)
    print('Episode {}\tAverage Score: {:.2f}'.format(step, np.average(total)))
    rolling_total.append(np.average(total))
plt.plot(range(len(rolling_total)), rolling_total)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.savefig("DQN_Test")
plt.show()





