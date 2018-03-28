# Cody Herberholz
# CS445
# HW6

import numpy as np
import random
import matplotlib.pyplot as plt

N = 5000
M = 300
INTERVALS = int(N / 100)

class Robby(object):
    def __init__(self):
        self.reward = 0
        self.epsilon = 1.0
        self.discount_factor = 0.9
        self.learning_rate = 0.2
        self.board = np.zeros((10, 10))
        self.q_matrix = np.zeros((100, 5))
        self.state = np.zeros(2)

    # the ones signify cans and zeros are not cans
    def initialize_grid(self):
        for i in range(10):
            for j in range(10):
                self.board[i][j] = random.randint(0, 1)

        # randomly place robby on grid
        self.state[0] = random.randint(0, 9)
        self.state[1] = random.randint(0, 9)

        # convert array types to int
        self.board = self.board.astype(np.int64)
        self.state = self.state.astype(np.int64)

    def update_qmatrix(self, old_qstate, new_qstate, move, reward):
        max_q = 0.0  # holds largest new state q value
        reward = float(reward)

        for i in range(5):
            q = self.q_matrix[new_qstate][i]
            if q > max_q:
                max_q = q
        value = reward + (self.discount_factor * max_q) - self.q_matrix[old_qstate][move]
        self.q_matrix[old_qstate][move] += self.learning_rate * value
        self.q_matrix[old_qstate][move] = round(self.q_matrix[old_qstate][move], 2)

    def choose_action(self):
        action = 0
        max_value = -8000
        # probability that ill choose a random action is epsilon
        rando = np.random.rand()
        if rando < self.epsilon:
            action = random.randint(0, 4)
        else:
            location = self.state[0] * 10 + self.state[1]
            for j in range(5):
                temp = self.q_matrix[location][j]
                if temp > max_value:
                    max_value = temp
                    action = j
            if max_value == 0:
                action = random.randint(0, 4)

        return action

    # if robby tries to move off grid then special case is caught and movement is prohibited
    def perform_action(self, move, train):
        temp_reward = 0
        # self.reward -= 0.5  # action tax

        # Move North
        if move == 0:
            old_qstate = self.state[0] * 10 + self.state[1]  # creates value that translate to q_matrix index

            if self.state[0] == 0:
                temp_reward = -5
                self.reward -= 5
            else:
                self.state[0] -= 1

            new_qstate = self.state[0] * 10 + self.state[1]

        # Move South
        elif move == 1:
            old_qstate = self.state[0] * 10 + self.state[1]

            if self.state[0] == 9:
                temp_reward = -5
                self.reward -= 5
            else:
                self.state[0] += 1

            new_qstate = self.state[0] * 10 + self.state[1]

        # Move East
        elif move == 2:
            old_qstate = self.state[0] * 10 + self.state[1]

            if self.state[1] == 9:
                temp_reward = -5
                self.reward -= 5
            else:
                self.state[1] += 1

            new_qstate = self.state[0] * 10 + self.state[1]

        # Move West
        elif move == 3:
            old_qstate = self.state[0] * 10 + self.state[1]

            if self.state[1] == 0:
                temp_reward = -5
                self.reward -= 5
            else:
                self.state[1] -= 1

            new_qstate = self.state[0] * 10 + self.state[1]

        # Pick Up Can
        elif move == 4:
            row = self.state[0]
            col = self.state[1]
            if self.board[row][col] == 1:
                temp_reward = 10
                self.reward += 10
                self.board[row][col] = 0
            else:
                temp_reward = -1
                self.reward -= 1
            old_qstate = self.state[0] * 10 + self.state[1]
            new_qstate = old_qstate

        if train == 1:
            self.update_qmatrix(old_qstate, new_qstate, move, temp_reward)

    def display(self):
        print(self.board)

    def plot(self, plot_rewards):
        plt.plot(plot_rewards)
        plt.xlim(0, INTERVALS)
        plt.xlabel("Episode Intervals")
        plt.ylabel("Episode Reward")
        plt.title("Training Reward Plot")
        plt.show()

    def test(self):
        reward_sum = np.zeros(N)

        # after training run 5000 test runs
        for i in range(N):
            self.initialize_grid()
            for j in range(M):
                choice = self.choose_action()
                self.perform_action(choice, 0)  # zero does not allow qmatrix to update
            reward_sum[i] = self.reward
            self.reward = 0

        test_average = np.mean(reward_sum)
        test_std = np.std(reward_sum)
        print(test_average)
        print(test_std)

    def run(self):
        plot_rewards = np.zeros(INTERVALS)
        k = 0

        # run 5000 training runs
        for i in range(N):
            self.initialize_grid()
            if i != 0 and self.epsilon > 0.1:
                if i % 50 == 0:
                    self.epsilon -= 0.01
            for j in range(M):
                choice = self.choose_action()
                self.perform_action(choice, 1)  # the one allows updating of qmatrix
            if i % 100 == 0:
                plot_rewards[k] = self.reward
                k += 1
            self.reward = 0

        print(self.q_matrix)
        self.plot(plot_rewards)
        self.test()

def main():
    rob = Robby()
    # each new run requires a new q matrix
    rob.run()

main()
