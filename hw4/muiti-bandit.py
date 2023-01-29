import numpy as np
import random
import scipy.stats as stats
import matplotlib.pyplot as plt


class SlotMachines:

    def __init__(self, n_machines=3, prop_reword=[], cash=10000):
        self.n_machines = n_machines
        # If proba
        if prop_reword:
            self.prop_reword = prop_reword
        else:
            self.prop_reword = np.random.normal(0, 1, n_machines)
        # Q for each arm
        self.Q = np.zeros(n_machines)
        # Pull times for each arm
        self.N = np.zeros(n_machines)
        # Number of wins for each arm
        self.W = np.zeros(n_machines)
        # Total reward for each arm
        # self.total_R = np.zeros(n_machines)
        # Adjust reward here
        self.reward = 16
        # All budget
        self.cash = cash
        self.total_reward = 0

    # Pull the arm by number
    def pullArm(self, arm):
        self.cash -= 10
        if random.random() >= (1 - self.prop_reword[arm]):
            return self.reward
        return 0

    # Print pull times for each arms
    def print_pull_times(self):
        for i, times in enumerate(self.N):
            print(f"arm {i} pulls {times} times")

# Epsilon greedy


def epsilon_greedy(epsilon):
    # Create slot machines
    slot_machines = SlotMachines(3, [0.3, 0.7, 0.8])
    # If have budget
    while slot_machines.cash > 0:
        # Random choose a arm to pull
        if random.random() < epsilon:
            arm = np.random.randint(0,  slot_machines.n_machines)
        # Pull the best arm
        else:
            arm = np.argmax(slot_machines.Q)
        # Get the reward by pull the arm
        reward = slot_machines.pullArm(arm)
        if reward:
            # Add reward to total reward
            slot_machines.total_reward += reward
            # Add reward to that arm
            slot_machines.W[arm] += 1
        slot_machines.N[arm] += 1
        slot_machines.Q[arm] = slot_machines.W[arm] / slot_machines.N[arm]

    print(slot_machines.total_reward)
    slot_machines.print_pull_times()

    arms = [
        stats.beta(a=1+w, b=1+t-w) for t, w in zip(slot_machines.N, slot_machines.W)]
    plot(arms, "Greedy")


def UCB(c=0.5):
    np.seterr(all="ignore")
    slot_machines = SlotMachines(3, [0.3, 0.7, 0.8])
    step = 0
    while slot_machines.cash > 0:
        step += 1
        arm = np.argmax(slot_machines.Q + c *
                        np.sqrt(np.log(step) / slot_machines.N))

        reward = slot_machines.pullArm(arm)
        if reward:
            slot_machines.total_reward += reward
            slot_machines.W[arm] += 1
        slot_machines.N[arm] += 1
        slot_machines.Q[arm] = slot_machines.W[arm] / slot_machines.N[arm]

    print(slot_machines.total_reward)
    slot_machines.print_pull_times()

    arms = [
        stats.beta(a=1+w, b=1+t-w) for t, w in zip(slot_machines.N, slot_machines.W)]
    plot(arms, "UCB")


def thompson_sampling():
    slot_machines = SlotMachines(3, [0.3, 0.7, 0.8])
    while slot_machines.cash > 0:
        beta_dis = [
            random.betavariate(1+w, 1+t-w) for t, w in zip(slot_machines.N, slot_machines.W)]
        arm = np.argmax(beta_dis)

        reward = slot_machines.pullArm(arm)
        if reward:
            slot_machines.total_reward += reward
            slot_machines.W[arm] += 1
        slot_machines.N[arm] += 1
        slot_machines.Q[arm] = slot_machines.W[arm] / slot_machines.N[arm]

    print(slot_machines.total_reward)
    slot_machines.print_pull_times()

    arms = [
        stats.beta(a=1+w, b=1+t-w) for t, w in zip(slot_machines.N, slot_machines.W)]
    plot(arms, "TS")


def plot(arms, type):
    plot_x = np.linspace(0.001, .999, 100)
    for arm in arms:
        y = arm.pdf(plot_x)
        plt.plot(plot_x, y)
    plt.xlim([0, 1])
    plt.ylim(bottom=0)
    plt.title(f"Probability Density Function ({type})")
    plt.show()


epsilon_greedy(0.1)
UCB()
thompson_sampling()
