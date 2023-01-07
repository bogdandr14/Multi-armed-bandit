import numpy as np
from agent_interface import AgentInterface
np.random.seed(0)

"""
The incremental update rule action-value U for each (action a, reward r):
U(a) <- U(a) + 1/n * (r - U(a))
n += 1
where:
n = number of times arm "a" was used
U(a) = value mean of arm "a"
r(a) = reward of sampling action bandit "a"
"""


# Implementation for the multi-armed bandit that uses the epsilon-greedy algorithm
class GreedyAgent(AgentInterface):

    def __init__(self, numberOfArms: np.int, eps: np.float, deg: np.float):
        self.numberOfArms = numberOfArms
        self.eps = eps  # epsilon value used to determine if we explore or exploit
        self.deg = deg  # value for degradation of epsilon with time
        self.reset()

    def update(self, nrArm: np.int, reward: np.int):
        self.armUsage[nrArm] += 1
        self.U[nrArm] += (1.0 / self.armUsage[nrArm]) * \
            (reward - self.U[nrArm])
        self.epsDeg *= self.deg  # degrade epsilon

    def reset(self):
        self.armUsage = np.zeros(
            self.numberOfArms, dtype=np.int)  # action counts n(a)
        # Set up bandit arms with fixed probability distribution of success U(a)
        self.U = np.zeros(self.numberOfArms, dtype=np.float)
        self.epsDeg = self.eps  # reset degradable epsilon value

    def getArm(self):

        if np.random.random() < self.epsDeg / self.numberOfArms:  # explore
            if (self.U.max() == 0.0):  # All arms have the same reward mean
                return np.random.randint(self.numberOfArms)
            else:  # Choose an arm which does not have the highest reward mean
                return np.random.choice(np.where(self.U < self.U.max())[0])
        else:  # exploit
            # Choose from the arms with the highest reward mean
            return np.random.choice(np.flatnonzero(self.U == self.U.max()))
