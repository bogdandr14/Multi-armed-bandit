
import numpy as np
from agent_interface import AgentInterface
np.random.seed(0)

"""
The incremental update rule TI for each (action a, reward r):
TI(a) <- TI(a) + beta * (r(a) - rm)
rm <- (1- alpha) * rm + alpha * r(a)
where:
n = number of times arm "a" was used
TI(a) = preference for arm "a"
r(a) = reward of sampling action bandit "a"
rm = reward mean for all rewards till present
"""


# Implementation for the multi-armed bandit that uses the reinforcement comparison
class ReinforcementAgent(AgentInterface):

    def __init__(self, numberOfArms: np.int, alpha: np.float, beta: np.float):
        self.numberOfArms = numberOfArms
        self.alpha = alpha  # Alpha is a learning rate of the agent
        self.beta = beta  # Beta is a learning rate of the agent
        self.reset()

    def update(self, nrArm: np.int, reward: np.int):
        # Update TI action-value given (action, reward)
        self.TI[nrArm] += self.beta * (reward - self.rewardMean)
        self.rewardMean = (1 - self.alpha) * \
            self.rewardMean + self.alpha * reward

    def reset(self):
        self.rewardMean = 0.0
        # Set up bandit arms with fixed preferences for the arms TI(a)
        self.TI = np.zeros(self.numberOfArms, dtype=np.float)
        # Probability of selecting a specific arm
        self.P = np.zeros(self.numberOfArms, dtype=np.float)

    def getArm(self):
        # Compute probability for each arm
        exp = np.power(np.e, self.TI)
        self.P = exp / exp.sum()
        
        # Randomnly select a value in interval [0.1)
        randArm = np.random.random()
        pUpperBound = 0.0
        for armIndex, pVal in enumerate(self.P):
            pUpperBound += pVal
            # determine if the randomly-selected value is in the current arm interval
            if (randArm < pUpperBound):
                return armIndex
