
import numpy as np
from agent_interface import AgentInterface
np.random.seed(0)


# Implementation for the multi-armed bandit that uses the pursuit algorithms
class PursuitAgent(AgentInterface):

    def __init__(self, numberOfArms: np.int, beta: np.float):
        self.numberOfArms = numberOfArms
        self.beta = beta  # The learning rate of the agent
        self.reset()

    def update(self, nrArm: np.int, reward: np.int):
        self.armUsage[nrArm] += 1
        self.U[nrArm] += (1.0 / self.armUsage[nrArm]) * \
            (reward - self.U[nrArm])

    def reset(self):
        self.armUsage = np.zeros(
            self.numberOfArms, dtype=np.int)  # action counts n(a)
        # Set up bandit arms with fixed probability distribution of success U(a)
        self.U = np.zeros(self.numberOfArms, dtype=np.float)
        # Probability of selecting a specific arm
        self.P = np.zeros(self.numberOfArms, dtype=np.float)
        # Initial probability for selecting any arm is 1 / k (number of arms)
        self.P += 1.0 / self.numberOfArms

    def getArm(self):
        # Check that at least one arm was used
        if (self.U.max() != 0.0):
            # Compute probability for each arm
            for armIndex, uVal in enumerate(self.U):
                self.P[armIndex] += self.beta * \
                    ((1 if uVal == self.U.max() else 0) - self.P[armIndex])

        # Randomnly select a value in interval [0,1) and multiply by sum of probabilities
        randArm = np.random.random() * self.P.sum()
        pUpperBound = 0.0
        for armIndex, pVal in enumerate(self.P):
            pUpperBound += pVal
            # determine if the randomly-selected value is in the current arm interval
            if (randArm < pUpperBound):
                return armIndex
