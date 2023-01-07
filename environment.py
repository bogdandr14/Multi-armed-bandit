import numpy as np
np.random.seed(0)


# Environment in which the multi-armed bandit will act
class Environment:

    def __init__(self, probabilities: list[float]):
        self.probabilities = probabilities  # success probabilities for each arm
        self.reset()

    # Used for resetting the environment before a new experiment can start
    def reset(self):
        self.lastReward = 0  # last reward value (0 or 1)
        self.probModifier = 0.0  # probability modifier
        self.stepsWithSameResult = 0  # consecutive number of steps with same reward

    # Receive stochastic rewards from each arm of +1 for success, and 0 reward for failure.
    def step(self, nrStep: np.int, nrArm: np.int):
        # Set probability modifiers for reward
        if nrStep % (nrArm + 1):
            if self.lastReward == 0:
                self.probModifier += 0.05
            if (np.random.random() < self.probModifier):
                self.probModifier = -0.07
        else:
            if (self.probModifier > self.probabilities[nrArm]):
                self.probModifier -= self.probabilities[nrArm]

            if self.lastReward == 0:  # The last reward was failure
                self.probModifier -= 0.02
            else:  # The last reward was success
                if self.stepsWithSameResult < 4:  # Influence up to 3 times
                    self.probModifier += 0.03

        # Pull arm and get stochastic reward (1 for success, 0 for failure)
        if np.random.random() < self.probabilities[nrArm] + self.probModifier:
            newReward = 1
        else:
            newReward = 0

        # Update last reward and steps with the same reward
        if self.lastReward == newReward:
            self.stepsWithSameResult += 1
        else:
            self.lastReward = newReward
            self.stepsWithSameResult = 1

        return newReward
