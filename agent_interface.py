import numpy as np

# Interface for the agent implementation


class AgentInterface:

    # Used for updating the agent information after receiving the rewards
    def update(self, nrArm: np.int, reward: np.int):
        pass

    # Used for resetting the agent before a new experiment can start
    def reset(self):
        pass

    # Used for determining which arm the bandit will act with
    def getArm(self) -> int:
        pass
