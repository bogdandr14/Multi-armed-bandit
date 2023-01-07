import numpy as np


# Keeps the history of the actions and the rewards that were produced at each step for all experiments of an agent
class History:

    def __init__(self, steps: np.int, nrArms: np.int):
        self.rewardHistorySum = np.zeros((steps,))  # reward history sum
        self.actionHistorySum = np.zeros((steps, nrArms))  # action history sum
        self.currentExperimentNr = 0
        self.steps = steps

    # Add the results of an experiment to the history
    def addExperimentResult(self, actions, rewards, showLog):
        self.currentExperimentNr += 1
        self.rewardHistorySum += rewards

        for j, a in enumerate(actions):
            self.actionHistorySum[j][a] += 1

        # Show the console log for the experiment
        if (showLog):
            print("[Experiment {}] ".format(self.currentExperimentNr) +
                  "reward_avg = {}".format(np.sum(rewards) / len(rewards)))

