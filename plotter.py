import numpy as np
import matplotlib.pyplot as plt
import os
from agent_type import AgentType
from history import History


# Plats the graphic images for the results
class Plotter:

    def __init__(self, saveFig: np.bool, nrExperiments: np.int, probabilities):
        self.saveFig = saveFig
        self.totalExperiments = nrExperiments
        self.probabilities = probabilities
        self.output_dir = os.path.join(os.getcwd(), "output")

    def plotRewards(self, history: History, agentType: AgentType):
        R_avg = history.rewardHistorySum / np.float(self.totalExperiments)
        plt.plot(R_avg, ".")
        plt.xlabel("Step")
        plt.ylabel("Average Reward")
        plt.grid()
        ax = plt.gca()
        plt.xlim([1, history.steps])
        if self.saveFig:
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
            plt.savefig(os.path.join(self.output_dir, "{}_rewards.png".format(agentType.value)),
                        bbox_inches="tight")
        else:
            plt.show()
        plt.close()
        print("Reward average for {} agent = {}".format(agentType.name, R_avg[len(R_avg)-1]))

    def plotActions(self, history: History, agentType: AgentType):
        for i in range(len(self.probabilities)):
            A_pct = 100 * history.actionHistorySum[:, i] / self.totalExperiments
            steps = list(np.array(range(len(A_pct)))+1)
            plt.plot(steps, A_pct, "-",
                     linewidth=1,
                     label="Arm {} ({:.0f}%)".format(i+1, 100*self.probabilities[i]))
        plt.xlabel("Step")
        plt.ylabel("Count Percentage (%)")
        leg = plt.legend(loc='upper left', shadow=True)
        plt.xlim([1, history.steps])
        plt.ylim([0, 100])
        for legobj in leg.legendHandles:
            legobj.set_linewidth(1.0)
        if self.saveFig:
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
            plt.savefig(os.path.join(self.output_dir, "{}_actions.png".format(agentType.value)),
                        bbox_inches="tight")
        else:
            plt.show()
        plt.close()
