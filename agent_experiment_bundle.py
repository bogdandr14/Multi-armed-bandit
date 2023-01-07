import numpy as np
from agent_type import AgentType
from experiment import Experiment
from plotter import Plotter
from history import History


# Used for executing a given number of experiments with the same agent
class AgentExperimentsBundle:
    def __init__(self, plotter: Plotter, nrExperiments: np.int, nrSteps: np.int, nrArms: np.int):
        self.plotter = plotter  # set plotter
        self.nrExperiments = nrExperiments  # set number of experiments to run
        self.nrSteps = nrSteps  # set the number of steps for the experiments
        self.nrArms = nrArms  # set the number of arms

    # Run the given experiment for the specified number of times, then create plots with the results
    def run(self, experiment: Experiment,  agentType: AgentType):
        history = History(self.nrSteps, self.nrArms)

        print("Running {} multi-armed bandits with {} arms".format(agentType.name, self.nrArms))

        for i in range(self.nrExperiments):
            # Show console log for 10 experiments
            showLog = (i + 1) % (self.nrExperiments / 10) == 0
            actions, rewards = experiment.run()  # Perform experiment
            history.addExperimentResult(
                actions, rewards, showLog)  # Add result to history

        # plot the results
        self.plotter.plotRewards(history, agentType)
        self.plotter.plotActions(history, agentType)
