"""
 Runnable class for solving the multi-armed bandit problem using different algorithms.
 To configure which algorithms to run, modify 'experiments_configuration' file.
"""

from experiments_configuration import *
from agent_experiment_bundle import AgentExperimentsBundle
from agent_experiment import getAgentExperiment
from plotter import Plotter

# Plotter for output images of the results
plotter = Plotter(SAVE_FIG, NO_EXPERIMENTS, ARMS_PROBABILITIES)

# experiments bundle used for executing multiple experiments with a specific type of agent
experimentsBundle = AgentExperimentsBundle(
    plotter, NO_EXPERIMENTS, NO_STEPS, len(ARMS_PROBABILITIES))

# Run greedy multi-armed bandit experiments
for agentType in AGENTS_TO_EXPERIMENT:
    experimentsBundle.run(getAgentExperiment(agentType), agentType)
