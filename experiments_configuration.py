from agent_type import AgentType
# Settings for the bandits

ARMS_PROBABILITIES = [0.25, 0.60, 0.80, 0.45,
                      0.50, 0.35, 0.10, 0.75]  # bandit arm probabilities of success
NO_EXPERIMENTS = 5000  # number of experiments to perform per agent type
NO_STEPS = 1000  # number of steps per experiment
eps = 0.32  # probability of random exploration for epsilon-greedy
deg = 0.999 # degrade rate
beta_p = 0.01  # learning rate for pursuit algorithm 
beta_r = 0.3  # learning rate for pursuit algorithm 
alpha = 0.35  # learning rate for the reinforcement comparison
SAVE_FIG = True  # save file in same directory


AGENTS_TO_EXPERIMENT = [
   AgentType.Greedy,
    # AgentType.Pursuit,
    # AgentType.Reinforced
]
