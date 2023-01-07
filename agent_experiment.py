
from agent_type import AgentType
from environment import Environment
from experiment import Experiment
from greedy_agent import GreedyAgent
from pursuit_agent import PursuitAgent
from reinforcement_agent import ReinforcementAgent
from experiments_configuration import *


# Used for retrieving an experiment that uses a specific agent implementation
def getAgentExperiment(agentType: AgentType):
    if (agentType.value == AgentType.Greedy.value):
        agent = GreedyAgent(len(ARMS_PROBABILITIES), eps, deg)
    else:
        if (agentType.value == AgentType.Pursuit.value):
            agent = PursuitAgent(len(ARMS_PROBABILITIES), beta_p)
        else:
            if (agentType.value == AgentType.Reinforced.value):
                agent = ReinforcementAgent(
                    len(ARMS_PROBABILITIES), alpha, beta_r)

    return Experiment(Environment(ARMS_PROBABILITIES), agent, NO_STEPS)
