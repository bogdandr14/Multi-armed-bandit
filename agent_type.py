import enum


# Contains all algorithm implementations for the multi-armed bandit agent
class AgentType(enum.Enum):
    Greedy = "epsilon_greedy"
    Pursuit = "pursuit_algorithm"
    Reinforced = "reinforcement_comparison"
