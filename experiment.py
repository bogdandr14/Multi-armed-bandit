import numpy as np
from environment import Environment
from agent_interface import AgentInterface


# Used for running a specific agent implementation
class Experiment:

    def __init__(self, environment: Environment, agent: AgentInterface, steps):
        self.agent = agent  # initialize agent
        self.env = environment  # initialize environment
        # number of actions (arms choosed) for the experiment
        self.steps = steps

    # Start multi-armed bandit simulation
    def run(self):
        # Setup initial values
        actions, rewards = [], []
        self.agent.reset()
        self.env.reset()

        # Run the multi-armed for the specified number of steps
        for step in range(self.steps):
            nrArm = self.agent.getArm()  # sample policy
            reward = self.env.step(step, nrArm)  # Take step + Get reward
            self.agent.update(nrArm, reward)  # update agent
            actions.append(nrArm)  # append the selected arm
            rewards.append(reward)  # append the reward

        return np.array(actions), np.array(rewards)
