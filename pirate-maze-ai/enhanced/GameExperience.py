# This class stores the episodes, all the states that come in between the initial state and the terminal state. 
# This is later used by the agent for learning by experience, called "exploration". 

import numpy as np

class GameExperience(object):
    
    # model = neural network model
    # max_memory = number of episodes to keep in memory. The oldest episode is deleted to make room for a new episode.
    # discount = discount factor; determines the importance of future rewards vs. immediate rewards
    
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]
    
    # Stores episodes in memory
    
    def remember(self, episode):
        # episode = [envstate, action, reward, envstate_next, game_over]
        # memory[i] = episode
        # envstate == flattened 1d maze cells info, including pirate cell (see method: observe)
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    # Predicts the next action based on the current environment state        
    def predict(self, envstate):
        return self.model.predict(envstate)[0]

    # Returns input and targets from memory, defaults to data size of 10
    def get_data(self, data_size=10):
        # envstate is stored as a 2D array: shape (1, env_size)
        env_size = self.memory[0][0].shape[1]
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)

        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))

        # sample a batch of episodes from memory
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            # NOTE: last value is a game_status string: 'win', 'lose', or 'not_over'
            envstate, action, reward, envstate_next, game_status = self.memory[j]

            inputs[i] = envstate

            # start from current Q-values for this state
            targets[i] = self.predict(envstate)

            # max future Q-value Q(s', a')
            Q_sa = np.max(self.predict(envstate_next))

            # treat 'win' and 'lose' as terminal, 'not_over' as non-terminal
            terminal = (game_status != 'not_over')

            if terminal:
                # terminal: just the immediate reward
                targets[i, action] = reward
            else:
                # non-terminal: reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa

        return inputs, targets

