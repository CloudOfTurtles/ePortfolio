# %% [markdown]
# # Treasure Hunt Game Notebook
# 
# ## Read and Review Your Starter Code
# The theme of this project is a popular treasure hunt game in which the player needs to find the treasure before the pirate does. While you will not be developing the entire game, you will write the part of the game that represents the intelligent agent, which is a pirate in this case. The pirate will try to find the optimal path to the treasure using deep Q-learning. 
# 
# You have been provided with two Python classes and this notebook to help you with this assignment. The first class, TreasureMaze.py, represents the environment, which includes a maze object defined as a matrix. The second class, GameExperience.py, stores the episodes – that is, all the states that come in between the initial state and the terminal state. This is later used by the agent for learning by experience, called "exploration". This notebook shows how to play a game. Your task is to complete the deep Q-learning implementation for which a skeleton implementation has been provided. The code blocks you will need to complete has #TODO as a header.
# 
# First, read and review the next few code and instruction blocks to understand the code that you have been given.

# %%
from __future__ import print_function
import os, sys, time, datetime, json, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt
from TreasureMaze import TreasureMaze
from GameExperience import GameExperience
#%matplotlib inline

# %% [markdown]
# The following code block contains an 8x8 matrix that will be used as a maze object:

# %%
maze = np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.]
])

# %% [markdown]
# This helper function allows a visual representation of the maze object:

# %%
def show(qmaze):
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row,col in qmaze.visited:
        canvas[row,col] = 0.6
    pirate_row, pirate_col, _ = qmaze.state
    canvas[pirate_row, pirate_col] = 0.3   # pirate cell
    canvas[nrows-1, ncols-1] = 0.9 # treasure cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    return img

# %% [markdown]
# The pirate agent can move in four directions: left, right, up, and down. 
# 
# While the agent primarily learns by experience through exploitation, often, the agent can choose to explore the environment to find previously undiscovered paths. This is called "exploration" and is defined by epsilon. This value is typically a lower value such as 0.1, which means for every ten attempts, the agent will attempt to learn by experience nine times and will randomly explore a new path one time. You are encouraged to try various values for the exploration factor and see how the algorithm performs.

# %%
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3


# Exploration factor
epsilon = 0.1

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)


# %% [markdown]
# The sample code block and output below show creating a maze object and performing one action (DOWN), which returns the reward. The resulting updated environment is visualized.

# %%
qmaze = TreasureMaze(maze)
canvas, reward, game_over = qmaze.act(DOWN)
print("reward=", reward)
show(qmaze)

# %% [markdown]
# This function simulates a full game based on the provided trained model. The other parameters include the TreasureMaze object and the starting position of the pirate.

# %%
def play_game(model, qmaze, pirate_cell):
    qmaze.reset(pirate_cell)
    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False

# %% [markdown]
# This function helps you to determine whether the pirate can win any game at all. If your maze is not well designed, the pirate may not win any game at all. In this case, your training would not yield any result. The provided maze in this notebook ensures that there is a path to win and you can run this method to check.

# %%
def completion_check(model, qmaze):
    for cell in qmaze.free_cells:
        if not qmaze.valid_actions(cell):
            return False
        if not play_game(model, qmaze, cell):
            return False
    return True

# %% [markdown]
# The code you have been given in this block will build the neural network model. Review the code and note the number of layers, as well as the activation, optimizer, and loss functions that are used to train the model.

# %%
def build_model(maze):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model

# %% [markdown]
# # #TODO: Complete the Q-Training Algorithm Code Block
# 
# This is your deep Q-learning implementation. The goal of your deep Q-learning implementation is to find the best possible navigation sequence that results in reaching the treasure cell while maximizing the reward. In your implementation, you need to determine the optimal number of epochs to achieve a 100% win rate.
# 
# You will need to complete the section starting with #pseudocode. The pseudocode has been included for you.

# %%
def qtrain(model, maze, **opt):
    # exploration factor
    global epsilon

    # number of epochs
    n_epoch = opt.get('n_epoch', 15000)

    # maximum memory to store episodes
    max_memory = opt.get('max_memory', 1000)

    # maximum data size for training
    data_size = opt.get('data_size', 50)

    # start time
    start_time = datetime.datetime.now()

    # Construct environment/game from numpy array: maze (see above)
    qmaze = TreasureMaze(maze)

    # Initialize experience replay object
    experience = GameExperience(model, max_memory=max_memory)

    win_history = [] # history of win/lose game
    hsize = qmaze.maze.size//2 # history window size
    win_rate = 0.0

    # Loop through epochs
    for epoch in range(n_epoch):
        # Agent_cell = randomly select a free cell
        agent_cell = qmaze.free_cells[np.random.randint(0, len(qmaze.free_cells))]

        # Reset the maze with agent set to above position
        qmaze.reset(agent_cell)

        # envstate = Environment.current_state
        envstate = qmaze.observe()

        n_episodes = 0  # Counter for episodes in an epoch
        loss = 0.0      # Variable to accumulate loss within an epoch

        # While state is not game over
        while True:
            previous_envstate = envstate

            # Action = randomly choose action (left, right, up, down) either by exploration or by exploitation
            if np.random.rand() < epsilon:
                action = np.random.choice([LEFT, UP, RIGHT, DOWN])
            else:
                # Choose action with the highest Q-value
                q = model.predict(previous_envstate)
                action = np.argmax(q[0])

            # Take action and observe the environment
            envstate, reward, game_status = qmaze.act(action)

            # Episode = [previous_envstate, action, reward, envstate, game_status]
            episode = [previous_envstate, action, reward, envstate, game_status]

            # Store episode in Experience replay object
            experience.remember(episode)

            # Train neural network model and evaluate loss
            # Get data for training from experience replay
            inputs, targets = experience.get_data(data_size=data_size)
            if inputs is not None:
                # Fit the model
                loss += model.train_on_batch(inputs, targets)

            # Increment episode counter
            n_episodes += 1

            # Check if game is over
            if game_status == 'win':
                win_history.append(1)  # Append win to history
                win_rate = sum(win_history[-hsize:]) / hsize  # Calculate win rate
                break
            elif game_status == 'lose':
                win_history.append(0)  # Append loss to history
                win_rate = sum(win_history[-hsize:]) / hsize  # Calculate win rate
                break

        # Print epoch details
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))

        # Check conditions for adjusting epsilon and completing training
        if win_rate > 0.9:
            epsilon = 0.05
        if sum(win_history[-hsize:]) == hsize and completion_check(model, qmaze):
            print(f"Reached 100% win rate at epoch: {epoch}")
            break

    # Determine the total time for training
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)

    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
    return seconds

# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)

# %%


# %% [markdown]
# ## Test Your Model
# 
# Now we will start testing the deep Q-learning implementation. To begin, select **Cell**, then **Run All** from the menu bar. This will run your notebook. As it runs, you should see output begin to appear beneath the next few cells. The code below creates an instance of TreasureMaze.

# %%
qmaze = TreasureMaze(maze)
show(qmaze)

# %% [markdown]
# In the next code block, you will build your model and train it using deep Q-learning. Note: This step takes several minutes to fully run.

# %%
model = build_model(maze)
qtrain(model, maze, epochs=1000, max_memory=8*maze.size, data_size=32)

# %% [markdown]
# This cell will check to see if the model passes the completion check. Note: This could take several minutes.

# %%
completion_check(model, qmaze)
show(qmaze)

# %% [markdown]
# This cell will test your model for one game. It will start the pirate at the top-left corner and run play_game. The agent should find a path from the starting position to the target (treasure). The treasure is located in the bottom-right corner.

# %%
pirate_start = (0, 0)
play_game(model, qmaze, pirate_start)
show(qmaze)

# %% [markdown]
# ## Save and Submit Your Work
# After you have finished creating the code for your notebook, save your work. Make sure that your notebook contains your name in the filename (e.g. Doe_Jane_ProjectTwo.ipynb). This will help your instructor access and grade your work easily. Download a copy of your IPYNB file and submit it to Brightspace. Refer to the Jupyter Notebook in Apporto Tutorial if you need help with these tasks.


