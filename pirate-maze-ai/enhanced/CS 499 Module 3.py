from __future__ import print_function
import datetime, random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, PReLU
import matplotlib.pyplot as plt

from TreasureMaze import TreasureMaze
from GameExperience import GameExperience

"""
Enhanced version of the CS-370 Treasure Hunt Q-learning agent.

Key improvements (for CS-499 Algorithms & Data Structures artifact enhancement):
- Fixed the epoch parameter so qtrain now correctly honors epochs/n_epoch.
- Added optional reproducible training via a random seed.
- Introduced configurable epsilon-greedy decay for better exploration/exploitation balance.
- Added clearer comments and small structural cleanups for readability and maintainability.
"""

# Maze definition
maze = np.array([
    [1., 0., 1., 1., 1., 1., 1., 1.],
    [1., 0., 1., 1., 1., 0., 1., 1.],
    [1., 1., 1., 1., 0., 1., 0., 1.],
    [1., 1., 1., 0., 1., 1., 1., 1.],
    [1., 1., 0., 1., 1., 1., 1., 1.],
    [1., 1., 1., 0., 1., 0., 0., 0.],
    [1., 1., 1., 0., 1., 1., 1., 1.],
    [1., 1., 1., 1., 0., 1., 1., 1.]
])

def show(qmaze):
    """Visualize the maze and the pirate's path using matplotlib."""
    plt.figure()
    plt.grid(True)
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row, col in qmaze.visited:
        canvas[row, col] = 0.6
    pirate_row, pirate_col, _ = qmaze.state
    canvas[pirate_row, pirate_col] = 0.3   # pirate cell
    canvas[nrows - 1, ncols - 1] = 0.9     # treasure cell
    plt.imshow(canvas, interpolation="none", cmap="gray")
    plt.title("Treasure Maze")
    plt.show()

# Actions and epsilon-greedy configuration
LEFT  = 0
UP    = 1
RIGHT = 2
DOWN  = 3

# Global exploration factor (managed inside qtrain)
epsilon = 0.1

actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}
num_actions = len(actions_dict)

def play_game(model, qmaze, pirate_cell):
    """
    Run a single game using a trained model starting from pirate_cell.
    Returns True if the agent reaches the treasure, otherwise False.
    """
    qmaze.reset(pirate_cell)
    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        # greedy action selection
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False

def completion_check(model, qmaze):
    """
    Check if the trained model can win starting from every valid free cell.
    This is used as a strong stopping condition for training.
    """
    for cell in qmaze.free_cells:
        if not qmaze.valid_actions(cell):
            return False
        if not play_game(model, qmaze, cell):
            return False
    return True

def build_model(maze):
    """
    Build the neural network that approximates the Q-function.
    Input size is the flattened maze; output size is the number of actions.
    """
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model

def qtrain(model, maze, **opt):
    """
    Deep Q-learning training loop.

    Parameters (via **opt):
        n_epoch / epochs : number of training epochs (outer episodes)
        max_memory       : replay buffer size
        data_size        : batch size sampled from replay
        epsilon          : initial exploration rate
        epsilon_min      : minimum exploration rate
        epsilon_decay    : multiplicative decay applied each epoch
        seed             : optional random seed for reproducibility
    """
    global epsilon

    # number of epochs (support both n_epoch and epochs for compatibility)
    n_epoch = opt.get('n_epoch', opt.get('epochs', 15000))

    # replay + batch configuration
    max_memory = opt.get('max_memory', 1000)
    data_size  = opt.get('data_size', 50)

    # epsilon-greedy configuration
    epsilon       = opt.get('epsilon', epsilon)
    epsilon_min   = opt.get('epsilon_min', 0.05)
    epsilon_decay = opt.get('epsilon_decay', 0.995)

    # Optional reproducibility
    seed = opt.get('seed', None)
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # start time
    start_time = datetime.datetime.now()

    # Construct environment/game from numpy array: maze
    qmaze = TreasureMaze(maze)

    # Initialize experience replay object
    experience = GameExperience(model, max_memory=max_memory)

    win_history = []                 # history of win/lose outcomes
    hsize = qmaze.maze.size // 2     # history window size for win rate
    win_rate = 0.0

    # Main training loop over epochs
    for epoch in range(n_epoch):
        # Select a starting cell at random from free cells
        agent_cell = qmaze.free_cells[np.random.randint(0, len(qmaze.free_cells))]

        # Reset the maze for a new game
        qmaze.reset(agent_cell)
        envstate = qmaze.observe()

        n_episodes = 0   # episodes (steps) taken in this epoch
        loss = 0.0       # accumulated loss for this epoch

        # Run the game until we either win or lose
        while True:
            previous_envstate = envstate

            # ---- Exploration vs Exploitation ----
            if np.random.rand() < epsilon:
                # Explore: random action
                action = np.random.choice([LEFT, UP, RIGHT, DOWN])
            else:
                # Exploit: choose action with highest predicted Q-value
                q = model.predict(previous_envstate)
                action = np.argmax(q[0])

            # Apply action in the environment
            envstate, reward, game_status = qmaze.act(action)

            # Store episode in replay memory
            episode = [previous_envstate, action, reward, envstate, game_status]
            experience.remember(episode)

            # Sample a batch from memory and train the network
            inputs, targets = experience.get_data(data_size=data_size)
            if inputs is not None:
                loss += model.train_on_batch(inputs, targets)

            n_episodes += 1

            # Check terminal condition (win or lose)
            if game_status in ('win', 'lose'):
                win_history.append(1 if game_status == 'win' else 0)
                # Compute win rate over the last hsize games (avoid division by zero)
                history = win_history[-hsize:] if len(win_history) >= hsize else win_history
                win_rate = float(sum(history)) / len(history)
                break

        # Apply epsilon decay at the end of the epoch (but keep it above epsilon_min)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # If we are consistently winning, clamp epsilon a bit lower to focus on exploitation
        if win_rate > 0.9:
            epsilon = max(epsilon_min, 0.05)

        # Progress report for this epoch
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = (
            "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | "
            "Win count: {:d} | Win rate: {:.3f} | epsilon: {:.3f} | time: {}"
        )
        print(template.format(
            epoch, n_epoch - 1, loss, n_episodes,
            sum(win_history), win_rate, epsilon, t)
        )

        # Strong stopping condition:
        # - 100% win rate over the last hsize games
        # - model passes completion_check (can win from every free cell)
        if len(win_history) >= hsize and sum(win_history[-hsize:]) == hsize:
            if completion_check(model, qmaze):
                print("Reached 100% win rate over last {} games at epoch {}."
                      .format(hsize, epoch))
                break

    # Final timing summary
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" %
          (epoch, max_memory, data_size, t))
    return seconds

def format_time(seconds):
    """Pretty-print elapsed time in seconds, minutes, or hours."""
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)

# --- Script entry point ---

if __name__ == "__main__":
    # Build maze and model
    qmaze = TreasureMaze(maze)

    print("Initial maze configuration:")
    # show(qmaze)

    model = build_model(maze)

    # Train with enhanced configuration
    qtrain(
        model,
        maze,
        epochs=200,                
        max_memory=4 * maze.size,
        data_size=16,
        epsilon=0.2,                
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=42                     
    )

    # Final completion check and a single demo game from (0, 0)
    print("Running completion check...")
    completion_check(model, qmaze)

    pirate_start = (0, 0)
    print("Playing a demo game from start cell:", pirate_start)
    play_game(model, qmaze, pirate_start)
    show(qmaze)
