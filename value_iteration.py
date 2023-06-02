import sys
import numpy as np
import matplotlib.pyplot as plt

def direction_calculations(dim, reward, gamma, mdp, i, j):
    # returns [up, down, left, right]
    directions = []
    
    # up
    if i == 0:
        directions.append(0)
    else:
        directions.append(reward + gamma * mdp[i-1][j])

    # down
    if i == dim - 1:
        directions.append(0)
    else:
        directions.append(reward + gamma * mdp[i+1][j])

    # left
    if j == 0:
        directions.append(0)
    else:
        directions.append(reward + gamma * mdp[i][j-1])

    # right
    if j == dim - 1:
        directions.append(0)
    else:
        directions.append(reward + gamma * mdp[i][j+1])
    
    return directions

def main():
    argv = sys.argv[1:]
    filename = argv[0]

    # read file
    with open(filename, 'r') as f:
        lines = f.readlines()

    dim = int(lines[0])  # dimension of grid
    gamma = float(lines[1])  # discount factor
    noise = [float(i) for i in lines[2].split(", ")] # noise for each action

    while len(noise) < 4:
        noise.append(0)

    # parse file
    grid = []
    for line in lines[4:]:
        grid.append(line.strip().split(","))

    mdp = [[0 for i in range(dim)] for j in range(dim)] 
    policies = [['' for i in range(dim)] for j in range(dim)]

    for i in range(dim):
        for j in range(dim):
            if grid[i][j] == 'X':
                mdp[i][j] = 0
            else:
                mdp[i][j] = float(grid[i][j])
                policies[i][j] = 'T' # terminal state
    
    convergence_threshold = 0.001
    convergence = False
    iterations = 0
    while not convergence:
        delta = 0
        for i in range(dim):
            for j in range(dim):
                temp = mdp[i][j]
                if grid[i][j] == 'X': # not terminal state
                    reward = 0    
                else: # terminal state
                    continue

                # calculate the value of each action
                directions = direction_calculations(dim, reward, gamma, mdp, i, j)
                up = noise[0] * directions[0] + noise[1] * directions[1] + noise[2] * directions[2] + noise[3] * directions[3]
                down = noise[0] * directions[1] + noise[1] * directions[2] + noise[2] * directions[3] + noise[3] * directions[0]
                left = noise[0] * directions[2] + noise[1] * directions[0] + noise[2] * directions[1] + noise[3] * directions[3]
                right = noise[0] * directions[3] + noise[1] * directions[0] + noise[2] * directions[1] + noise[3] * directions[2]

                # choose the action with the highest value
                mdp[i][j] = max(up, down, left, right)

                if mdp[i][j] == up:
                    policies[i][j] = 'U'
                elif mdp[i][j] == down:
                    policies[i][j] = 'D'
                elif mdp[i][j] == left:
                    policies[i][j] = 'L'
                else:
                    policies[i][j] = 'R'
        
                # calculate the difference between the old and new value
                delta = max(delta, abs(temp - mdp[i][j]))

        # check if the difference is less than the threshold
        if delta < convergence_threshold:
            convergence = True
        
        iterations += 1

    print("iterations: {}".format(iterations))

    # display arrows and terminal states as a grid
    fig, ax = plt.subplots()
    ax.set_xlim(-1, dim)
    ax.set_ylim(-1, dim)
    ax.set_xticks(np.arange(-0.5, dim, 1))
    ax.set_yticks(np.arange(-0.5, dim, 1))
    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Value Iteration")
    for i in range(dim):
        for j in range(dim):
            if policies[i][j] == 'T':
                ax.text(j, i, "{:.3f}".format(mdp[i][j]), ha="center", va="center", color="black")
            elif policies[i][j] == 'U':
                ax.arrow(j, i, 0, -0.4, head_width=0.1, head_length=0.1, fc='k', ec='k')
            elif policies[i][j] == 'D':
                ax.arrow(j, i, 0, 0.4, head_width=0.1, head_length=0.1, fc='k', ec='k')
            elif policies[i][j] == 'L':
                ax.arrow(j, i, -0.4, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
            else:
                ax.arrow(j, i, 0.4, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    plt.show()

if __name__ == '__main__':
    main()