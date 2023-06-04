import sys
import numpy as np
import matplotlib.pyplot as plt
import random

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

    policy = [['' for i in range(dim)] for j in range(dim)]
    mdp = [[0 for i in range(dim)] for j in range(dim)]

    for i in range(dim):
        for j in range(dim):
            if grid[i][j] == 'X':
                mdp[i][j] = 0
                policy[i][j] = random.choice(['U', 'D', 'L', 'R'])
            else:
                mdp[i][j] = float(grid[i][j])
                policy[i][j] = 'T'

    # policy iteration
    policy_stable = False
    iterations = 0
    while not policy_stable:
        # policy evaluation
        delta = 1
        while delta > 0.001:
            delta = 0
            for i in range(dim):
                for j in range(dim):
                    temp = mdp[i][j]
                    if grid[i][j] == 'X': # not terminal state
                        reward = 0 # can have a more sophisticated reward function here
                    else: # terminal state
                        continue

                    # calculate the value of each action
                    directions = direction_calculations(dim, reward, gamma, mdp, i, j)
                    match policy[i][j]:
                        case 'U':
                            mdp[i][j] = noise[0] * directions[0] + noise[1] * directions[1] + noise[2] * directions[2] + noise[3] * directions[3]
                        case 'D':
                            mdp[i][j] = noise[0] * directions[1] + noise[1] * directions[2] + noise[2] * directions[3] + noise[3] * directions[0]
                        case 'L':
                            mdp[i][j] = noise[0] * directions[2] + noise[1] * directions[0] + noise[2] * directions[1] + noise[3] * directions[3]
                        case 'R':
                            mdp[i][j] = noise[0] * directions[3] + noise[1] * directions[0] + noise[2] * directions[1] + noise[3] * directions[2]
                    
                    delta = max(delta, abs(temp - mdp[i][j]))

        # policy improvement
        policy_stable = True
        for i in range(dim):
            for j in range(dim):
                if grid[i][j] == 'X':
                    old_action = policy[i][j]
                    directions = direction_calculations(dim, reward, gamma, mdp, i, j)
                    up = noise[0] * directions[0] + noise[1] * directions[1] + noise[2] * directions[2] + noise[3] * directions[3]
                    down = noise[0] * directions[1] + noise[1] * directions[2] + noise[2] * directions[3] + noise[3] * directions[0]
                    left = noise[0] * directions[2] + noise[1] * directions[0] + noise[2] * directions[1] + noise[3] * directions[3]
                    right = noise[0] * directions[3] + noise[1] * directions[0] + noise[2] * directions[1] + noise[3] * directions[2]

                    max_value = max(up, down, left, right)
                    if max_value == up:
                        policy[i][j] = 'U'
                    elif max_value == down:
                        policy[i][j] = 'D'
                    elif max_value == left:
                        policy[i][j] = 'L'
                    elif max_value == right:
                        policy[i][j] = 'R'

                    if old_action != policy[i][j]:
                        policy_stable = False

        iterations += 1

    print("iterations: ", iterations)

    # plot
    fig, ax = plt.subplots()
    ax.set_xlim(-1, dim)
    ax.set_ylim(-1, dim)
    ax.set_xticks(np.arange(-0.5, dim, 1))
    ax.set_yticks(np.arange(-0.5, dim, 1))
    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Policy Iteration")
    for i in range(dim):
        for j in range(dim):
            if policy[i][j] == 'T':
                ax.text(j, dim-i-1, "{:.3f}".format(mdp[i][j]), ha="center", va="center", color="black")
            else:
                directions = direction_calculations(dim, reward, gamma, mdp, i, j)
                up = noise[0] * directions[0] + noise[1] * directions[1] + noise[2] * directions[2] + noise[3] * directions[3]
                down = noise[0] * directions[1] + noise[1] * directions[2] + noise[2] * directions[3] + noise[3] * directions[0]
                left = noise[0] * directions[2] + noise[1] * directions[0] + noise[2] * directions[1] + noise[3] * directions[3]
                right = noise[0] * directions[3] + noise[1] * directions[0] + noise[2] * directions[1] + noise[3] * directions[2]

                ax.text(j, dim-i-1+0.4, "{:.2f}".format(up), ha="center", va="top", color="black", fontsize=6)
                ax.text(j, dim-i-1-0.4, "{:.2f}".format(down), ha="center", va="bottom", color="black", fontsize=6)
                ax.text(j-0.1, dim-i-1, "{:.2f}".format(left), ha="right", va="center", color="black", fontsize=6)
                ax.text(j+0.1, dim-i-1, "{:.2f}".format(right), ha="left", va="center", color="black", fontsize=6)

                # highlight the action with the highest value
                if policy[i][j] == 'U':
                    ax.text(j, dim-i-1+0.4, "{:.2f}".format(up), ha="center", va="top", color="green", fontsize=6)
                elif policy[i][j] == 'D':
                    ax.text(j, dim-i-1-0.4, "{:.2f}".format(down), ha="center", va="bottom", color="green", fontsize=6)
                elif policy[i][j] == 'L':
                    ax.text(j-0.1, dim-i-1, "{:.2f}".format(left), ha="right", va="center", color="green", fontsize=6)
                else:
                    ax.text(j+0.1, dim-i-1, "{:.2f}".format(right), ha="left", va="center", color="green", fontsize=6)

    plt.show()

if __name__ == '__main__':
    main()