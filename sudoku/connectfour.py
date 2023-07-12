import numpy as np
import re

"""
Utility function: Return a list of all consecutive board positions in state that satisfy regex
"""
def k_in_row(state, regex):
    # Return a list of all consecutive board positions in state that satisfy regex
    flipped = np.fliplr(state)
    sequences = []

    for i in range(state.shape[0]):
        sequences.extend(re.findall(regex, ''.join(state[i])))
        sequences.extend(re.findall(regex, ''.join(np.diag(state, k=-i))))
        sequences.extend(re.findall(regex, ''.join(np.diag(flipped, k=-i))))

    for j in range(state.shape[1]):
        sequences.extend(re.findall(regex, ''.join(state[:, j])))
        if j != 0:
            sequences.extend(re.findall(regex, ''.join(np.diag(state, k=j))))
            sequences.extend(re.findall(regex, ''.join(np.diag(flipped, k=j))))

    return sequences


"""
Functions to be used by alpha-beta search
"""
def terminal(state, k):
    # If the given state is terminal, return computed utility (positive for X win, negative for O win, 0 for draw)
    # Otherwise, return None
    if k_in_row(state, "X{" + str(k) + "}"): return 1
    if k_in_row(state, "O{" + str(k) + "}"): return -1
    if np.count_nonzero(state == '.') == 0: return 0
    return None


def eval(state, k):
    # Evaluate a non-terminal state based on both players' potential for winning the game
    score = terminal(state, k)
    if score is not None:
        return score

    score = 0
    possible_Xseq = k_in_row(state, "[X\.]{"+str(k)+",}")
    possible_Oseq = k_in_row(state, "[O\.]{"+str(k)+",}")
    score += sum([len(x)*x.count('X') for x in possible_Xseq])
    score -= sum([len(o)*o.count('O') for o in possible_Oseq])
    if score != 0:
        maxstr = max(possible_Xseq+possible_Oseq, key=len)
        score /= (k * len(maxstr) * (len(possible_Xseq)+len(possible_Oseq)))

    return score


def successors(state, player):
    # Given board state (2d NumPy array) and player to move, return list of all possible successor states
    return []


"""
Alpha-beta depth-limited search
Params: Board state (2d NumPy array), player ('X' or 'O'), connect-k value, optional maximum search depth
Return: Value and best successor state
"""
def alpha_beta_search(state, player, k, max_depth):
    if player == 'X':
        value, next = max_value(state, -float("inf"), float("inf"), k, 0, max_depth)
    else:
        value, next = min_value(state, -float("inf"), float("inf"), k, 0, max_depth)
    return value, next


def max_value(state, alpha, beta, k, depth, max_depth):
    return 0, None


def min_value(state, alpha, beta, k, depth, max_depth):
    return 0, None


"""
Set parameters in main function, which will call game_loop to simulate a game
"""
def game_loop(m, n, k, X_depth=float("inf"), O_depth=float("inf")):
    # Play a Connect-k game given grid size (mxn)
    # Optional search depth parameters for player X and player O
    state = np.full((m,n), '.')
    print("Connect", k, "on a", m, "by", n, "board")
    player = 'X'

    while state is not None:
        print(np.matrix(state), "\n")
        if player == 'X':
            value, state = alpha_beta_search(state, player, k, X_depth)
            player = 'O'
        else:
            value, state = alpha_beta_search(state, player, k, O_depth)
            player = 'X'

    if value > 0: print("X wins!")
    elif value < 0: print("O wins!")
    else: print("Draw!")


if __name__ == '__main__':
    m, n, k = 4, 4, 3
    game_loop(m, n, k)
