import enchant, string
import matplotlib.pyplot as plt
from heapq import heappush, heappop

def successors(state):
    """
    Given a word, find all possible English word results from changing one letter.
    Return a list of (action, word) pairs, where action is the index of the
    changed letter.
    """
    d = enchant.Dict("en_US")
    child_states = []
    for i in range(len(state)):
        new = [state[:i]+x+state[i+1:] for x in string.ascii_lowercase]
        words = [x for x in new if d.check(x) and x != state]
        child_states = child_states + [(i, word) for word in words]
    return child_states

def f_helper(node):
    expanded = []
    nexts = successors(node['state'])
    for next in nexts:
       cur_node =  { 'state': next[1], 'parent': node, 'cost': node['cost']+1}
       expanded.append(cur_node)
    return expanded

"""
5.1: Best-first search
"""
def best_first_search(start, goal, f):
    """
    Inputs: Start state, goal state, priority function
    Returns node containing goal or None if no goal found, total nodes expanded,
    frontier size per iteration
    """
    node = {'state':start, 'parent':None, 'cost':0}
    frontier = []
    reached = { start: node }
    nodes_expanded = 0
    frontier_size = []

    res = None
    heappush(frontier, f(node, goal))
    while len(frontier):
        frontier_size.append(len(frontier))
        cur_node = heappop(frontier)
        # print(cur_node)
        if cur_node[2]['state'] == goal:
            res = cur_node[2]
            break
        expanded = f_helper(cur_node[2])
        nodes_expanded += len(expanded)
        for exp_node in expanded:
            if exp_node['state'] not in reached or exp_node['cost'] < reached[exp_node['state']]['cost']:
                reached[exp_node['state']] = exp_node
                heappush(frontier, f(exp_node, goal))

    return res, nodes_expanded, frontier_size


"""
5.2: Priority functions
"""
def f_dfs(node, goal=None):
    return (-node['cost'], node['state'], node)

def f_bfs(node, goal=None):
    return (node['cost'], node['state'], node)


def f_ucs(node, goal=None):
    return (node['cost'], node['state'], node)


def f_astar(node, goal):
    dist = 0
    for i in range(len(goal)):
        if node['state'][i] != goal[i]:
            dist += 1
    return (node['cost']+dist, node['state'], node)


def sequence(node):
    """
    Given a node, follow its parents back to the start state.
    Return sequence of words from start to goal.
    """
    words = [node['state']]
    while node['parent'] is not None:
        node = node['parent']
        words.insert(0, node['state'])
    return words


if __name__ == '__main__':
    start = 'cold'
    goal = 'warm'
    solution = best_first_search(start, goal, f_bfs)
    print(f'nodes_expanded {solution[1]}')
    print(sequence(solution[0]))
