import sys
import os
from maze_env import GameState
from agents.q_agent import load_qtable, get_state, get_action, ACTIONS, MAX_STEPS
from agents.minimax import chaser_move

q_table = load_qtable()

wins = 0
losses = 0

for episode in range(1000):
    gs = GameState(rows=21, cols=21)
    state = get_state(gs)
    while not gs.game_over and gs.step_count < MAX_STEPS:
        gs.step_count += 1
        action = get_action(state, q_table, gs, training=False, epsilon=0)
        dr, dc = ACTIONS[action]
        new_pos = (gs.runner_pos[0] + dr, gs.runner_pos[1] + dc)
        gs.move_runner(new_pos)
        chaser = chaser_move(gs)
        if chaser:
            gs.move_chaser(chaser)
        state = get_state(gs)
    if gs.winner == "runner":
        wins += 1
    else:
        losses += 1

print(f"Win rate: {wins}/1000")
print(f"Loss rate: {losses}/1000")