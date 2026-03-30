import sys
import os
import pygame
from maze_env import GameState
from renderer import Renderer
from agents.astar import astar
from agents.astar import eastar
from agents.minimax import chaser_move
from agents.astar import heuristic
from agents.q_agent import load_qtable, get_action, get_state, ACTIONS

def runner_move(state):
    target = None
    best_dist = float('inf')
    for i,cp in enumerate(state.checkpoints):
        if not state.collected[i]:
            d= heuristic(state.runner_pos, cp)
            chaser_penalty= max(0,10-heuristic(state.chaser_pos,cp))
            if d+chaser_penalty<best_dist:
                best_dist= d+chaser_penalty
                target= cp
    if target is None:
        target = state.exit_pos
    return eastar(state.maze,state.runner_pos,target,state.chaser_pos)

q_table = load_qtable()

def Qrunner_move(gs):
    state= get_state(gs)
    action= get_action(state, q_table, gs)
    dr,dc= ACTIONS[action]
    return (gs.runner_pos[0]+dr, gs.runner_pos[1]+dc)

def main():
    pygame.init()
    gs=GameState(rows=21,cols=21)
    renderer= Renderer(gs)
    clock=pygame.time.Clock()
    running=True
    chaser_hist=[]
    while running and not gs.game_over:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False
    
        rmove = Qrunner_move(gs)
        if rmove:
            gs.move_runner(rmove)
        cmove= chaser_move(gs,chaser_hist)
        if cmove:
            chaser_hist.append(gs.chaser_pos)
            if len(chaser_hist)>4:
                chaser_hist.pop(0)
            gs.move_chaser(cmove)
        
        gs.step_count += 1
        renderer.draw()
        clock.tick(5) 


    print(f"Game over! Winner: {gs.winner} in {gs.step_count} steps")
    pygame.time.wait(2000)
    renderer.close()

if __name__ == "__main__":
    main()



