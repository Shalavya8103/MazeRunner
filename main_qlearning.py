import pygame
from maze_env import FixedGameState
from renderer import Renderer
from agents.minimax import chaser_move
from agents.q_agent import load_qtable, get_action, get_state, ACTIONS

q_table= load_qtable()

def runner_move(gs):
    state=get_state(gs)
    action=get_action(state,q_table,gs,training=False,epsilon=0.05)
    dr,dc=ACTIONS[action]
    return (gs.runner_pos[0]+dr,gs.runner_pos[1]+dc)

def main():
    pygame.init()
    gs=FixedGameState()
    renderer= Renderer(gs)
    clock=pygame.time.Clock()
    running=True
    chaser_hist=[]
    visited= {}
    while running and not gs.game_over:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False

        state_key=gs.runner_pos
        visited[state_key]= visited.get(state_key,0)+1
        if visited[state_key]>=5:
            epsilon=1 
        elif visited[state_key] >= 3:
            epsilon= 0.5
        else: 
            epsilon=0.05
        state= get_state(gs)
        action= get_action(state,q_table,gs,training=False, epsilon=epsilon)
        dr,dc=ACTIONS[action]
        rmove=(gs.runner_pos[0]+dr,gs.runner_pos[1]+dc)

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

    print(f"Game over. Winner: {gs.winner} in {gs.step_count} steps")
    pygame.time.wait(2000)
    renderer.close()

if __name__ == "__main__":
    main()



