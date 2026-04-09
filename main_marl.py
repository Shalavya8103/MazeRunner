import pygame
from maze_env import FixedGameState
from renderer import Renderer
from agents.marl import load_marl,get_action,get_state, ACTIONS

runner_qtable, chaser_qtable= load_marl()

def main():
    pygame.init()
    gs = FixedGameState()
    renderer=Renderer(gs)
    clock= pygame.time.Clock()
    running=True
    while running and not gs.game_over:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running = False
        state= get_state(gs)
        
        runner_action= get_action(state,runner_qtable, gs,gs.runner_pos,training=False, epsilon=0)
        dr,dc = ACTIONS[runner_action]
        gs.move_runner((gs.runner_pos[0] + dr, gs.runner_pos[1] + dc))
        chaser_action=get_action(state,chaser_qtable, gs,gs.chaser_pos, training=False,epsilon=0)
        dr,dc=ACTIONS[chaser_action]
        gs.move_chaser((gs.chaser_pos[0]+dr, gs.chaser_pos[1]+dc))

        gs.step_count += 1
        renderer.draw()
        clock.tick(5)
    print(f"Game over! Winner: {gs.winner} in {gs.step_count} steps")
    pygame.time.wait(2000)
    renderer.close()

if __name__ == "__main__":
    main()