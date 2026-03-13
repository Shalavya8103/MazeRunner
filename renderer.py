import pygame

BLACK= (0,0,0)
WHITE =(200,200,200)
BLUE =(50,120,255)
RED=(220,50,50)
YELLOW= (255,210,0)
GREEN =(50, 200,100)
DARK=(30,30, 30)
CELL_SIZE = 28

class Renderer:
    def __init__(self, game_state):
        pygame.init()
        self.gs= game_state
        self.width=self.gs.cols* CELL_SIZE
        self.height=self.gs.rows*CELL_SIZE
        self.screen= pygame.display.set_mode((self.width, self.height))
        self.font =pygame.font.SysFont("Arial", 14, bold=True)
        pygame.display.set_caption("Maze Runner")
        self.clock=pygame.time.Clock()

    def draw(self):
        self.screen.fill(DARK)
        for r in range(self.gs.rows):
            for c in range(self.gs.cols):
                rect=pygame.Rect(c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if self.gs.maze[r][c]==1:
                    pygame.draw.rect(self.screen,BLACK,rect)
                else:
                    pygame.draw.rect(self.screen,WHITE,rect)

        # exit
        er,ec = self.gs.exit_pos
        exit_rect = pygame.Rect(ec*CELL_SIZE,er*CELL_SIZE,CELL_SIZE,CELL_SIZE)
        pygame.draw.rect(self.screen,GREEN,exit_rect)
        self.screen.blit(self.font.render("E", True, BLACK),(ec*CELL_SIZE+8,er*CELL_SIZE+6))

        # checkpoints
        for i, cp in enumerate(self.gs.checkpoints):
            if not self.gs.collected[i]:
                cr,cc=cp
                cp_rect=pygame.Rect(cc *CELL_SIZE,cr* CELL_SIZE,CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen,YELLOW, cp_rect)
                self.screen.blit(self.font.render(str(i+1),True, BLACK), (cc*CELL_SIZE+9, cr*CELL_SIZE+6))

        # runner
        rr, rc = self.gs.runner_pos
        pygame.draw.circle(self.screen, BLUE,
            (rc *CELL_SIZE+CELL_SIZE//2,rr*CELL_SIZE+CELL_SIZE//2),CELL_SIZE//2- 2)

        # chaser
        cr, cc = self.gs.chaser_pos
        pygame.draw.circle(self.screen, RED,
            (cc* CELL_SIZE+CELL_SIZE//2, cr* CELL_SIZE+CELL_SIZE//2),CELL_SIZE//2 -2)

        pygame.display.flip()
    def tick(self, fps=10):
        self.clock.tick(fps)
    def close(self):
        pygame.quit()

from maze_env import GameState
gs = GameState(rows=21, cols=21)
renderer = Renderer(gs)
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    renderer.draw()
    renderer.tick()
renderer.close()