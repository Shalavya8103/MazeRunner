import numpy as np
import random

class MazeGenerator:
    def __init__(self, rows, cols):
        self.rows= rows if rows%2==1 else rows+1
        self.cols=cols if cols%2==1 else cols+1

    def _init_grid(self):
        self.grid = np.ones((self.rows, self.cols),dtype=int) 
    
    def _frontier(self,r,c, frontier): 
        dirs=[(-2,0),(2,0),(0,-2),(0,2)] 
        for dr, dc in dirs:
            newr= r+dr
            newc=c+dc
            if 0<=newr<self.rows and 0<=newc<self.cols and self.grid[newr][newc] == 1 and (newr, newc) not in frontier:
                frontier.append((newr,newc))

    def _prims(self):
        self.grid[1][1]=0
        frontier=[]
        self._frontier(1,1,frontier)
        while len(frontier)!=0:
            fr,fc = random.choice(frontier)
            frontier.remove((fr,fc))
            dirs=[(-2,0),(2,0),(0,-2),(0,2)]
            open_cells=[]
            for i,j in dirs:
                nr,nc=fr+i, fc+j
                if 0<=nr<self.rows and 0<=nc<self.cols:
                    if self.grid[nr][nc]==0:
                        open_cells.append((i,j))
            if len(open_cells)== 1:
                i,j=open_cells[0]
                self.grid[fr + i//2][fc + j//2]=0
                self.grid[fr][fc]=0 
                self._frontier(fr, fc, frontier)

    def _add_loops(self, prob=0.6):
        for r in range(1,self.rows-1):
            for c in range(1,self.cols-1):
                if self.grid[r][c]==1:
                    open_cells=0
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc = r+dr,c+dc
                        if self.grid[nr][nc]==0:
                            open_cells+=1
                    if open_cells>=2:
                        if random.random()<prob:
                            self.grid[r][c]=0
      
    def generate(self):
        self._init_grid()
        self._prims()
        self._add_loops()
        return self.grid
            
class GameState:
    def __init__(self, maze=None, rows=21, cols=21):
        if maze is None:
            mg=MazeGenerator(rows,cols)
            maze= mg.generate()
        self.maze= maze
        self.rows=maze.shape[0]
        self.cols= maze.shape[1]
        self.open_cells=self.get_open()
        self.step_count= 0
        self.game_over=False
        self.winner =None
        self.place_entities()

    def get_open(self):
        cells=np.argwhere(self.maze==0)
        return [tuple(c) for c in cells]


    def place_entities(self):
        positions= random.sample(self.open_cells,6)
        self.runner_pos=positions[0]
        self.chaser_pos= positions[1]
        self.checkpoints =list(positions[2:5])
        self.exit_pos=positions[5]
        self.collected=[False, False,False]


    def valid_move(self, pos):
        r,c=pos
        if 0<= r<self.rows and 0<=c<self.cols:
            return self.maze[r][c]==0
        return False

    def get_neighbors(self, pos):
        r,c=pos
        neighbors= []
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc=r+dr,c+dc
            if self.valid_move((nr,nc)):
                neighbors.append((nr,nc))
        return neighbors

    def move_runner(self, new_pos):
        if not self.valid_move(new_pos):
            return
        self.runner_pos=new_pos
        for i,cp in enumerate(self.checkpoints):
            if not self.collected[i] and self.runner_pos==cp:
                self.collected[i] = True
        if all(self.collected) and self.runner_pos==self.exit_pos:
            self.game_over =True
            self.winner= "runner"
        if self.runner_pos==self.chaser_pos:
            self.game_over= True
            self.winner ="chaser"


    def move_chaser(self, new_pos):
        if not self.valid_move(new_pos):
            return
        self.chaser_pos=new_pos
        if self.chaser_pos==self.runner_pos:
            self.game_over=True
            self.winner="chaser"

    def all_checkpoints_collected(self):
        return all(self.collected)

    def copy(self):
        new_state= GameState.__new__(GameState)
        new_state.maze= self.maze
        new_state.rows =self.rows
        new_state.cols=self.cols
        new_state.open_cells = self.open_cells
        new_state.runner_pos= self.runner_pos
        new_state.chaser_pos =self.chaser_pos
        new_state.checkpoints=self.checkpoints.copy()
        new_state.exit_pos= self.exit_pos
        new_state.collected =self.collected.copy()
        new_state.step_count= self.step_count
        new_state.game_over=self.game_over
        new_state.winner=self.winner
        return new_state

    # def __repr__(self):
    #     return (f"Runner: {self.runner_pos}, Chaser: {self.chaser_pos}, "
    #             f"Checkpoints: {self.checkpoints}, Collected: {self.collected}, "
    #             f"Exit: {self.exit_pos}, Steps: {self.step_count}, Winner: {self.winner}")

if __name__ == "__main__":
    gs = GameState(rows=21, cols=21)
    print(gs)