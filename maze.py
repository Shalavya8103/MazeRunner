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
            
