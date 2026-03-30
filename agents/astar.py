import heapq

def heuristic(a,b):
    r1,c1= a
    r2,c2 =b
    return abs(r1-r2)+ abs(c1-c2)

def astar(maze, start, goal):
    start= tuple(int(x) for x in start)
    goal= tuple(int(x) for x in goal)
    heap=[]
    heapq.heappush(heap,(0,start))
    came_from={}
    g_score= {start: 0}
    while heap:
        _, curr= heapq.heappop(heap)
        if curr== goal:
            if curr == start:
                return start 
            while curr!= start:
                node= curr
                curr= came_from[curr]
            return node
        
        dist=[(0,1),(1,0),(-1,0),(0,-1)]
        for dr,dc in dist:
            nr,nc= curr[0]+dr,curr[1]+dc
            if 0<= nr< maze.shape[0] and 0<= nc< maze.shape[1] and maze[nr][nc]== 0:
                ng= g_score[curr]+1
                if (nr, nc) not in g_score or ng< g_score[(nr,nc)]:
                    came_from[(nr,nc)]= curr 
                    g_score[(nr,nc)]=ng
                    f= ng+heuristic((nr,nc),goal)
                    heapq.heappush(heap,(f,(nr,nc)))
    return None

def eastar(maze, start, goal, chaser_pos, danger_rad=4):
    start= tuple(int(x) for x in start)
    goal= tuple(int(x) for x in goal)
    heap=[]
    heapq.heappush(heap,(0,start))
    came_from= {}
    g_score={start:0}
    
    while heap:
        _,curr=heapq.heappop(heap)
        if curr== goal:
            if curr== start:
                return start
            while curr!= start:
                node= curr
                curr=came_from[curr]
            return node
        dist=[(0,1),(1,0),(-1,0),(0,-1)]
        for dr,dc in dist:
            nr,nc= curr[0]+dr,curr[1]+dc
            if 0<= nr< maze.shape[0] and 0<= nc< maze.shape[1] and maze[nr][nc]== 0:
                chaser_dist= heuristic((nr,nc),chaser_pos)
                danger_penalty= max(0,danger_rad-chaser_dist)*3
                ng = g_score[curr]+1+danger_penalty
                if (nr, nc) not in g_score or ng< g_score[(nr,nc)]:
                    came_from[(nr,nc)]= curr 
                    g_score[(nr,nc)]=ng
                    f= ng+heuristic((nr,nc),goal)
                    heapq.heappush(heap,(f,(nr,nc)))
    return None

# if __name__ == "__main__":
#     import numpy as np
#     import sys
#     import os
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     from maze_env import GameState
    
#     gs = GameState(rows=21, cols=21)
#     next_step = astar(gs.maze, gs.runner_pos, gs.checkpoints[0])
#     print(f"Runner at: {gs.runner_pos}")
#     print(f"First checkpoint at: {gs.checkpoints[0]}")
#     print(f"Next step: {next_step}")

            


