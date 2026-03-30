import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.astar import heuristic

DEPTH_LIMIT=2

def minimax(state, depth, alpha, beta, is_maximizing):
    if state.game_over:
        if state.winner == "chaser":
            return -1000
        else:
            return 1000
    if depth == 0:
        return evaluate(state)
    if is_maximizing:  #for the runner
        max_eval= float('-inf')
        for neighbor in state.get_neighbors(state.runner_pos):
            nstate= state.copy()
            nstate.move_runner(neighbor)
            eval=minimax(nstate, depth-1, alpha, beta, False)
            max_eval=max(max_eval, eval)
            alpha=max(alpha, eval)
            if beta<=alpha:
                break 
        return max_eval
    else: #for chaser 
        min_eval= float("inf")
        for neighbor in state.get_neighbors(state.chaser_pos):
            n_state=state.copy()
            n_state.move_chaser(neighbor)
            eval=minimax(n_state,depth-1,alpha,beta,True)
            min_eval=min(min_eval,eval)
            beta=min(beta,eval)
            if beta<=alpha:
                break
        return min_eval
            
    
def evaluate(state):
    chaser=state.chaser_pos
    runner=state.runner_pos
    r_dist=heuristic(chaser,runner)
    next_target = None
    for i, cp in enumerate(state.checkpoints):
        if not state.collected[i]:
            next_target = cp
            break
    if next_target is None:
        next_target = state.exit_pos
    
    rt_target = heuristic(runner, next_target)
    return r_dist + 0.5 * rt_target


def chaser_move(state,hist=[]):
    state.chaser_pos=tuple(int(x) for x in state.chaser_pos)
    state.runner_pos=tuple(int(x) for x in state.runner_pos)
    best_move= None
    best_score=float("inf")
    for neighbor in state.get_neighbors(state.chaser_pos):
        if neighbor in hist[-2:]:
            continue
        n_state= state.copy()
        n_state.move_chaser(neighbor)
        score=minimax(n_state,DEPTH_LIMIT,float("-inf"),float("inf"),True)
        dist= heuristic(neighbor,state.runner_pos)
        score=score*1000+dist
        if score<best_score:
            best_score=score
            best_move= neighbor
    if best_move is None:
        best_move=chaser_move(state,[])
    return best_move

# if __name__ == "__main__":
#     import sys
#     import os
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     from maze_env import GameState
    
#     gs = GameState(rows=21, cols=21)
#     print(f"Chaser at: {gs.chaser_pos}")
#     move = chaser_move(gs)
#     print(f"Chaser best move: {move}")