import random
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maze_env import GameState
from agents.minimax import chaser_move
from agents.astar import heuristic
import pickle

ACTIONS=[(-1,0),(1,0),(0,-1),(0,1)]
ALPHA= 0.1      
GAMMA= 0.9
EPSILON=1 
EPSILON_DECAY= 0.9995
NUM_EPISODES=20000
MAX_STEPS= 500
CURRICULUM_SWITCH= 8000  # episodes before switching from random to minimax chaser

REWARD_CHECKPOINT= 50
REWARD_WIN=200
REWARD_CAUGHT=-200
REWARD_STEP=-1

def get_state(gs):
    runner = tuple(int(x) for x in gs.runner_pos)
    chaser = tuple(int(x) for x in gs.chaser_pos)

    dr = chaser[0] - runner[0]
    dc = chaser[1] - runner[1]
    chaser_dir = (int(np.sign(dr)), int(np.sign(dc)))
    chaser_dist = min(heuristic(runner, chaser) // 3, 5)

    target = next(
        (cp for i, cp in enumerate(gs.checkpoints) if not gs.collected[i]),
        gs.exit_pos
    )
    tr = target[0] - runner[0]
    tc = target[1] - runner[1]
    target_dir = (int(np.sign(tr)), int(np.sign(tc)))
    target_dist = min(heuristic(runner, target) // 3, 5)

    walls = tuple(
        0 if gs.valid_move((runner[0] + ddr, runner[1] + ddc)) else 1
        for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
    )

    collected = tuple(gs.collected)

    return (chaser_dir, chaser_dist, target_dir, target_dist, walls, collected)

def get_action(state, q_table, game_state,training=False,epsilon=0):
    valid_actions=[]
    for i, (dr,dc) in enumerate(ACTIONS):
        nr,nc=game_state.runner_pos[0]+dr,game_state.runner_pos[1]+dc
        if game_state.valid_move((nr,nc)):
            valid_actions.append(i)
    if training and random.random()< epsilon:
        return random.choice(valid_actions)
  
    best_action= None
    best_value= float("-inf")
    for a in valid_actions:
        value= q_table.get((state,a),0)
        if value>best_value:
            best_value=value
            best_action=a
    
    if best_action is None:
        return random.choice(valid_actions)
    return best_action
    
def get_reward(old_state, new_state):
    if new_state.game_over:
        if new_state.winner == "chaser":
            return REWARD_CAUGHT
        else:
            return REWARD_WIN
    if old_state[5]!=tuple(new_state.collected):
        return REWARD_CHECKPOINT
    else:
        return REWARD_STEP

def save_qtable(q_table, filename="q_table.pkl"):
    with open(filename,"wb") as f:
        pickle.dump(q_table,f)
    print("Q-table saved to", filename)

def load_qtable(filename="q_table.pkl"):
    with open(filename, "rb") as f:
        q_table = pickle.load(f)
    print("Q-table loaded from", filename)
    return q_table

    
def random_chaser_move(gs):
    neighbors = gs.get_neighbors(gs.chaser_pos)
    return random.choice(neighbors) if neighbors else None

def train():
    q_table={}
    epsilon= EPSILON
    for episode in range(NUM_EPISODES):
        gs=GameState(rows=21,cols=21)
        state=get_state(gs)
        chaser_hist=[]
        use_minimax = episode >= CURRICULUM_SWITCH
        while not gs.game_over and gs.step_count < MAX_STEPS:
            gs.step_count+=1
            action=get_action(state,q_table,gs,True,epsilon=epsilon)
            [dr,dc]=ACTIONS[action]
            new_pos= (gs.runner_pos[0]+dr,gs.runner_pos[1]+dc)
            gs.move_runner(new_pos)
            if use_minimax:
                chaser=chaser_move(gs, chaser_hist)
                if chaser:
                    chaser_hist.append(gs.chaser_pos)
                    if len(chaser_hist) > 4:
                        chaser_hist.pop(0)
            else:
                chaser=random_chaser_move(gs)
            if chaser:
                gs.move_chaser(chaser)
            reward= get_reward(state,gs)
            new_state=get_state(gs)
            best_future=float("-inf")
            for a in range(len(ACTIONS)):
                value=q_table.get((new_state, a), 0)
                if value>best_future:
                    best_future=value
            current_q= q_table.get((state,action),0)
            q_table[(state,action)]=current_q+ ALPHA* (reward+ GAMMA* best_future- current_q)
            state= new_state
        epsilon = epsilon * EPSILON_DECAY
        if episode % 500 == 0:
            phase = "minimax" if use_minimax else "random"
            print(f"Episode {episode}/{NUM_EPISODES} [{phase} chaser] eps={epsilon:.3f}")
    return q_table



if __name__ == "__main__":
    print("Training Q-learning runner...")
    q_table = train()
    print(f"Training complete. Q-table size:", len(q_table))
    save_qtable(q_table)
    

    