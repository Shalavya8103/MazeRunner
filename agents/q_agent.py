import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maze_env import FixedGameState
from agents.minimax import chaser_move
from agents.astar import heuristic
import pickle

ACTIONS=[(-1,0),(1,0),(0,-1),(0,1)]
ALPHA= 0.1      
GAMMA= 0.95
EPSILON=1 
EPSILON_DECAY= 0.99995
NUM_EPISODES=100000
MAX_STEPS= 500
REWARD_CHECKPOINT= 100
REWARD_WIN=500
REWARD_CAUGHT=-300
REWARD_STEP=-2
REWARD_REVISIT= -5

def get_state(game_state):
    runner=tuple(int(x) for x in game_state.runner_pos)
    chaser=tuple(int(x) for x in game_state.chaser_pos)
    checkpoint= tuple(game_state.collected)
    return (runner, chaser, checkpoint)

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
    
def get_reward(old_state, new_state, oldpos_r):
    if new_state.game_over:
        if new_state.winner=="chaser":
            return REWARD_CAUGHT
        else:
            return REWARD_WIN
    if old_state[2]!=tuple(new_state.collected):
        return REWARD_CHECKPOINT
    target = None
    for i, cp in enumerate(new_state.checkpoints):
        if not new_state.collected[i]:
            target = tuple(int(x) for x in cp)
            break
    if target is None:
        target= tuple(int(x) for x in new_state.exit_pos)
    
    old_dist= heuristic(tuple(int(x) for x in oldpos_r), target)
    new_dist= heuristic(tuple(int(x) for x in new_state.runner_pos), target)
    shaping= old_dist-new_dist 
    return REWARD_STEP + shaping

def save_qtable(q_table, filename="q_tables/q_table.pkl"):
    with open(filename,"wb") as f:
        pickle.dump(q_table,f)
    print("Q-table saved to",filename)

def load_qtable(filename="q_tables/q_table.pkl"):
    with open(filename,"rb") as f:
        q_table = pickle.load(f)
    print("Q-table loaded from",filename)
    return q_table

    
def train():
    q_table={}
    epsilon= EPSILON
    for episode in range(NUM_EPISODES):
        gs=FixedGameState()
        state=get_state(gs)
        recent_positions= []
        chaser_hist= []
        while not gs.game_over and gs.step_count < MAX_STEPS:
            gs.step_count+=1
            oldpos_r= gs.runner_pos
            action=get_action(state,q_table,gs,True,epsilon=epsilon)
            dr,dc=ACTIONS[action]
            new_pos= (gs.runner_pos[0]+dr,gs.runner_pos[1]+dc)
            gs.move_runner(new_pos)

            chaser=chaser_move(gs,chaser_hist)
            if chaser:
                chaser_hist.append(gs.chaser_pos)
                if len(chaser_hist)>4:
                    chaser_hist.pop(0)
                gs.move_chaser(chaser)

            reward= get_reward(state,gs,oldpos_r)
            if new_pos in recent_positions:
                reward += REWARD_REVISIT
            recent_positions.append(new_pos)
            if len(recent_positions)>10:
                recent_positions.pop(0)

            new_state=get_state(gs)
            best_future=float("-inf")
            for a in range(len(ACTIONS)):
                value=q_table.get((new_state, a), 0)
                if value>best_future:
                    best_future=value
            current_q= q_table.get((state,action),0)
            q_table[(state,action)]=current_q+ ALPHA* (reward+ GAMMA* best_future- current_q)
            state= new_state
        epsilon = max(0.05, epsilon*EPSILON_DECAY)
        if episode % 500 == 0:
            print(f"Episode {episode}/{NUM_EPISODES}")
    return q_table



if __name__ == "__main__":
    print("Training Runnr")
    q_table = train()
    print("Q-table size:", len(q_table))
    save_qtable(q_table)
    

    