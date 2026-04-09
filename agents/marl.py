import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maze_env import FixedGameState
from agents.astar import heuristic
import pickle

ACTIONS= [(-1,0),(1,0),(0,-1),(0,1)]
ALPHA= 0.1
GAMMA= 0.9
EPSILON= 1.0
#EPSILON_DECAY= 0.99995
RUNNER_EPSILON_DECAY= 0.99995 
CHASER_EPSILON_DECAY= 0.99995
NUM_EPISODES= 100000
MAX_STEPS= 500

# runner rewards
R_CHECKPOINT= 100
R_WIN= 500
R_CAUGHT= -500
R_STEP= -2
R_REVISIT= -5

# chaser rewards
C_CATCH= 500
C_LOSS= -500
C_STEP= -1

def get_state(game_state):
    runner= tuple(int(x) for x in game_state.runner_pos)
    chaser= tuple(int(x) for x in game_state.chaser_pos)
    collected=tuple(game_state.collected)
    return (runner,chaser,collected)

def runner_reward(old_state, new_state, oldpos_r):
    if new_state.game_over:
        if new_state.winner=="chaser":
            return R_CAUGHT
        else:
            return R_WIN
    if old_state[2]!=tuple(new_state.collected):
        return R_CHECKPOINT
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
    return R_STEP + shaping

def chaser_reward(new_state, old_chaser_pos):
    if new_state.game_over:
        if new_state.winner == "chaser":
            return C_CATCH
        else:
            return C_LOSS
    old_dist= heuristic(tuple(int(x) for x in old_chaser_pos), tuple(int(x) for x in new_state.runner_pos))
    new_dist= heuristic(tuple(int(x) for x in new_state.chaser_pos), tuple(int(x) for x in new_state.runner_pos))
    shaping= old_dist-new_dist
    return C_STEP + shaping

def get_action(state,q_table, game_state,pos, training=False, epsilon=0):
    valid_actions = []
    for i, (dr, dc) in enumerate(ACTIONS):
        nr,nc= pos[0]+dr, pos[1]+dc
        if game_state.valid_move((nr,nc)):
            valid_actions.append(i)
    if training and random.random()<epsilon:
        return random.choice(valid_actions)
    best_action= None
    best_value= float("-inf")
    for a in valid_actions:
        value= q_table.get((state, a), 0)
        if value>best_value:
            best_value=value
            best_action= a
    if best_action is None:
        return random.choice(valid_actions)
    return best_action

def train():
    runner_qtable= {}
    chaser_qtable= {}
    #epsilon= EPSILON
    runner_epsilon = 1.0
    chaser_epsilon = 1.0
    
    for episode in range(NUM_EPISODES):
        gs= FixedGameState()
        state= get_state(gs)
        recent_positions= []
        
        while not gs.game_over and gs.step_count< MAX_STEPS:
            gs.step_count+=1
            old_runner_pos= gs.runner_pos
            old_chaser_pos= gs.chaser_pos
    
            runner_action= get_action(state,runner_qtable,gs,gs.runner_pos,True,runner_epsilon)
            chaser_action= get_action(state,chaser_qtable,gs,gs.chaser_pos,True,chaser_epsilon)
            
            dr,dc= ACTIONS[runner_action]
            new_runner_pos=(gs.runner_pos[0]+dr, gs.runner_pos[1]+dc)
            gs.move_runner(new_runner_pos)

            dr,dc = ACTIONS[chaser_action]
            new_chaser_pos = (gs.chaser_pos[0] +dr, gs.chaser_pos[1]+dc)
            gs.move_chaser(new_chaser_pos)
        
            r_reward = runner_reward(state, gs, old_runner_pos)
            if new_runner_pos in recent_positions:
                r_reward+=R_REVISIT
            recent_positions.append(new_runner_pos)
            if len(recent_positions)>10:
                recent_positions.pop(0)
            c_reward= chaser_reward(gs, old_chaser_pos)
        
            new_state = get_state(gs)
            best_r = float("-inf")
            for a in range(len(ACTIONS)):
                value= runner_qtable.get((new_state, a),0)
                if value>best_r:
                    best_r= value
            current_q= runner_qtable.get((state, runner_action),0)
            runner_qtable[(state,runner_action)]= current_q+ALPHA*(r_reward+GAMMA*best_r-current_q)
            
            best_c = float("-inf")
            for a in range(len(ACTIONS)):
                value = chaser_qtable.get((new_state, a), 0)
                if value > best_c:
                    best_c = value
            current_q = chaser_qtable.get((state, chaser_action), 0)
            chaser_qtable[(state,chaser_action)]=current_q+ALPHA*(c_reward+GAMMA*best_c-current_q)
            state= new_state
        #epsilon= max(0.05,epsilon*EPSILON_DECAY)
        runner_epsilon= max(0.05,runner_epsilon*RUNNER_EPSILON_DECAY)
        chaser_epsilon= max(0.05,chaser_epsilon*CHASER_EPSILON_DECAY)
        if episode%1000== 0:
            print(f"Episode {episode}/{NUM_EPISODES}")
    
    return runner_qtable, chaser_qtable


def save_marl(runner_qtable,chaser_qtable):
    with open("q_tables/runner_marl.pkl", "wb") as f:
        pickle.dump(runner_qtable, f)
    with open("q_tables/chaser_marl.pkl", "wb") as f:
        pickle.dump(chaser_qtable, f)
    print("MARL Q-tables saved")

def load_marl(runner_file="q_tables/runner_marl.pkl", chaser_file="q_tables/chaser_marl.pkl"):
    with open("q_tables/runner_marl.pkl", "rb") as f:
        runner_qtable = pickle.load(f)
    with open("q_tables/chaser_marl.pkl", "rb") as f:
        chaser_qtable = pickle.load(f)
    print("MARL Q-tables loaded")
    return runner_qtable, chaser_qtable


if __name__ == "__main__":
    print("Training MARL...")
    runner_qtable, chaser_qtable = train()
    print("Runner Q-table size:",len(runner_qtable))
    print("Chaser Q-table size:",len(chaser_qtable))
    save_marl(runner_qtable, chaser_qtable)