import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from maze_env import FixedGameState
from agents.minimax import chaser_move
from agents.q_agent import load_qtable,get_action,get_state,ACTIONS

NUM_EPISODES= 1000
MAX_STEPS= 500

q_table= load_qtable("q_tables/q_table.pkl")

def run_episode():
    gs= FixedGameState()
    state= get_state(gs)
    chaser_hist= []
    visited={} 
    while not gs.game_over and gs.step_count<MAX_STEPS:
        gs.step_count+= 1
        state_key= gs.runner_pos
        visited[state_key] = visited.get(state_key,0)+1
        if visited[state_key]>=5:
            epsilon=1.0
        elif visited[state_key]>=3:
            epsilon=0.5
        else:
            epsilon=0.05
        action=get_action(state,q_table,gs,training=False, epsilon= epsilon)
        dr,dc = ACTIONS[action]
        gs.move_runner((gs.runner_pos[0]+dr, gs.runner_pos[1]+dc))
        cmove= chaser_move(gs, chaser_hist)
        if cmove:
            chaser_hist.append(gs.chaser_pos)
            if len(chaser_hist)>4:
                chaser_hist.pop(0)
            gs.move_chaser(cmove)
        state= get_state(gs)
    return gs.winner, gs.step_count, sum(gs.collected)

winners= []
steps= []
checkpoints= []

for ep in range(NUM_EPISODES):
    winner,step_count,cp_collected= run_episode()
    winners.append(1 if winner=="runner" else 0)
    steps.append(step_count)
    checkpoints.append(cp_collected)

win_rate=sum(winners)/NUM_EPISODES*100
avg_steps=sum(steps)/NUM_EPISODES
print(f"Q-learning Runner vs Minimax Chaser")
print(f"Win rate: {sum(winners)}/{NUM_EPISODES} ({win_rate:.1f}%)")
print(f"Average steps: {avg_steps:.1f}")
print(f"Timeouts: {steps.count(MAX_STEPS)}/{NUM_EPISODES}")
print(f"Avg checkpoints collected: {sum(checkpoints)/NUM_EPISODES:.2f}")

window= 50
rolling_wins= [sum(winners[max(0,i-window):i+1])/min(i+1,window)*100 for i in range(NUM_EPISODES)]

fig,axes= plt.subplots(2,2,figsize=(12,8))
fig.suptitle("Q-learning Runner vs Minimax Chaser", fontsize=14)

axes[0,0].plot(rolling_wins, color='blue')
axes[0,0].set_title("Rolling Win Rate")
axes[0,0].set_xlabel("Episode")
axes[0,0].set_ylabel("Win Rate")
axes[0,0].axhline(y=win_rate, color='red',label=f'Overall: {win_rate:.1f}%')
axes[0,0].legend()

axes[0,1].plot(steps, alpha=0.3, color='green')
axes[0,1].axhline(y=avg_steps, color='red', label=f'Avg: {avg_steps:.1f}')
axes[0,1].set_title("Steps per Episode")
axes[0,1].set_xlabel("Episode")
axes[0,1].set_ylabel("Steps")
axes[0,1].legend()

cp_counts = [checkpoints.count(i) for i in range(4)]
axes[1,0].bar(['0','1','2','3'], cp_counts, color=['red','orange','yellow','green'])
axes[1,0].set_title("Checkpoints Collected Distribution")
axes[1,0].set_xlabel("Checkpoints Collected")
axes[1,0].set_ylabel("Episodes")

axes[1,1].pie([sum(winners),NUM_EPISODES-sum(winners)],
              labels=['Runner wins','Chaser wins'],
              colors=['blue','red'],
              autopct='%1.1f%%')
axes[1,1].set_title("Win/Loss Distribution")

plt.tight_layout()
plt.savefig("tests/results_q.png", dpi=150)
plt.show()