import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from maze_env import GameState
from agents.astar import eastar, heuristic
from agents.minimax import chaser_move

NUM_EPISODES= 1000

def runner_move(state):
    target= None
    best_dist= float('inf')
    for i,cp in enumerate(state.checkpoints):
        if not state.collected[i]:
            d= heuristic(state.runner_pos,cp)
            chaser_penalty= max(0,10-heuristic(state.chaser_pos,cp))
            if d+chaser_penalty<best_dist:
                best_dist=d+chaser_penalty
                target= cp
    if target is None:
        target=state.exit_pos
    return eastar(state.maze,state.runner_pos,target,state.chaser_pos)

def run_episode():
    gs = GameState(rows=21, cols=21)
    chaser_hist = []
    MAX_STEPS = 500
    while not gs.game_over and gs.step_count < MAX_STEPS:
        gs.step_count += 1
        rmove = runner_move(gs)
        if rmove:
            gs.move_runner(rmove)
        cmove = chaser_move(gs, chaser_hist)
        if cmove:
            chaser_hist.append(gs.chaser_pos)
            if len(chaser_hist) > 4:
                chaser_hist.pop(0)
            gs.move_chaser(cmove)
    return gs.winner, gs.step_count, sum(gs.collected)

winners= []
steps= []
checkpoints= []

for ep in range(NUM_EPISODES):
    winner,step_count,cp_collected= run_episode()
    winners.append(1 if winner=="runner" else 0)
    steps.append(step_count)
    checkpoints.append(cp_collected)

win_rate= sum(winners)/NUM_EPISODES*100
avg_steps = sum(steps)/NUM_EPISODES
print(" A* Runner vs Minimax Chaser")
print(f"Win rate: {sum(winners)}/{NUM_EPISODES} ({win_rate:.1f}%)")
print(f"Average steps: {avg_steps:.1f}")
print(f"Timeouts: {steps.count(500)}/{NUM_EPISODES}")
print(f"Avg checkpoints collected: {sum(checkpoints)/NUM_EPISODES:.2f}")

window= 50
rolling_wins=[sum(winners[max(0,i-window):i+1])/min(i+1,window)*100 for i in range(NUM_EPISODES)]

fig,axes= plt.subplots(2,2,figsize=(12,8))
fig.suptitle("A* Runner vs Minimax Chaser",fontsize=14)

axes[0,0].plot(rolling_wins, color='blue')
axes[0,0].set_title("Rolling Win Rate")
axes[0,0].set_xlabel("Episode")
axes[0,0].set_ylabel("Win Rate")
axes[0,0].axhline(y=win_rate,color='red',label=f'Overall: {win_rate:.1f}%')
axes[0,0].legend()
axes[0,1].plot(steps,alpha=0.3,color='green')
axes[0,1].axhline(y=avg_steps,color='red',label=f'Avg: {avg_steps:.1f}')
axes[0,1].set_title("Steps per Episode")
axes[0,1].set_xlabel("Episode")
axes[0,1].set_ylabel("Steps")
axes[0,1].legend()

cp_counts = [checkpoints.count(i) for i in range(4)]
axes[1,0].bar(['0','1','2','3'], cp_counts, color=['red','orange','yellow','green'])
axes[1,0].set_title("Checkpoints Collected Distribution")
axes[1,0].set_xlabel("Checkpoints Collected")
axes[1,0].set_ylabel("Episodes")

axes[1,1].pie([sum(winners), NUM_EPISODES-sum(winners)], labels=['Runner wins','Chaser wins'],
              colors=['blue','red'],autopct='%1.1f%%')
axes[1,1].set_title("Win/Loss Distribution")

plt.tight_layout()
plt.savefig("tests/results_minimax.png", dpi=150)
plt.show()