import pandas as pd 
import os 
import matplotlib.pyplot as plt 

dqn1 = pd.read_csv('DQN/Rew_newstff_run2_DQN.txt')
dqn_nocnn = pd.read_csv('DQN/Rew_newstff_nocnn.txt')
dqn_nocnnrep = pd.read_csv('DQN/Rew_newstff_nocnn_noreplay3.txt')
# dqn2 = pd.read_csv('DQN/Rew_newstff_nocnn.txt')
# run2 = pd.read_csv('DDQN/Rew_newstff_run2.txt')
# run1 = pd.read_csv('DDQN/Rew_newstff_run1.txt')
# run3 = pd.read_csv('DDQN/Rew_newstff_run3.txt')
# run4 = pd.read_csv('DDQN/Rew_newstff_run4.txt')
# run_nocnn = pd.read_csv('DDQN/Rew_newstff_run_nocnn.txt')
run_evol = pd.read_csv('Evol/Rew_newstff_evol.txt')
run_ddpg = pd.read_csv('ddpg/Rewphil.txt')
run_car = pd.read_csv('ddpg/rewards.txt')
run_dqn = pd.read_csv('DQN/Rew_newstff_nocnn.txt')
# runs = [run1, run2, run3, run4]

# run2_running_ave = run2.values
print(dqn_nocnn.values[100:].mean())
# print('2 ',dqn2.values.mean())
def averages(run):
	run2_running_ave = run.values
	running = []
	for n in range(len(run2_running_ave)):
	    running.append(run2_running_ave[max(0,n-20):(n)].mean())
	return running

# avgs = [averages(run) for run in runs]
plt.ylabel('Reward')
plt.xlabel('Episode count')
plt.title('DQN Handcrafted with Replay Buffer Averages')
# plt.plot(averages(dqn2))
plt.plot(averages(dqn_nocnn))
# plt.legend(['Run 1'])
# plt.plot(avgs[1][:200], color = 'orange')
# plt.legend(['Run 2'])
# plt.plot(avgs[2][:200], color = 'green')
# plt.legend(['Run 3'])
# plt.plot(avgs[3][:200], color = 'red')
# plt.legend(['Run 1', 'Run 2', 'Run 3', 'Run 4'])
plt.show()
plt.close()

 # if n <= 10:
 #        eps -= 0.05
 # elif eps <= 0.05:
 #        eps = 0.05
 # else:
 #        eps = 1/np.sqrt(n)
