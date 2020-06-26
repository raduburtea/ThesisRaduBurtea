# ThesisRaduBurtea
In this repository you can find all the codes used in the experiments performed in my thesis.

In the Q-Learning folder you can find the implementation of the Q-learning algorithm which uses images as inputs.

In the DQN folder are included all the implementation of the Deep Q-Learning algorithm. On the DQNCNN.py script have been run all the experiments in which images were considered as inputs and on the DQN_NOIMAGE.py all the experiments which used handcrafted features.

In the DDQN folder all the implementations of the Double Deep Q Learning algorithm can be found. The DDQNCNN.py script was used for the experiments where one image was considered as input, DDQN4STACKED.py was used for the experiments with 4 stacked images as input and DDQN_NOIMAGE.py was used when considering handcrafted features as inputs. Also videos of some succesful training sessions have been included.

The DDPG folder contains the implementations of the Deep Deterministic Policy Gradients algorithm. The folder DDPGOriginalVersion contains the original code inspired by Yan Panlau. The script DDPGLunarNEW.py contains the implementation of the new DDPG algorithm on the LunarLanderContinuous-v2 Environment and the script DDPGRacingNEW.py contains the implementation for the CarRacing-v0 environment.

In the Evolutionary folder the script for the evolutionary algorithm can be found and the Results folder contains all the graphs used in the paper and the script (graphs.py) that was used to generate them.
