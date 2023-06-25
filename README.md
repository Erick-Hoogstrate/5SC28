# Overview

This folder contains all the relevant code and results that were used and generated during the system identification and swing-up policy learning for the course 5SC28 by project group 21.
The code and results are divided as follows:
- __00_disc-benchmark-files contains the train and test data used for the generation of the submission files, and the two submission files for the best performing models. Other submission files can be found in the folders of those models.
- __01_system_id contains all the code and models used during the system identification, namely NARX ANN and GP, and Neural Net state space.
- __02_swingup_policy contains all the code and agents used during the reinforcement learning, namely DQN, SAC, and PILCO. It also contains a folder with two utility reward-plotting notebooks used for reward function design.