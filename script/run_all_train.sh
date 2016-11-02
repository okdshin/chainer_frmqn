#!/bin/bash
#
# bash run_all_train.sh <GPU ID>
#
# ex
#  bash ./run_all_train.sh 0
#
# ex (for CPU)
#  .sh ./run_all_train.sh -1
#
python i_maze_rogue.py -y=1 -gpu=$1 -horizontal=1 -vertical=5 -testoutput=1 -modeltype=DQN -outdir=i_maze_rogue_dqn -updatefreq=4 -lr=0.00025 &
python i_maze_rogue.py -y=1 -gpu=$1 -horizontal=1 -vertical=5 -testoutput=1 -modeltype=DQN -outdir=i_maze_rogue_dqn -updatefreq=1  -lr=0.00025 &
python i_maze_rogue.py -y=1 -gpu=$1 -horizontal=1 -vertical=5 -testoutput=1 -modeltype=DRQN -outdir=i_maze_rogue_drqn &
python i_maze_rogue.py -y=1 -gpu=$1 -horizontal=1 -vertical=5 -testoutput=1 -modeltype=MQN -outdir=i_maze_rogue_mqn &
python i_maze_rogue.py -y=1 -gpu=$1 -horizontal=1 -vertical=5 -testoutput=1 -modeltype=RMQN -outdir=i_maze_rogue_rmqn &
python i_maze_rogue.py -y=1 -gpu=$1 -horizontal=1 -vertical=5 -testoutput=1 -modeltype=FRMQN -outdir=i_maze_rogue_frmqn &
