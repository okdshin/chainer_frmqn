#!/bin/bash
#
# bash ./script/run_one_test.sh <GPU ID> <MODEL TYPE> <TRAINED MODEL PATH> <VERTICAL CORRIDOR LENGTH> <OUT DIR PATH> <FRAME HISTORY NUM>
#
# ex
#  bash ./script/run_one_test.sh 0 DQN "trained/dqn" 5 "log" 12
#
# ex
#  bash ./script/run_one_test.sh 0 FRMQN "trained/frmqn" 5 "log" 50
#
python3 i_maze_rogue.py -gpu=$1 -modeltype=$2 -modelpath=$3 -vertical=$4  -horizontal=1 -validation=1 -outdir=$5 -framehistnum $6
