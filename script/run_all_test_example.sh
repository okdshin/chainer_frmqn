#!/bin/bash
python i_maze_rogue.py -y=1 -gpu=$1 -modeltype=DQN -modelpath=./0926i_maze_rogue_dqn/2016-09-26-21:00:43.119614-i_maze_rogue_episode25000 -horizontal=1 -validation=1 -outdir=dqn_validation -framehistnum 12
python i_maze_rogue.py -y=1 -gpu=$1 -modeltype=DRQN -modelpath=./0921i_maze_rogue_drqn_gpu1/2016-09-21-18:57:22.266530-i_maze_rogue_episode59900 -horizontal=1 -validation=1 -outdir=drqn_validation &
python i_maze_rogue.py -y=1 -gpu=$1 -modeltype=MQN -modelpath=./0921i_maze_rogue_mqn_gpu1/2016-09-21-18:57:22.242571-i_maze_rogue_episode59900 -horizontal=1 -validation=1 -outdir=mqn_validation50 -framehistnum 50 &
python i_maze_rogue.py -y=1 -gpu=$1 -modeltype=RMQN -modelpath=./0921i_maze_rogue_rmqn_gpu1/2016-09-21-18:57:22.261642-i_maze_rogue_episode59900 -horizontal=1 -validation=1 -outdir=rmqn_validation50 -framehistnum 50 &
python i_maze_rogue.py -y=1 -gpu=$1 -modeltype=FRMQN -modelpath=./0921i_maze_rogue_frmqn_gpu1/2016-09-21-18:57:22.208064-i_maze_rogue_episode59900 -horizontal=1 -validation=1 -outdir=frmqn_validation50 -framehistnum 50 &
