yes | python i_maze_rogue.py -gpu=$1 -horizontal=1 -vertical=5 -testoutput=1 -modeltype=DQN -batchsize=8 -outdir=0921i_maze_rogue_dqn_gpu$1 -lr=0.00025 > 0921i_maze_rogue_dqn_gpu$1/test_log.txt &
yes | python i_maze_rogue.py -gpu=$1 -horizontal=1 -vertical=5 -testoutput=1 -modeltype=DRQN -batchsize=8 -outdir=0921i_maze_rogue_drqn_gpu$1 > 0921i_maze_rogue_drqn_gpu$1/test_log.txt &
yes | python i_maze_rogue.py -gpu=$1 -horizontal=1 -vertical=5 -testoutput=1 -modeltype=MQN -batchsize=8 -outdir=0921i_maze_rogue_mqn_gpu$1 > 0921i_maze_rogue_mqn_gpu$1/test_log.txt &
yes | python i_maze_rogue.py -gpu=$1 -horizontal=1 -vertical=5 -testoutput=1 -modeltype=RMQN -batchsize=8 -outdir=0921i_maze_rogue_rmqn_gpu$1 > 0921i_maze_rogue_rmqn_gpu$1/test_log.txt &
yes | python i_maze_rogue.py -gpu=$1 -horizontal=1 -vertical=5 -testoutput=1 -modeltype=FRMQN -batchsize=8 -outdir=0921i_maze_rogue_frmqn_gpu$1 > 0921i_maze_rogue_frmqn_gpu$1/test_log.txt &
