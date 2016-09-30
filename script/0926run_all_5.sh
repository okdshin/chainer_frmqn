python i_maze_rogue.py -y=1 -gpu=2 -horizontal=1 -vertical=5 -testoutput=1 -modeltype=DQN -outdir=0926i_maze_rogue_dqn -updatefreq=4 -lr=0.00025 &
python i_maze_rogue.py -y=1  -gpu=2 -horizontal=1 -vertical=5 -testoutput=1 -modeltype=DQN -outdir=0926i_maze_rogue_dqn -updatefreq=1  -lr=0.00025 &
python i_maze_rogue.py -y=1 -gpu=3 -horizontal=1 -vertical=5 -testoutput=1 -modeltype=DRQN -outdir=0926i_maze_rogue_drqn &
python i_maze_rogue.py -y=1 -gpu=3 -horizontal=1 -vertical=5 -testoutput=1 -modeltype=MQN -outdir=0926i_maze_rogue_mqn &
python i_maze_rogue.py -y=1 -gpu=3 -horizontal=1 -vertical=5 -testoutput=1 -modeltype=RMQN -outdir=0926i_maze_rogue_rmqn &
python i_maze_rogue.py -y=1 -gpu=2 -horizontal=1 -vertical=5 -testoutput=1 -modeltype=FRMQN -outdir=0926i_maze_rogue_frmqn &
