import os
import random
import time
import enum
import datetime
import argparse

import scipy.misc as sm
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import RogueGym as gym
from multistate_dqn import DeepQNet
from model import DQN, DRQN, MQN, RMQN, FRMQN

def make_i_maze_map_data(vertical, horizontal):
    map_str = []

    upper_map_str = [
        "".join(["#" for _ in range(5+horizontal*2)]),
        "#2"+"".join(["." for _ in range(1+horizontal*2)])+"3#",
    ]
    map_str.extend(upper_map_str)

    middle_map_str = [
        "".join(["#" for _ in range(2+horizontal)])+"."+"".join(["#" for _ in range(2+horizontal)])
    ]
    vert = random.choice(vertical)
    for i in range(vert):
        map_str.extend(middle_map_str)

    lower_map_str = [
        "#."+"".join(["." for _ in range(horizontal)])+"0"+"".join(["." for _ in range(horizontal)])+"1#",
        "".join(["#" for _ in range(5+horizontal*2)]),
    ]
    map_str.extend(lower_map_str)

    return map_str

def make_i_maze_map(vertical, horizontal, block_set, args):
    map_data = make_i_maze_map_data(horizontal=horizontal, vertical=vertical)
    print(map_data)
    map_data, positions = gym.load_map(block_set, map_data)
    map_ = gym.Map(*map_data)
    for pos in positions:
        map_.set_block(pos, block_set["."])
    map_.set_block(positions[2], block_set["B"])
    map_.set_block(positions[3], block_set["R"])
    indicator = block_set[np.random.choice(("G", "Y"))] if not args.validation else block_set["Y"]
    map_.set_block(positions[1], indicator)
    return map_, indicator, positions[0], positions[2], positions[3]

class I_MazeEnv(gym.RogueEnv):
    def __init__(self, horizontal, vertical, max_step):
        self.horizontal = horizontal
        self.vertical = vertical
        self.max_step = max_step

        self.total_reward = 0.0

        self.block_set = {}
        self.block_set["."] = gym.Block(block_id=0, indicator=".", name="air", movable=True)
        self.block_set["#"] = gym.Block(block_id=1, indicator="#", name="stone",
                color=(127, 127, 127), movable=False)
        self.block_set["R"] = gym.Block(block_id=2, indicator="R", name="red_tile",
                block_type="tile", color=(255, 0, 0), movable=True)
        self.block_set["B"] = gym.Block(block_id=3, indicator="B", name="blue_tile",
                block_type="tile", color=(0, 0, 255), movable=True)
        self.block_set["Y"] = gym.Block(block_id=4, indicator="Y", name="yellow_tile",
                block_type="tile", color=(255, 255, 0), movable=True)
        self.block_set["G"] = gym.Block(block_id=5, indicator="G", name="green_tile",
                block_type="tile", color=(0, 255, 0), movable=True)
        self.action_set = [("move", 1), ("move", -1), ("turn", 1), ("turn", -1)]
        self.rogue_env = gym.RogueEnv(self.action_set)

    def print_state(self, args, train_total_step, episode_id, step, eps):
        if not args.testoutput:
            os.system("clear")
        print(vars(args))
        print("train_total_step", "{0:5d}".format(train_total_step))
        print("episode_id", "{0:5d}".format(episode_id))
        print("step", "{0:2d}".format(step))
        print("total_reward", "{0:3.2f}".format(self.total_reward))
        print("eps", "{0:3.6f}".format(eps))
        self.rogue_env.print_map()

    def _get_ob(self):
        """
        ob = gym.map_to_ob(self.rogue_env.map_,
                pos=self.rogue_env.agent_position,
                direction=self.rogue_env.agent_direction)
        hv = gym.ob_to_hot_vectors(ob, block_type_num=len(self.block_set))
        return hv.reshape((hv.shape[0]*hv.shape[1],))
        """
        env = self.rogue_env
        if env.agent_direction[0] == -1 and env.agent_direction[1] == 0:
            direction = 180
        elif env.agent_direction[0] == 0 and env.agent_direction[1] == -1:
            direction = 90
        elif env.agent_direction[0] == 1 and env.agent_direction[1] == 0:
            direction = 0
        elif env.agent_direction[0] == 0 and env.agent_direction[1] == 1:
            direction = 270
        frame = gym.map_to_image((32, 32), self.rogue_env.map_,
                self.rogue_env.agent_position, direction)
        return np.moveaxis(frame, -1, 0), frame

    def reset(self, args):
        self.total_reward = 0.0
        map_, self.indicator, start_pos, self.blue_pos, self.red_pos \
            = make_i_maze_map(horizontal=self.horizontal, vertical=self.vertical, block_set=self.block_set, args=args)
        self.rogue_env.reset(map_, start_direction=random.choice(gym.directions), start_position=start_pos)
        ob, _ = self._get_ob()
        return ob

    def step(self, action, step):
        self.rogue_env.step(action)
        if self.rogue_env.map_.get_block(self.rogue_env.agent_position).name == "red_tile":
            done = True
            reward = 1.0 if self.indicator.name == "yellow_tile" else -1.0
        elif self.rogue_env.map_.get_block(self.rogue_env.agent_position).name == "blue_tile":
            done = True
            reward = 1.0 if self.indicator.name == "green_tile" else -1.0
        elif step == self.max_step-1:
            done = True
            reward = -0.04
        else:
            done = False
            reward = -0.04

        ob_dash, frame = self._get_ob()
        #sm.imsave("frame.png", frame)

        self.total_reward += reward
        return ob_dash, reward, done

class Agent:
    def __init__(self, gpu, ob_shape, action_num, frame_history_num, model,
            lr, eps_delta, eps, batch_size):
        self.dqn = DeepQNet(gpu, ob_shape, action_num, frame_history_num,
                replay_batch_size=batch_size, replay_memory_size=k_replay_memory_size,
                model=model, gamma=k_gamma, max_step=k_max_step, lr=lr)
        self.frame_history_num = frame_history_num
        self.eps_delta = eps_delta
        self.eps = eps

    def act(self, ob, is_test):
        if is_test:
            action, q = self.dqn.action_sample_eps_greedy(ob, eps=0.0)
        else:
            action, q = self.dqn.action_sample_eps_greedy(ob, self.eps)

            if self.eps > 0.1:
                self.eps -= self.eps_delta
            if self.eps < 0.1:
                self.eps = 0.1

        return action, q

    def update_state(self, ob, action, reward, done, test):
        self.dqn.stock_experience(ob, action, reward, done)
        if done:
            self.dqn.finish_current_episode(test)

    def learn(self, ob, action, reward, done):
        self.dqn.learn(ob, action, reward, done, soft_update=True, tau=0.001)

    def save(self, name):
        serializers.save_npz(name+".model", self.dqn.model)
        serializers.save_npz(name+".optimizer", self.dqn.optimizer)

    def load(self, name):
        serializers.load_npz(name+".model", self.dqn.model)
        serializers.load_npz(name+".optimizer", self.dqn.optimizer)

class run_mode(enum.IntEnum):
    test = 0
    train = 1
    validation = 2

def run_episode(current_id, args, env, agent, mode, episode_id, train_total_step):
    ob = env.reset(args)
    step = 0
    while True:
        #current_train_total_step = train_total_step+step
        is_initexp = train_total_step < args.initexp
        if not args.testoutput:
            env.print_state(args, train_total_step, episode_id, step, eps=agent.eps)
        elif mode == run_mode.test and episode_id%1000 < 10:
            env.print_state(args, train_total_step, episode_id, step, eps=agent.eps)

        if mode == run_mode.train and is_initexp:
            action = random.randrange(len(env.action_set))
            q = 0
        else:
            action, q = agent.act(ob, is_test=(mode!=run_mode.train))

        ob_dash, reward, done = env.step(action, step)
        if mode == run_mode.train:
            if is_initexp:
                print("initexp", train_total_step, "<", args.initexp)
            elif train_total_step % args.updatefreq == 0:
                agent.learn(ob, action, reward, done)
            train_total_step += 1
        agent.update_state(ob, action, reward, done, test=(not mode==run_mode.train))

        ob = ob_dash

        if args.validation:
            time.sleep(0.1)
        step += 1
        if done:
            break

    with open(os.path.join(args.outdir, current_id+"."+mode.name), "a") as f:
        f.write(str(episode_id)+" "+str(reward)+" "+str(env.total_reward)+" "+str(train_total_step)+"\n")

    return train_total_step

k_ob_shape = (3,32,32) # 3channel 32x32 image
k_frame_history_num = 12
k_max_step = 50
#k_max_episode = int(150/50)*(10**4)*2
k_max_train_total_step = 140*(10**4)*10
k_gamma = 0.99
k_replay_memory_size = 5*10**4

k_default_modeltype = str("MQN")
k_default_lr = 0.0005
k_default_replay_batch_size = 32
k_default_update_freq = 4
def main():
    current_id = datetime.datetime.today().isoformat("-")+"-"+os.path.splitext(os.path.basename(__file__))[0]
    parser = argparse.ArgumentParser(description='I-Maze with Block obs')
    parser.add_argument("-modelpath", type=str,
            help="modelpath without extension(eg .model, .optimizer)")
    parser.add_argument("-vertical", type=int, nargs="+", default=[2], help="vertical corridor length")
    parser.add_argument("-horizontal", type=int, default=0, help="horizontal corridor length")
    parser.add_argument("-validation", type=int, default=0, help="validation flag, default:0")
    parser.add_argument("-outdir", type=str, default="log",
            help="output dir for loggin, default:'log'")
    parser.add_argument("-epsdelta", type=float, default=10**-6,
            help="delta of epsilon, default:10**-6")
    parser.add_argument("-initexp", type=int, default=10**4,
            help="initial exproration, default:10**4")
    parser.add_argument("-eps", type=float, default=1.0, help="epsilon, default:1.0")
    parser.add_argument("-lr", type=float, default=k_default_lr,
            help="epsilon, default:"+str(k_default_lr))
    parser.add_argument("-modeltype", type=str, default=k_default_modeltype,
            help="ModelType, default:'"+k_default_modeltype+"'")
    parser.add_argument("-batchsize", type=int, default=k_default_replay_batch_size,
            help="replay batch size, default:"+str(k_default_replay_batch_size))
    parser.add_argument("-updatefreq", type=int, default=k_default_update_freq,
            help="update frequency, default:"+str(k_default_update_freq))
    parser.add_argument("-gpu", type=int, default=0,
            help="gpu id, default:0 (cpu is -1)")
    parser.add_argument("-testoutput", type=int, default=0,
            help="output only at test, default:0")
    parser.add_argument("-y", type=int, default=0,
            help="OK?, default:0")
    args = parser.parse_args()

    print(args)
    if args.y == 0:
        input("OK?")

    ## Make directory and write setting log
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    with open(os.path.join(args.outdir, current_id+".args"), "w") as argsf:
        argsf.write(str(args))

    env = I_MazeEnv(horizontal=args.horizontal, vertical=args.vertical, max_step=k_max_step)

    ## Init model
    input_dim = k_ob_shape[0]
    output_dim = len(env.action_set)
    if args.modeltype == "DQN":
        model = DQN(input_dim*k_frame_history_num, output_dim)
    elif args.modeltype == "DRQN":
        model = DRQN(input_dim, output_dim)
    elif args.modeltype == "MQN":
        model = MQN(input_dim, output_dim, max_buff_size=11, m=256, e=4096)
    elif args.modeltype == "RMQN":
        model = RMQN(input_dim, output_dim, max_buff_size=11, m=256, e=4096)
    elif args.modeltype == "FRMQN":
        model = FRMQN(input_dim, output_dim, max_buff_size=11, m=256, e=4096)
    else:
        print("not implemented", args.modeltype)
        exit(0)

    ## Use GPU
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    ## Init agent
    agent = Agent(args.gpu, k_ob_shape, len(env.action_set), k_frame_history_num, model,
            lr=args.lr, eps_delta=args.epsdelta, eps=args.eps, batch_size=args.batchsize)

    train_total_step = 0
    episode_id = 0
    #for episode_id in range(k_max_episode):
    while train_total_step < k_max_train_total_step:
        try:
            if args.validation:
                ## Run validation
                mode = run_mode.validation
                run_episode(current_id, args, env, agent, mode, episode_id, train_total_step)
            else:
                if episode_id%100 == 0 and episode_id !=0:
                    ## Run test
                    mode = run_mode.test
                    for j in range(10):
                        run_episode(current_id, args, env, agent, mode, episode_id+j, train_total_step)

                    ## Save model
                    agent.save(os.path.join(args.outdir, current_id+"_episode"+str(episode_id)))

                ## Run train
                mode = run_mode.train
                train_total_step \
                    = run_episode(current_id, args, env, agent, mode, episode_id, train_total_step)
        except:
            ark = {}
            ark["args"] = vars(args)
            ark["episode_id"] = episode_id
            ark["train_total_step"] = train_total_step
            ark["eps"] = current_eps
            with open(os.path.join(args.outdir, current_id+"_episode"+str(episode_id)+"_ark.json"), "w") as arkf:
                ark_str = json.dumps(ark, indent=4, sort_keys=True)
                arkf.write(ark_str)
            with open(os.path.join(args.outdir, current_id+"_episode"+str(episode_id)+"_dataset.pkl"), "wb") as datasetf:
                pickle.dump(agent.dqn.dataset, datasetf)
            exit(0)
        episode_id += 1


if __name__ == "__main__":
    main()
