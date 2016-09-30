import copy
#logging.basicConfig(level=logging.INFO)
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import dataset

from functools import reduce

"""
def state_to_variable(state):
    return np.concatenate(state).astype(np.float32)
"""

class DeepQNet:
    def __init__(self, state_shape, action_num, image_num_per_state,
            model,
            gamma=0.99, # discount factor
            replay_batch_size=32,
            replay_memory_size=5*10**4,
            target_model_update_freq=1,
            max_step=50,
            lr=0.00025,
            clipping=False # if True, ignore reward intensity
            ):
        print("initializing DQN...")
        self.action_num = action_num
        self.image_num_per_state = image_num_per_state
        self.gamma = gamma
        self.replay_batch_size = replay_batch_size
        self.replay_memory_size = replay_memory_size
        self.target_model_update_freq = target_model_update_freq
        self.max_step = max_step
        self.clipping = clipping

        print("Initializing Model...")
        self.model = model
        self.model_target = copy.deepcopy(self.model)

        print("Initializing Optimizer")
        self.optimizer = optimizers.RMSpropGraves(lr=lr, alpha=0.95, momentum=0.0, eps=0.01)
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer.GradientClipping(20))

        print("Initializing Replay Buffer...")
        self.dataset = dataset.DataSet(
                max_size=replay_memory_size, max_step=max_step, frame_shape=state_shape, frame_dtype=np.uint8)

        self.xp = model.xp
        self.state_shape = state_shape

    def calc_loss(self, state, state_dash, actions, rewards, done_list):
        assert(state.shape == state_dash.shape)
        s = state.reshape((state.shape[0], reduce(lambda x, y: x*y, state.shape[1:]))).astype(np.float32)
        s_dash = state_dash.reshape((state.shape[0], reduce(lambda x, y: x*y, state.shape[1:]))).astype(np.float32)
        q = self.model.q_function(s)

        q_dash = self.model_target.q_function(s_dash)  # Q(s',*)
        max_q_dash = np.asarray(list(map(np.max, q_dash.data)), dtype=np.float32) # max_a Q(s',a)

        target = q.data.copy()
        for i in range(self.replay_batch_size):
            assert(self.replay_batch_size == len(done_list))
            r = np.sign(rewards[i]) if self.clipping else rewards[i]
            if done_list[i]:
                discounted_sum = r
            else:
                discounted_sum = r + self.gamma * max_q_dash[i]
            assert(self.replay_batch_size == len(actions))
            target[i, actions[i]] = discounted_sum

        loss = F.sum(F.huber_loss(Variable(target), q, delta=1.0)) #/ self.replay_batch_size
        return loss, q

    def calc_loss_recurrent(self, frames, actions, rewards, done_list, size_list):
        # TODO self.max_step -> max_step
        s = Variable(frames.astype(np.float32))

        self.model_target.reset_state() # Refresh model_target's state
        self.model_target.q_function(s[0]) # Update target model initial state

        target_q = self.xp.zeros((self.max_step, self.replay_batch_size), dtype=np.float32)
        selected_q_tuple = [None for _ in range(self.max_step)]

        for frame in range(0, self.max_step):
            q = self.model.q_function(s[frame])
            q_dash = self.model_target.q_function(s[frame+1])  # Q(s',*): shape is (batch_size, action_num)
            max_q_dash = q_dash.data.max(axis=1) # max_a Q(s',a): shape is (batch_size,)
            if self.clipping:
                rs = self.xp.sign(rewards[frame])
            else:
                rs = rewards[frame]
            target_q[frame] = rs + self.xp.logical_not(done_list[frame]).astype(np.int)*(self.gamma*max_q_dash)
            selected_q_tuple[frame] = F.select_item(q, actions[frame].astype(np.int))

        enable = self.xp.broadcast_to(self.xp.arange(self.max_step), (self.replay_batch_size, self.max_step))
        size_list = self.xp.expand_dims(cuda.to_gpu(size_list), -1)
        enable = (enable < size_list).T

        selected_q = F.concat(selected_q_tuple, axis=0)

        # element-wise huber loss
        huber_loss = F.huber_loss(
                F.expand_dims(F.flatten(target_q), axis=1),
                F.expand_dims(selected_q, axis=1), delta=1.0)
        huber_loss = F.reshape(huber_loss, enable.shape)

        zeros = self.xp.zeros(enable.shape, dtype=np.float32)
        loss = F.sum(F.where(enable, huber_loss, zeros)) #/ self.replay_batch_size
        #print("loss", loss.data)

        return loss

    def experience_replay(self):
        if self.model.is_reccurent():
            self.model.push_state() # Save current state
            replay_episodes = self.dataset.extract_episodes(self.replay_batch_size)
            frame_shape = replay_episodes[0].frame_shape
            frame_dtype = replay_episodes[0].frame_dtype
            frames = self.xp.zeros((self.replay_batch_size, self.max_step+1)+frame_shape, dtype=frame_dtype)
            actions = self.xp.zeros((self.replay_batch_size, self.max_step), dtype=np.uint8)
            rewards = self.xp.zeros((self.replay_batch_size, self.max_step), dtype=np.float32)
            done_list = self.xp.zeros((self.replay_batch_size, self.max_step), dtype=np.bool)
            for i in range(self.replay_batch_size):
                frames[i][:-1] = cuda.to_gpu(replay_episodes[i].frames)
                actions[i] = cuda.to_gpu(replay_episodes[i].actions)
                rewards[i] = cuda.to_gpu(replay_episodes[i].rewards)
                done_list[i] = cuda.to_gpu(replay_episodes[i].done_list)

            transpose_index = np.arange(len(frames.shape))
            transpose_index[0] = 1
            transpose_index[1] = 0
            frames = frames.transpose(*transpose_index)
            assert(frames.shape[0] == self.max_step+1 and frames.shape[1] == self.replay_batch_size)

            size_list = np.asarray([episode.size for episode in replay_episodes])

            self.optimizer.zero_grads()
            loss = self.calc_loss_recurrent(frames, actions.T, rewards.T, done_list.T, size_list)
            loss.backward()
            #print("loss grad is", loss.grad)
            self.optimizer.update()
            self.model.pop_state() # Load current state
        else: # normal DQN
            state, state_dash, actions, rewards, done_list = \
                    self.dataset.extract_batch(self.replay_batch_size, self.image_num_per_state)
            state = cuda.to_gpu(state)
            state_dash = cuda.to_gpu(state_dash)
            self.optimizer.zero_grads()
            loss, _ = self.calc_loss(state, state_dash, actions, rewards, done_list)
            loss.backward()
            self.optimizer.update()

    def random_action(self):
        return int(np.random.randint(0, self.action_num))

    def action_sample_eps_greedy(self, image, eps):
        if self.model.is_reccurent():
            #print(image.shape)
            s = Variable(self.xp.array([image]).astype(np.float32))
        else:
            state = np.zeros((self.image_num_per_state,)+self.state_shape, np.float32)
            state[:-1] = self.dataset.get_latest_frames(self.image_num_per_state-1)
            state[-1] = image
            state = np.concatenate(state, axis=0)
            s = Variable(self.xp.array([state]).astype(np.float32))
        q = self.model.q_function(s).data[0]

        if np.random.rand() < eps:
            action = self.random_action()
            #print("RANDOM : ", action)
        else:
            action = self.xp.argmax(q).astype(np.int8)
            #print("GREEDY : ", action, q)
        #print("action type", type(action), action.shape, action)
        return int(action), q

    def target_model_soft_update(self, tau):
        model_params = dict(self.model.namedparams())
        model_target_params = dict(self.model_target.namedparams())
        for name in model_target_params:
            model_target_params[name].data = tau*model_params[name].data\
                + (1 - tau)*model_target_params[name].data

    def target_model_hard_update(self):
        self.model_target = copy.deepcopy(self.model)

    def stock_experience(self, state, action, reward, done):
        self.dataset.stock_experience(state, action, reward, done)

    def finish_current_episode(self, test):
        assert(self.dataset.current_episode.done_list[self.dataset.current_episode.size-1] == True)
        if not test:
            self.dataset.stock_current_episode()
        self.dataset.clear_current_episode()
        if self.model.is_reccurent():
            print("reset internal state of model")
            self.model.reset_state()

    def learn(self, state, action, reward, done, soft_update=True, tau=0.001):
        self.experience_replay()
        if soft_update:
            self.target_model_soft_update(tau)
        else:
            self.target_model_hard_update()

def main():
    class PoleModel(Chain):
        def __init__(self, input_num, action_num):
            print(input_num, action_num)
            super(PoleModel, self).__init__(
                l1=L.Linear(input_num, 32),
                l2=L.Linear(32, 32),
                l3=L.Linear(32, action_num)
            )

        def q_function(self, state):
            h1 = F.leaky_relu(self.l1(state))
            h2 = F.leaky_relu(self.l2(h1))
            return self.l3(h2)

    dqn = DeepQNet(state_shape=(3, 32, 32), action_num=2, image_num_per_state=12,
            model=PoleModel(3*12*32*32, action_num=2))


if __name__ == "__main__":
    main()
