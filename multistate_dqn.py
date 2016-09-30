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
    def __init__(self, gpu, state_shape, action_num, image_num_per_state,
            model,
            gamma=0.99, # discount factor
            replay_batch_size=32,
            replay_memory_size=5*10**4,
            target_model_update_freq=1,
            max_step=50,
            lr=0.00025,
            clipping=False # if True, ignore reward intensity
            ):
        self.gpu = gpu
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
        s = state.astype(np.float32)
        q = self.model.q_function(s)
        selected_q = F.select_item(q, actions.astype(np.int32))

        r = self.xp.sign(rewards) if self.clipping else rewards
        s_dash = state_dash.astype(np.float32)
        q_dash = self.model_target.q_function(s_dash) # Q(s',*)
        max_q_dash = q_dash.data.max(axis=1) # max_a Q(s',a)
        target = r + self.xp.logical_not(done_list).astype(np.float32)*self.gamma*max_q_dash

        target = target.reshape((1,)+target.shape)
        selected_q = F.reshape(selected_q, (1,)+selected_q.shape)
        loss = F.sum(F.huber_loss(target, selected_q, delta=1.0)) #/ self.replay_batch_size
        return loss, q

    def calc_loss_recurrent(self, state, state_dash, actions, rewards, done_list):
        frames = F.swapaxes(state, 0, 1) # (Batch, FrameNum, FrameData) -> (FrameNum, Batch, FrameData)
        for f in range(0, self.image_num_per_state-1):
            self.model.q_function(frames[f])
        q = self.model.q_function(frames[-1])
        selected_q = F.select_item(q, actions.astype(np.int32))

        r = self.xp.sign(rewards) if self.clipping else rewards
        frames_dash = F.swapaxes(state_dash, 0, 1) # same for frames
        self.model_target.reset_state()
        for f in range(0, self.image_num_per_state-1):
            self.model_target.q_function(frames_dash[f])
        q_dash = self.model_target.q_function(frames_dash[-1])  # Q(s',*): shape is (batch_size, action_num)
        max_q_dash = q_dash.data.max(axis=1) # max_a Q(s',a): shape is (batch_size,)
        target = r + self.xp.logical_not(done_list).astype(np.float32)*self.gamma*max_q_dash

        target = target.reshape((1,)+target.shape)
        selected_q = F.reshape(selected_q, (1,)+selected_q.shape)
        loss = F.sum(F.huber_loss(target, selected_q, delta=1.0)) #/ self.replay_batch_size
        return loss, q

    def experience_replay(self):
        state, state_dash, actions, rewards, done_list = \
                self.dataset.extract_batch(self.replay_batch_size, self.image_num_per_state)
        state = state.astype(np.float32)/255.0
        state_dash = state_dash.astype(np.float32)/255.0
        if self.gpu >= 0:
            state = cuda.to_gpu(state)
            state_dash = cuda.to_gpu(state_dash)
            actions = cuda.to_gpu(actions)
            rewards = cuda.to_gpu(rewards)
            done_list = cuda.to_gpu(done_list)

        if self.model.is_reccurent():
            self.model.push_state() # Save current episode state
            self.optimizer.zero_grads()
            loss, q = self.calc_loss_recurrent(state, state_dash, actions, rewards, done_list)
            loss.backward()
            self.optimizer.update()
            self.model.pop_state() # Load current episode state
        else: # normal DQN
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
            s = Variable(self.xp.array([state]).astype(np.float32))/255.0
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
