import random, copy
import numpy as np
from collections import deque
import episode
from arange import begin_arange, end_arange, pad_take

class DataSet:
    def __init__(self, max_size, max_step, frame_shape, frame_dtype, rng=np.random.RandomState()):
        self.max_size = max_size
        self.max_step = max_step
        self.frame_shape = frame_shape
        self.frame_dtype = frame_dtype
        self.rng = rng

        self.current_episode = episode.Episode(max_step, frame_shape, frame_dtype)
        self.episodes = deque()
        self.total_size = 0

    def stock_experience(self, frame, action, reward, done):
        self.current_episode.stock_experience(frame, action, reward, done)

    def stock_current_episode(self):
        assert(self.current_episode.done_list[self.current_episode.size-1] == True)
        self.episodes.append(copy.deepcopy(self.current_episode))
        self.total_size += self.current_episode.size
        while self.total_size > self.max_size:
            oldest_epi = self.episodes.popleft()
            self.total_size -= oldest_epi.size

    def clear_current_episode(self):
        self.current_episode = episode.Episode(self.max_step, self.frame_shape, self.frame_dtype)

    def get_latest_frames(self, frame_num):
        epi = self.current_episode
        return epi.get_frames(end_arange(end=epi.size, size=frame_num))

    def extract_episodes(self, episode_num):
        assert(0 < len(self.episodes))
        #print(episode_num, len(self.episodes))

        return self.rng.choice(self.episodes, episode_num, replace=(len(self.episodes) < episode_num))

    def extract_batch(self, batch_size, frame_num):
        assert(0 < len(self.episodes))
        state = np.zeros((batch_size, frame_num)+self.frame_shape, dtype=self.frame_dtype)
        state_dash = np.zeros((batch_size, frame_num)+self.frame_shape, dtype=self.frame_dtype)
        actions = np.zeros(batch_size, dtype=np.uint8)
        rewards = np.zeros(batch_size, dtype=np.float32)
        done_list = np.zeros(batch_size, dtype=np.bool)

        episode_indices = self.rng.randint(0, len(self.episodes), batch_size)
        for i in range(batch_size):
            epi = self.episodes[episode_indices[i]]
            frame_index = self.rng.randint(0, epi.size)

            if frame_index == epi.size-1:
                #state[i] = epi.get_frames(frame_index, frame_num)
                state[i] = epi.get_frames(end_arange(end=frame_index, size=frame_num))
                # state_dash[i] is not accessed
                actions[i] = epi.actions[frame_index]
                rewards[i] = epi.rewards[frame_index]
                assert(epi.done_list[frame_index] == True)
                done_list[i] = True
            else:
                blob = epi.get_frames(end_arange(end=frame_index+2, size=frame_num+1))
                state[i] = blob[:-1]
                state_dash[i] = blob[1:]
                actions[i] = epi.actions[frame_index]
                rewards[i] = epi.rewards[frame_index]
                done_list[i] = epi.done_list[frame_index]
        return state, state_dash, actions, rewards, done_list



def simple_tests():
    dataset = DataSet(max_size=20, max_step=10, frame_shape=(1,), frame_dtype=np.int8)
    for i in range(100):
        frame = np.random.randint(1, 10, size=(1,))
        action = np.random.randint(16)
        reward = np.random.random()
        done = False
        #if np.random.random() < 0.05:
        if i % 3 == 0: #and i != 0:
            done = True
        print("frame:", frame)

        state = np.zeros((2, 1), dtype=np.int8)
        state[:-1] = dataset.get_latest_frames(frame_num=1)
        state[-1] = frame
        print("STATE\n", state)
        assert(np.array_equal(state[-1], frame))

        dataset.stock_experience(frame, action, reward, done)
        print(dataset.episodes)
        print(dataset.current_episode.frames)
        print(dataset.current_episode.done_list)
        print(done)
        if done:
            dataset.stock_current_episode()
            dataset.clear_current_episode()
        #print("done_list", list(zip(dataset.images, dataset.done_list)))
        #print("size", dataset.size)
        print()
        """
        if dataset.size >= 9:# TODO
            batch = dataset.extract_batch(image_num_per_state=1, batch_size=1)
            print("BATCH", batch)
            states, actions, rewards = dataset.extract_episodes(2, 4)
            print("states\n", states)
        """
        input("--------")

def main():
    simple_tests()

if __name__ == "__main__":
    main()

