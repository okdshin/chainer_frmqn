import random
import numpy as np
from arange import begin_arange, end_arange, pad_take

"""
def padded_sub_array(array, begin_end):
    begin, end = begin_end
    assert(begin < end)
    size = end - begin
    array_len = array.shape[0]
    sub_array = np.zeros((size,)+array.shape[1:], dtype=array.dtype)
    if 0 <= begin and end <= array_len:
        sub_array = array[begin:end]
    elif begin < 0 and end <= array_len:
        assert(len(sub_array[-begin:]) == len(array[:end]))
        sub_array[-begin:] = array[:end]
    elif 0 <= begin and array_len < end:
        assert(len(sub_array[:self.size]) == len(array[begin:]))
        sub_array[:array_len] = array[begin:]
    elif begin < 0 and array_len < end:
        assert(len(sub_array[-begin:(-begin+array_len)]) == array_len)
        sub_array[-begin:(-begin+array_len)] = array
    return sub_array
"""

class Episode:
    def __init__(self, max_step, frame_shape, frame_dtype):
        self.max_step = max_step
        self.frame_shape = frame_shape
        self.frame_dtype = frame_dtype
        self.frames = np.zeros((max_step,)+frame_shape, dtype=frame_dtype)
        self.actions = np.zeros(max_step, dtype=np.uint8)
        self.rewards = np.zeros(max_step, dtype=np.float32)
        self.done_list = np.zeros(max_step, dtype=np.bool)
        self.size = 0

    def stock_experience(self, frame, action, reward, done):
        assert(self.size <= self.max_step)
        self.frames[self.size] = frame
        self.actions[self.size] = action
        self.rewards[self.size] = reward
        self.done_list[self.size] = done
        self.size += 1

    def get_frames(self, indices):
        return pad_take(self.frames, indices)

    def get_actions(self, indices):
        return pad_take(self.actions, indices)

    def get_rewards(self, indices):
        return pad_take(self.rewards, indices)

    def get_done_list(self, indices):
        return pad_take(self.done_list, indices)

def test():
    epi = Episode(max_step=10, frame_shape=(1,), frame_dtype=np.int8)
    for i in range(10):
        epi.stock_experience(frame=i, action=i, reward=i, done=i==9)
        state = epi.get_frames(end_arange(end=epi.size, size=3))
        print(state, epi.actions[i], epi.rewards[i], epi.done_list[i])
        print("---")

    print("===")
    for i in range(10):
        state = epi.get_frames(end_arange(end=i+1, size=3))
        actions = epi.get_actions(end_arange(end=i+1, size=3))
        print(state, epi.actions[i], epi.rewards[i], epi.done_list[i])
        print(actions)
        print("---")

def main():
    test()

if __name__ == "__main__":
    main()
