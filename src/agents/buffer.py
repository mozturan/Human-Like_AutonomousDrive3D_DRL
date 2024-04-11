from typing import Tuple
import numpy as np
import random as rndm
from abc import ABC, abstractmethod

class AbstractReplayBuffer(ABC):
    """
    Abstract class for replay buffer
    """
    @abstractmethod
    def store_transition(self, state, action, reward, state_, done):
        """
        Store a new transition
        """
        pass

    @abstractmethod
    def sample_buffer(self, batch_size, priority_scale=1.0):
        """
        Sample a batch of transitions from the buffer
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Return the number of stored transitions
        """
        pass

class PER(AbstractReplayBuffer):
    """
    * init the values
    * for DQN actions are discrete
    """
    
    def __init__(self, max_size: int, input_shape: Tuple[int, int, int], n_actions: int, discrete: bool = True):
        super().__init__()
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.index = 0
        
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)

        dtype = np.int8 if self.discrete else np.float16
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.priorities = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):

        index = self.mem_cntr % self.mem_size     

        self.state_memory[index] = state
        self.new_state_memory[index] = state_

        #* store one hot encoding of actions, if appropriate
        if self.discrete:
            #* Create an zeros-array size of the number of actions
            actions = np.zeros(self.action_memory.shape[1])
            #* Make 1 the value of performed action
            actions[action] = 1.0
            #* Store in action memory
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action

        #* store reward and if it's terminal info 
        self.reward_memory[index] = reward
        #* we send inverse done info!!!
        self.terminal_memory[index] = 1 - done
        self.priorities[index] = max((self.priorities.max()), 1.0)
        self.mem_cntr +=1
        self.index = self.mem_cntr

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities
        
    def get_importance(self, probabilities):
        importance = 1/(self.mem_cntr) * 1/probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample_buffer(self, batch_size, priority_scale=1.0):
        
        if self.mem_cntr >= self.mem_size:
            self.index = self.mem_size
            
        sample_size = batch_size
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = rndm.choices(range(self.index), k=sample_size, weights=sample_probs[:self.index])

        states = self.state_memory[sample_indices]
        actions = self.action_memory[sample_indices]
        rewards = self.reward_memory[sample_indices]
        states_ = self.new_state_memory[sample_indices]
        terminal = self.terminal_memory[sample_indices]

        # samples = np.array(self.buffer)[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        return states, actions, rewards, states_, terminal, sample_indices

    def set_priorities(self, indices, errors, offset=0.1):
        for i,e in zip(indices, errors):
            error = abs(e) + offset
            clipped_error = np.minimum(error, 1.0)
            self.priorities[i] = clipped_error

    def __len__(self):
        return self.mem_cntr
    
class ReplayBuffer:

    """
        Initializes the replay buffer with the given maximum size, input shape, number of actions, and type of action space.

        Parameters:
            max_size (int): The maximum size of the replay buffer.
            input_shape (tuple): The shape of the input state.
            n_actions (int): The number of possible actions.
            discrete (bool, optional): Flag indicating whether the action space is discrete or continuous. Defaults to False.
    """

    def __init__(self, max_size, input_shape, n_actions, discrete = False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete

        dtype = np.int8 if self.discrete else np.float32
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):

        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_

        if self.discrete:

            #* Create an zeros-array size of the number of actions
            actions = np.zeros(self.action_memory.shape[1])
            #* Make 1 the value of performed action
            actions[action] = 1.0
            #* Store in action memory
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action



        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
    
    def __len__(self):
        return self.mem_cntr
    
