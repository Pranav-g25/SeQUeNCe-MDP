import pickle
import numpy as np
from typing import List, Tuple, Any

class MultiMemPolicy:
    """
    Wraps a pickled list of {state, action_space, policy} dicts for
    multimemory entanglement control. 
    
    - state:      an (N×N) array of ints (number of links per node‑pair)
    - action_space: List of actions; each action is List[Tuple[node_index, num_swaps]]
    - policy:     one‑hot vector selecting which entry in action_space to take
    """
    def __init__(self, policy_data: List[dict[str, Any]]):
        # load and normalize
        self.states        = [np.array(d['state'],      dtype=float) for d in policy_data]
        self.action_spaces = [d['action_space']        for d in policy_data]
        self.policies      = [np.array(d['policy'],    dtype=float) for d in policy_data]
        self.N             = self.states[0].shape[0] if self.states else 0
        self.num_states    = len(self.states)

    @classmethod
    def load(cls, filename: str) -> "MultiMemPolicy":
        """Load pickled policy file and return a MultiMemPolicy instance."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return cls(data)

    def get_action(self, state: np.ndarray) -> List[Tuple[int,int]]:
        """
        Given the current state matrix, return the selected action
        (a list of (node_index, num_swaps) tuples).
        """
        # find matching state index
        for idx, s in enumerate(self.states):
            if np.array_equal(s, state):
                # pick the action with highest probability (one‑hot)
                action_index = int(self.policies[idx].argmax())
                return self.action_spaces[idx][action_index]
        raise KeyError("State not found in policy table", state)

    def __repr__(self):
        return (f"<MultiMemPolicy N={self.N}, "
                f"num_states={self.num_states}>")
