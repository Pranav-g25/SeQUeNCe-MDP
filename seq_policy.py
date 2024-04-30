import json
import numpy as np

def policy_reader(file_name):
    jsonFile =  open(file_name, encoding="utf-8-sig") 
    policy_data = json.load(jsonFile)
    return policy_data

class policy():
    def __init__(self, policy_data) -> None:
        self.states = []
        self.action_spaces = []
        self.policies = []
        self.N = len(policy_data['state_info'])
        for i in range(self.N):
            self.states.append(np.array(policy_data['state_info'][i]['state']))
            self.action_spaces.append(policy_data['state_info'][i]['action_space'])
            self.policies.append(policy_data['state_info'][i]['policy'])



policy_data_ = policy_reader('policy_n3_test.json')

policy_1 = policy(policy_data_)



