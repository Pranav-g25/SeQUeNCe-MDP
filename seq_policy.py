import json
import numpy as np
import sys, os, time
from math import inf

from sequence.topology.node import Node
from sequence.components.memory import MemoryWithRandomCoherenceTime
from sequence.entanglement_management.generation import EntanglementGenerationA, GenProtoStatus
from sequence.kernel.timeline import Timeline
from sequence.kernel.process import Process
from sequence.kernel.event import Event
from sequence.topology.node import BSMNode
from sequence.components.optical_channel import QuantumChannel, ClassicalChannel
from sequence.entanglement_management.entanglement_protocol import EntanglementProtocol
from sequence.entanglement_management.swapping import EntanglementSwappingA, EntanglementSwappingB
from sequence.resource_management.resource_manager import ResourceManagerMessage, ResourceManagerMsgType
from sequence.message import Message


class policy():
    '''Retreives policy data from  a json file for implementation.'''
    def __init__(self, policy_data) -> None:
        self.states = []
        self.action_spaces = []
        self.policies = []
        self.N = len(policy_data['state_info'])
        for i in range(self.N):
            self.states.append(np.array(policy_data['state_info'][i]['state']))
            self.action_spaces.append(policy_data['state_info'][i]['action_space'])
            self.policies.append(policy_data['state_info'][i]['policy'])
   
    def policy_reader(file_name):
        jsonFile =  open(file_name, encoding="utf-8-sig") 
        policy_data = json.load(jsonFile)
        return policy_data

class NodesTracker():
    ''' Creates, manages all nodes, bsmnodes, links etc.'''
    def __init__( self, _tl, _numRouters, _numGenerations, _light_speed, _endnode_distance ):

class SimpleManager():
    '''Resource manager for 'EntangleGenNode' objects '''

class EntangleGenNode(Node):
    '''overwriting functions like receive_message etc. to incorporate features of c_node control operations and policy.'''
        






