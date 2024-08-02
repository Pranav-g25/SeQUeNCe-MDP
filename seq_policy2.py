import json
import numpy as np
import sys, os, time
from math import inf

from enum import Enum, auto
from sequence.components.memory import Memory
from sequence.components.circuit import Circuit
from typing import TYPE_CHECKING, List
from sequence.utils import log

from sequence.topology.node import Node
from sequence.components.memory import MemoryWithRandomCoherenceTime
from sequence.entanglement_management.generation import EntanglementGenerationA
from sequence.kernel.timeline import Timeline
from sequence.kernel.process import Process
from sequence.kernel.event import Event
from sequence.topology.node import BSMNode
from sequence.components.optical_channel import QuantumChannel, ClassicalChannel
from sequence.entanglement_management.entanglement_protocol import EntanglementProtocol
from sequence.entanglement_management.swapping import EntanglementSwappingMessage, SwappingMsgType, EntanglementSwappingA, EntanglementSwappingB
from sequence.resource_management.resource_manager import ResourceManagerMessage, ResourceManagerMsgType
from sequence.message import Message

doSimulations = True
doDubugOutput = False

class policy():
    '''Retreives policy data from  a json file for implementation.'''
    def __init__(self, policy_data) -> None:
        self.states = []
        self.action_spaces = []
        self.policies = []
        self.N = len(np.array(policy_data['state_info'][0]['state'])[0,:])
        self.NumOfStates = len(policy_data['state_info'])
        for i in range(self.NumOfStates):
            self.states.append(np.array(policy_data['state_info'][i]['state']))
            self.action_spaces.append(policy_data['state_info'][i]['action_space'])
            self.policies.append(policy_data['state_info'][i]['policy'])
   
    def policy_reader(file_name):
        jsonFile =  open(file_name, encoding="utf-8-sig") 
        policy_data = json.load(jsonFile)
        return policy_data

class ControlNodeMsgType(Enum):
    #LINK_WAS_ESTABLISHED = auto()
    LINK_STATUS_UPDATED = auto()
    ESTABLISHED_LINK_WAS_BROKEN = auto()
    
class ControlNodeMessage(Message):
    def __init__(self, msg_type: ControlNodeMsgType, _origin: str, **kwargs ):
        Message.__init__(self, msg_type, "EntangleGenNode")
        self.origin = _origin
        if msg_type is ControlNodeMsgType.LINK_STATUS_UPDATED:
            self.established = kwargs[ 'established' ] if 'established' in kwargs else True
            self.left = kwargs['left'] if 'left' in kwargs else None
            self.right = kwargs['right'] if 'right' in kwargs else None
            self.middle = kwargs['middle'] if 'middle' in kwargs else None
            pass
        else:
            raise Exception("ControlNodeMessage gets unknown type of message: %s" % str(msg_type))

class ControlNodeData():
    def __init__(self):
        self.genLinksUpdated = None
        self.genLinksEstablished = None
        self.numLinksToBeUpdatedAtThisSession = None
        self.policy : policy = None
        self.current_state : np.array = None
        self.current_action = None
        self.t_cut = None
        self.numSwapsToBeUpdatedAtThisSession = None
        self.numSwapsUpdated = 0 

    def addPolicy(self,_policy):
        self.policy = _policy
        self.current_state = np.full((_policy.N, _policy.N), -1)
        self.terminal_states = [] 
        for i in range(self.t_cut):
             a = np.full((_policy.N, _policy.N), - 1)
             a[0,_policy.N - 1] = a[_policy.N - 1, 0] = i
             self.terminal_states.append(a)

    
    def findActionForCurrentStateAccToPolicy(self, _state : np.array) -> list[int]:
        state_index = 0
    
        for i in range(len(self.policy.states)):
            if ((self.policy.states[i] == _state).all()):
                state_index = i
                break
        _action_space = self.policy.action_spaces[state_index]
        _policy = self.policy.policies[state_index]
        print('action space :        ', _action_space)
        return _action_space[_policy.index(1.0)]

    def reset(self):
        self.numSwapsToBeUpdatedAtThisSession = None
        self.numSwapsUpdated = 0
        self.current_state = np.full((self.policy.N, self.policy.N), -1)
        
class Link():
    def __init__(self, _nodesNos =None):
        if not _nodesNos is None:
            assert _nodesNos[0] < _nodesNos[1]
            self.node_left = _nodesNos[0]
            self.node_right = _nodesNos[1]
        else:
            self.node_left = None
            self.node_right = None
            
        self.node_left_memo_updated = False
        self.node_right_memo_updated = False
        self.genProtocolFinished = False
        self.established = False
        self.memoryExpired = False
    
    def copy_from( self, _other ):
        self.node_left = _other.node_left
        self.node_right = _other.node_right
        self.node_left_memo_updated = _other.node_left_memo_updated
        self.node_right_memo_updated = _other.node_right_memo_updated
        self.genProtocolFinished = _other.genProtocolFinished
        self.established = _other.established
        self.memoryExpired = _other.memoryExpired
            
        return self

class SimpleManager():
    def __init__(self, _owner, _nodesTracker):
        self.owner = _owner
        self.nodeNo = int( _owner.name[4:] )
        self.leftMemoName = '%s.left_memo' % _owner.name
        self.rightMemoName = '%s.right_memo' % _owner.name
        self.nodesTracker = _nodesTracker
        #self.eg_updates = [ False, False ]
        #self.wasEntangled = [ False, False ]

    def update( self, protocol, memory, state ):
        if len( protocol.name ) < 3:
            raise ValueError( 'I don\'t know this protocol name: %s' % protocol.name )
            
        if protocol.name[ -3: ] == '_eg':
            self.update_eg( protocol, memory, state )
        elif protocol.name[ -3: ] == 'ESA':
            self.update_es_new( protocol, memory, state )
        else:
            if len( protocol.name ) < 4:
                raise ValueError( 'I don\'t know this protocol name: %s' % protocol.name )
            if protocol.name[-4] == 'G':
                self.update_eg( protocol, memory, state )
            elif protocol.name[-4] == 'S':
                self.update_es_new( protocol, memory, state )
            else:
                raise ValueError( 'Wrong protocol name: %s' % protocol.name )

    def update_eg( self, protocol, memory, state ):
        _nodes  = self.nodesTracker.nodes
        _memo_id = None
        _established = None
        if memory.name == self.leftMemoName:
            _memo_id = 0
            #self.updated[0] = True
        elif memory.name == self.rightMemoName:
            _memo_id = 1
            #self.updated[1] = True
        else:
            raise ValueError( 'I don\'t have another sides of a node' )
        
        _remote_node_name = None
        _remote_node_no = None

        if state == 'RAW':
            _established = False
        
        elif state == 'ENTANGLED':
            _established = True
        
        _remote_node_name = self.owner.protocols[_memo_id].remote_node_name
        _remote_node_no = int(_remote_node_name[4:])

        srcNodeNo = None

        if (self.nodeNo >_remote_node_no):
            srcNodeNo = self.nodeNo
            msg = ControlNodeMessage( ControlNodeMsgType.LINK_STATUS_UPDATED, 'generation', established = _established )
            if srcNodeNo != self.nodesTracker.cnodeNo:
                _nodes[2*srcNodeNo].send_message( _nodes[2*self.nodesTracker.cnodeNo].name, msg )
            else:
                self.nodesTracker.cnode.receive_message( _nodes[2*srcNodeNo].name, msg )
            
    def update_es_new(self, protocol, memory, state):
        print('update_es_new called by :   ', self.owner.name, 'with state:', state, 'with protocol name:', protocol.name)
        _nodes  = self.nodesTracker.nodes
        _memo_id = None
        if memory.name == self.leftMemoName:
            _memo_id = 0
            #self.updated[0] = True
        elif memory.name == self.rightMemoName:
            _memo_id = 1
            #self.updated[1] = True
        
        if protocol.name[-3:] == 'ESA':
            if _memo_id == 0:
                midNodeName = self.owner.name
                leftNodeName = self.owner.protocols[2].left_node
                rightNodeName = self.owner.protocols[2].right_node
                midNodeNo = int(midNodeName[4:])
                leftNodeNo = int(leftNodeName[4:])
                rightNodeNo = int(rightNodeName[4:])
                _established = None
                if state == 'NOTSwapped':
                    _established = False
                    self.owner.protocols[2].left_memo.reset()
                    self.owner.protocols[2].right_memo.reset()
                    self.nodesTracker.nodes[2*leftNodeNo].right_memo.reset()
                    self.nodesTracker.nodes[2*rightNodeNo].left_memo.reset()
                if state == 'SWAPPED':
                    _established = True
                    self.owner.protocols[2].left_memo.reset()
                    self.owner.protocols[2].right_memo.reset()
                    self.nodesTracker.nodes[2*leftNodeNo].right_memo.entangled_memory["node_id"] = rightNodeName
                    self.nodesTracker.nodes[2*rightNodeNo].left_memo.entangled_memory["node_id"] = leftNodeName

                print("Cnode swapping msg sent to be:    ", _established)
                msg = ControlNodeMessage( ControlNodeMsgType.LINK_STATUS_UPDATED, 'swapping', established = _established , middle = midNodeNo, left =  leftNodeNo, right = rightNodeNo )        
                if midNodeNo != self.nodesTracker.cnodeNo:
                    _nodes[2*midNodeNo].send_message( _nodes[2*self.nodesTracker.cnodeNo].name, msg )
                else:
                    self.nodesTracker.cnode.receive_message( _nodes[2*midNodeNo].name, msg )
        else:
            pass

    def update_es( self, protocol, memory, state ):
        _memo_id = None
        if memory.name == self.leftMemoName:
            _memo_id = 0
            #self.updated[0] = True
        elif memory.name == self.rightMemoName:
            _memo_id = 1
            #self.updated[1] = True
        
        if state == 'RAW':
            memory.reset()
        #else:
        #    pass
        #    #memory.reset()
        
        #if self.nodesTracker.blockResourceManagerUpdating:
        #    return

        _nodes = self.nodesTracker.nodes
        #_memosToTrack = self.nodesTracker.memosToTrack
        #print( _memosToTrack, self.nodeNo, _memo_id )
        
        if doDubugOutput:
            print( '########### UPDATE swapping ###################' )
            print( 'memory state:', state, 'memory name:', memory.name )
            print( 'time:', self.nodesTracker.tl.now() )
            for j in range( 1, self.nodesTracker.numRouters + 2 ):
                print( _nodes[2*j].left_memo.entangled_memory,
                       _nodes[2*j-2].right_memo.entangled_memory )

        for _i in range( len( self.nodesTracker.links ) ):
            _l = self.nodesTracker.links[_i]
            _found = False
            if _memo_id == 1 and self.nodeNo == _l.node_left:
                if not _l.node_left_memo_updated:
                    _l.node_left_memo_updated = True
                _found = True
                _remote_nodeNo = _l.node_right
            elif _memo_id == 0 and self.nodeNo == _l.node_right:
                if not _l.node_right_memo_updated:
                    _l.node_right_memo_updated = True
                _found = True
                _remote_nodeNo = _l.node_left
            else:
                continue
            
            if _found:
                if _l.node_left_memo_updated and _l.node_right_memo_updated:
                    self.nodesTracker.links.pop( _i )
                    
                break

        if len( self.nodesTracker.links ) == 0: #and self.nodesTracker.tl.now() <= self.nodesTracker.lastButExpireEventTime:
            #self.nodesTracker.removeAllButExpirePendingEvents()
            self.nodesTracker.blockResourceManagerUpdating = True
            __wait_time = self.nodesTracker.signalingDelay + self.nodesTracker.swpAccDelayDelta + 1 #1.0*self.nodesTracker.endnode_distance/self.nodesTracker.light_speed
            self.nodesTracker.tl.schedule( Event( self.nodesTracker.tl.now() + __wait_time, self.nodesTracker.process ) )
            #print( 'dbg: swp protocol ended:',  self.nodesTracker.tl.now() * self.nodesTracker.light_speed / self.nodesTracker.endnode_distance )
            
    def release_remote_protocol(self, dst: str, protocol: "EntanglementProtocol") -> None:
        """Method to release protocols from memories on distant nodes.

        Release the remote protocol 'protocol' on the remote node 'dst'.
        The local protocol was paired with the remote protocol but local protocol becomes invalid.
        The resource manager needs to notify the remote node to cancel the paired protocol.

        Args:
            dst (str): name of the destination node.
            protocol (EntanglementProtocol): protocol to release on node.
        """

        msg = ResourceManagerMessage(ResourceManagerMsgType.RELEASE_PROTOCOL, protocol=protocol)
        #print( 'from:', self.owner.name, 'to:', self.nodesTracker.nodes[self.nodeNo].name )
        #[ print( ch ) for ch in self.owner.cchannels ]
    	#raise ValueError( 'Let\'s stop here' )
        self.owner.send_message(dst, msg)

class EntangleGenNode(Node):
    def __init__( self, _name: str, tl: Timeline, _nodesTracker, **memory_params ):
        super().__init__(_name, tl)
        #fidelity=0.9, frequency=2000, efficiency=1, coherence_time=-1, wavelength=500 
        #self.memory = MemoryWithRandomCoherenceTime('%s.memo'%name, tl, **memory_params ) #0.9, 2000, 1, -1, 500)
        #self.memory.owner = self
        self.left_memo = MemoryWithRandomCoherenceTime('%s.left_memo' % _name,
                                                       tl, **memory_params ) #WithRandomCoherenceTime
        self.left_memo.owner = self
        self.left_memo.add_receiver(self)
        self.left_memo.attach( self )
        self.right_memo = MemoryWithRandomCoherenceTime('%s.right_memo' % _name,
                                                        tl, **memory_params )
        self.right_memo.owner = self
        self.right_memo.add_receiver(self)
        self.right_memo.attach( self )
        self.resource_manager = SimpleManager( self, _nodesTracker )
        self.protocols = [ None ] * 4
        self.cnodeData = None
    
    def get(self, photon, **kwargs):
        self.send_qubit(kwargs['dst'], photon)

    def receive_message( self, src: str, msg: "Message") -> None:
        msg_type = str( msg.msg_type )
        #print( msg_type )
        
        if len( msg_type ) >= 22: #ResourceManagerMsgType.RELEASE_PROTOCOL
            if msg_type[:22] == 'ResourceManagerMsgType':
                
                if msg.msg_type is ResourceManagerMsgType.RELEASE_PROTOCOL:
                    assert isinstance(msg.protocol, EntanglementProtocol)
                    #print( msg.msg_type, msg.protocol.name )
                    msg.protocol.release()
                    return
                elif msg.msg_type is ResourceManagerMsgType.RELEASE_MEMORY:
                    #print( msg.msg_type, msg.memory.name )
                    target_id = msg.memory
                    for p in self.protocols:
                        if not p is None:
                            for memory in p.memories:
                                if memory.name == target_id:
                                    p.release()
                    return
                else:
                    raise ValueError( 'I\'m not expecting this kind of messages: %s' % msg_type )
        
        if len( msg_type ) >= 17:
            if msg_type[:17] == 'GenerationMsgType':
                if len(src) > len( 'node' ): 
                    if src[:4] == 'node':
                        if int( src[4:] ) > int( self.name[4:] ):
                            self.protocols[1].received_message(src, msg)
                        elif int( src[4:] ) < int( self.name[4:] ):
                            self.protocols[0].received_message(src, msg)
                        else:
                            raise ValueError( 'Don\'t know what message has been received' )
                        return
                if len(src) > len( 'bsm_node' ):
                    if src[:8] == 'bsm_node':
                        if int( src[8:] ) > int( self.name[4:] ):
                            self.protocols[1].received_message(src, msg)
                        elif int( src[8:] ) == int( self.name[4:] ):
                            self.protocols[0].received_message(src, msg)
                        else:
                            raise ValueError( 'Don\'t know what message has been received' )
                        return
        
        if len( msg_type ) >= 15:
            if msg_type[:15] == 'SwappingMsgType':
                if len(src) > len( 'node' ): 
                    pass
                    # print("############################################################### Msg received at ", self.name, "from", src)
                    # if src[:4] == 'node':
                    #     if int( src[4:] ) > int( self.name[4:] ):
                    #         self.protocols[3].received_message(src, msg)
                    #     elif int( src[4:] ) < int( self.name[4:] ):
                    #         self.protocols[2].received_message(src, msg)
                return
            
        if type( msg.msg_type ) is ControlNodeMsgType:
            if msg.msg_type is ControlNodeMsgType.LINK_STATUS_UPDATED:
                if msg.origin == 'generation':
                    _nodesTracker = self.resource_manager.nodesTracker
                    _nodes = _nodesTracker.nodes
                    _cnodeData = _nodesTracker.cnode.cnodeData
                    srcNodeNo = int(src[4:])
                    _cnodeData.genLinksUpdated += 1
                    if msg.established:
                        _cnodeData.genLinksEstablished += 1
                        _cnodeData.current_state[srcNodeNo, srcNodeNo - 1] \
                              = _cnodeData.current_state[srcNodeNo - 1, srcNodeNo] = 0
                    
                    else:
                        _nodesTracker.linksForNextSession.append( Link( ( srcNodeNo - 1, srcNodeNo ) ) )
                    
                    if _cnodeData.genLinksUpdated == _cnodeData.numLinksToBeUpdatedAtThisSession:
                        _nodesTracker.links.clear()
                        for i in range(len(_nodesTracker.linksForNextSession)):
                            _nodesTracker.links.append( _nodesTracker.linksForNextSession[i])
                        _nodesTracker.linksForNextSession.clear()
                        _cnodeData.current_action = _cnodeData.findActionForCurrentStateAccToPolicy(_cnodeData.current_state)
                        _cnodeData.numSwapsToBeUpdatedAtThisSession = len(_cnodeData.current_action)
                        print("state and action", _cnodeData.current_state, _cnodeData.current_action)
                        #######################################################################################################
                        #######################################################################################################
                        #######################################################################################################
                        #######################################################################################################
                        #######################################################################################################
                        #######################################################################################################
                        _process1 = Process( _nodesTracker, 'doAction',[ _cnodeData.current_action, _cnodeData.numSwapsUpdated] )
                        _nodesTracker.updateProcessToSchedule( _process1 )
                        _time_to_wait = ( _nodesTracker.signalingDelay + 1 #0.5*_nodesTracker.endnode_distance/_nodesTracker.light_speed
                                              if _nodesTracker.numRouters > 0
                                              else 1 )
                        _nodesTracker.tl.schedule( Event( _nodesTracker.tl.now() + _time_to_wait,
                                                              _process1 ) )
                        
                        
                        
                        

                        
                    '''            
                    else:
                        for j in range( 1, _nodesTracker.numRouters + 1 ):
                            _nodes[2*j-2].protocols[1].status = GenProtoStatus.INITIALIZED
                            _nodes[2*j].protocols[0].status = GenProtoStatus.INITIALIZED
                        #_nodesTracker.numGeneratedLinksReported = 0
                        _cnodeData.genLinksEstablished = 0
                        _cnodeData.genLinksUpdated = 0
                        _nodesTracker.resetAllMemories()
                        _nodesTracker.flushTotallyAllPendingEvents()
                        _nodesTracker.result = False
                            #print( 'dbg: cnode received gen failed:',  _nodesTracker.tl.now() * _nodesTracker.light_speed / _nodesTracker.endnode_distance )'''
                elif msg.origin == 'swapping':
                    _nodesTracker = self.resource_manager.nodesTracker
                    _nodes = _nodesTracker.nodes
                    _cnodeData = _nodesTracker.cnode.cnodeData

                    leftNodeNo = msg.left
                    midNodeNo = msg.middle
                    rightNodeNo = msg.right

                    if msg.established:
                        print("Swapping established on :           ",midNodeNo )
                        _cnodeData.current_state[leftNodeNo, rightNodeNo] = _cnodeData.current_state[rightNodeNo, leftNodeNo] = \
                            max(_cnodeData.current_state[leftNodeNo, midNodeNo], _cnodeData.current_state[midNodeNo, rightNodeNo])
                        
                    else:
                        print("Swapping failed on :           ",midNodeNo )
                        for i in range(1, abs(leftNodeNo - rightNodeNo) + 1):
                            _nodesTracker.links.append(Link((leftNodeNo + i - 1, leftNodeNo + i)))
                    
                    _cnodeData.current_state[leftNodeNo, midNodeNo] = _cnodeData.current_state[midNodeNo, leftNodeNo] = \
                        _cnodeData.current_state[midNodeNo, rightNodeNo] =_cnodeData.current_state[rightNodeNo, midNodeNo] = -1
                    

                    _time_to_wait = 1
                    if _cnodeData.numSwapsUpdated == _cnodeData.numSwapsToBeUpdatedAtThisSession:
                        _cnodeData.numSwapsUpdated = 0
                        _process2 = Process( _nodesTracker, 'postAction',[])
                        _nodesTracker.tl.schedule( Event( _nodesTracker.tl.now() + _time_to_wait,
                                                              _process2 ) )    
                    else:
                        _process = Process(_nodesTracker, 'doAction', [_cnodeData.current_action, _cnodeData.numSwapsUpdated])
                        _nodesTracker.tl.schedule( Event(_nodesTracker.tl.now() + _time_to_wait, _process))

            '''
            elif msg.msg_type is ControlNodeMsgType.ESTABLISHED_LINK_WAS_BROKEN:
            _nodesTracker = self.resource_manager.nodesTracker
            _nodesTracker.resetAllMemories()
            _nodesTracker.flushTotallyAllPendingEvents()
            _nodesTracker.resetNodesUpdatesFlags()
            _nodesTracker.result = False
            '''
        else:
            raise ValueError( 'Don\'t know what message has been received:', msg_type, src )
            
    def lastScheduledEventTimeEG( self ):
        _time_l = None
        if not self.protocols[0] is None:
            _time_l = self.tl.now()
            for event in self.protocols[0].scheduled_events:
                if not event.is_removed:
                    if event.time > _time_l:
                        _time_l = event.time
        _time_r = None
        if not self.protocols[1] is None:
            _time_r = self.tl.now()
            for event in self.protocols[1].scheduled_events:
                if not event.is_removed:
                    if event.time > _time_r:
                        _time_r = event.time
        if _time_l is None and _time_r is None:
            return None
        elif _time_l is None:
            return _time_r
        elif _time_r is None:
            return _time_l
        else:
            return max( _time_l, _time_r )
        
    def isTheLastProtocolEvent( self, _memo_id ):
        _en = None
        for e in self.timeline.events:
            if not e.is_invalid():
                _en = e
                break
        if _en is None:
            return True
        
        _res = True
        for e in self.protocols[_memo_id].scheduled_events:
            if e >= _en:
                _res = False
                break
        return _res

    def create_eg_protocol( self, middle: str, other: str, memo_id: int ):
        if memo_id == 0:
            self.protocols[0] = EntanglementGenerationA(self, '%s.EGA.r'%self.name, middle, other, self.left_memo )
            #self.protocols = [EntanglementGenerationA(self, '%s.eg'%self.name, middle, other, self.left_memo )]
        elif memo_id == 1:
            self.protocols[1] = EntanglementGenerationA(self, '%s.EGA.l'%self.name, middle, other, self.right_memo )
            #self.protocols = [EntanglementGenerationA(self, '%s.eg'%self.name, middle, other, self.right_memo )]
        else:
            raise ValueError( 'Bad memory index: %d' % memo_id )

    def create_es_protocol_middle( self, _swapping_params ):
        self.protocols[2] = EntanglementSwappingA_global(self, '%s.ESA'%self.name, self.left_memo, self.right_memo, **_swapping_params)

    def create_es_protocol_ends( self, memo_id: int ):
        if memo_id == 0:
            self.protocols[2] = EntanglementSwappingB_global(self, '%s.ESB.r'%self.name, self.left_memo)
        elif memo_id == 1:
            self.protocols[3] = EntanglementSwappingB_global(self, '%s.ESB.l'%self.name, self.right_memo)
        else:
            raise ValueError( 'Bad memory index: %d' % memo_id )
            
    def memory_expire( self, memo ):
        posDot = memo.name.find( '.' )
        posDash = memo.name.find( '_' )
        __suffix = memo.name[ posDot+1:posDash ]
        if __suffix == 'left':
            mem_id = 0
        elif __suffix == 'right':
            mem_id = 1
        else:
            print( __suffix )
            raise ValueError( 'I don\'t know this name: %s' % memo.name )
        for p in self.protocols:
            if not p is None:
                if p.is_ready():

                    if len( p.name ) < 3:
                        raise ValueError( 'I don\'t know this protocol name: %s' % p.name )
                        
                    if p.name[ -3: ] == '_eg':
                        #self.update_eg( memory, state )
                        pass
                    elif p.name[ -3: ] == 'ESA':
                        p.memory_expire( memo )
                    else:
                        if len( p.name ) < 4:
                            raise ValueError( 'I don\'t know this protocol name: %s' % p.name )
                        if p.name[-4] == 'G':
                            if p.name[-1] == 'r' and mem_id == 0:
                                p.memory_expire( memo )
                            elif p.name[-1] == 'l' and mem_id == 1:
                                p.memory_expire( memo )
                            else:
                                pass
                                
                        elif p.name[-4] == 'S':
                            p.memory_expire( memo )
                        else:
                            raise ValueError( 'Wrong protocol name: %s' % p.name )

def pair_eg_protocol( p1: EntanglementProtocol, p2: EntanglementProtocol ):
    p1.set_others( p2.name, p2.own.name ,p2.own.left_memo.name )
    p2.set_others( p1.name, p1.own.name, p1.own.left_memo.name )

def pair_es_protocol(p1, p2):
    p1.set_others( p2.name, p2.own.name ,p2.own.left_memo.name )
    p2.set_others( p1.name, p1.own.name, p1.own.left_memo.name )

class EntanglementSwappingA_global(EntanglementProtocol):
    """Entanglement swapping protocol for middle router.

    The entanglement swapping protocol is an asymmetric protocol.
    EntanglementSwappingA should be instantiated on the middle node, where it measures a memory from each pair to be swapped.
    Results of measurement and swapping are sent to the end routers.

    Variables:
        EntanglementSwappingA.circuit (Circuit): circuit that does swapping operations.

    Attributes:
        own (Node): node that protocol instance is attached to.
        name (str): label for protocol instance.
        left_memo (Memory): a memory from one pair to be swapped.
        right_memo (Memory): a memory from the other pair to be swapped.
        left_node (str): name of node that contains memory entangling with left_memo.
        left_remote_memo (str): name of memory that entangles with left_memo.
        right_node (str): name of node that contains memory entangling with right_memo.
        right_remote_memo (str): name of memory that entangles with right_memo.
        success_prob (float): probability of a successful swapping operation.
        degradation (float): degradation factor of memory fidelity after swapping.
        is_success (bool): flag to show the result of swapping
        left_protocol_name (str): name of left protocol.
        right_protocol_name (str): name of right protocol.
    """

    circuit = Circuit(2)
    circuit.cx(0, 1)
    circuit.h(0)
    circuit.measure(0)
    circuit.measure(1)

    def __init__(self, own: "Node", name: str, left_memo: "Memory", right_memo: "Memory", success_prob=1,
                 degradation=0.95):
        """Constructor for entanglement swapping A protocol.

        Args:
            own (Node): node that protocol instance is attached to.
            name (str): label for swapping protocol instance.
            left_memo (Memory): memory entangled with a memory on one distant node.
            right_memo (Memory): memory entangled with a memory on the other distant node.
            success_prob (float): probability of a successful swapping operation (default 1).
            degradation (float): degradation factor of memory fidelity after swapping (default 0.95).
        """

        assert left_memo != right_memo
        EntanglementProtocol.__init__(self, own, name)
        self.memories = [left_memo, right_memo]
        self.left_memo = left_memo
        self.right_memo = right_memo
        self.left_node = left_memo.entangled_memory['node_id']
        self.left_remote_memo = left_memo.entangled_memory['memo_id']
        self.right_node = right_memo.entangled_memory['node_id']
        self.right_remote_memo = right_memo.entangled_memory['memo_id']
        self.success_prob = success_prob
        self.degradation = degradation
        self.is_success = False
        self.left_protocol_name = None
        self.right_protocol_name = None

    def is_ready(self) -> bool:
        return self.left_protocol_name is not None \
               and self.right_protocol_name is not None

    def set_others(self, protocol: str, node: str, memories: List[str]) -> None:
        """Method to set other entanglement protocol instance.

        Args:
            protocol (str): other protocol name.
            node (str): other node name.
            memories (List[str]): the list of memories name used on other node.
        """

        if node == self.left_memo.entangled_memory["node_id"]:
            self.left_protocol_name = protocol
        elif node == self.right_memo.entangled_memory["node_id"]:
            self.right_protocol_name = protocol
        else:
            raise Exception("Cannot pair protocol %s with %s" % (self.name, protocol))

    def start(self) -> None:
        """Method to start entanglement swapping protocol.

        Will run circuit and send measurement results to other protocols.

        Side Effects:
            Will call `update_resource_manager` method.
            Will send messages to other protocols.
        """

        log.logger.info(f"{self.own.name} middle protocol start with ends "
                        f"{self.left_protocol_name}, "
                        f"{self.right_protocol_name}")

        assert self.left_memo.fidelity > 0 and self.right_memo.fidelity > 0
        assert self.left_memo.entangled_memory["node_id"] == self.left_node
        assert self.right_memo.entangled_memory["node_id"] == self.right_node

        if self.own.get_generator().random() < self.success_probability():
            fidelity = self.updated_fidelity(self.left_memo.fidelity, self.right_memo.fidelity)
            self.is_success = True

            expire_time = min(self.left_memo.get_expire_time(), self.right_memo.get_expire_time())

            meas_samp = self.own.get_generator().random()
            meas_res = self.own.timeline.quantum_manager.run_circuit(
                self.circuit, [self.left_memo.qstate_key,
                               self.right_memo.qstate_key], meas_samp)
            meas_res = [meas_res[self.left_memo.qstate_key], meas_res[self.right_memo.qstate_key]]

            msg_l = EntanglementSwappingMessage(SwappingMsgType.SWAP_RES,
                                                self.left_protocol_name,
                                                fidelity=fidelity,
                                                remote_node=self.right_memo.entangled_memory["node_id"],
                                                remote_memo=self.right_memo.entangled_memory["memo_id"],
                                                expire_time=expire_time,
                                                meas_res=[])
            msg_r = EntanglementSwappingMessage(SwappingMsgType.SWAP_RES,
                                                self.right_protocol_name,
                                                fidelity=fidelity,
                                                remote_node=self.left_memo.entangled_memory["node_id"],
                                                remote_memo=self.left_memo.entangled_memory["memo_id"],
                                                expire_time=expire_time,
                                                meas_res=meas_res)
            self.update_resource_manager(self.left_memo, "SWAPPED")
            self.update_resource_manager(self.right_memo, "SWAPPED")

        else:
            msg_l = EntanglementSwappingMessage(SwappingMsgType.SWAP_RES,
                                                self.left_protocol_name,
                                                fidelity=0)
            msg_r = EntanglementSwappingMessage(SwappingMsgType.SWAP_RES,
                                                self.right_protocol_name,
                                                fidelity=0)
            self.update_resource_manager(self.left_memo, "NOTSwapped")
            self.update_resource_manager(self.right_memo, "NOTSwapped")
            
        self.own.send_message(self.left_node, msg_l)
        print("###Swapping msg sent from", self.own.name, "to", self.left_node)
        self.own.send_message(self.right_node, msg_r)
        print("###Swapping msg sent from", self.own.name, "to", self.right_node)

    def success_probability(self) -> float:
        """A simple model for BSM success probability."""

        return self.success_prob

    #@lru_cache(maxsize=128)
    def updated_fidelity(self, f1: float, f2: float) -> float:
        """A simple model updating fidelity of entanglement.

        Args:
            f1 (float): fidelity 1.
            f2 (float): fidelity 2.

        Returns:
            float: fidelity of swapped entanglement.
        """

        return f1 * f2 * self.degradation

    def received_message(self, src: str, msg: "Message") -> None:
        """Method to receive messages (should not be used on A protocol)."""

        raise Exception("EntanglementSwappingA protocol '{}' should not receive messages.".format(self.name))

    def memory_expire(self, memory: "Memory") -> None:
        """Method to receive memory expiration events.

        Releases held memories on current node.
        Memories at the remote node are released as well.

        Args:
            memory (Memory): memory that expired.

        Side Effects:
            Will invoke `update` method of attached resource manager.
            Will invoke `release_remote_protocol` or `release_remote_memory` method of resource manager.
        """

        assert self.is_ready() is True
        if self.left_protocol_name:
            self.release_remote_protocol(self.left_node)
        else:
            self.release_remote_memory(self.left_node, self.left_remote_memo)
        if self.right_protocol_name:
            self.release_remote_protocol(self.right_node)
        else:
            self.release_remote_memory(self.right_node, self.right_remote_memo)

        for memo in self.memories:
            if memo == memory:
                self.update_resource_manager(memo, "RAW")
            #else:
                #self.update_resource_manager(memo, "ENTANGLED")

    def release_remote_protocol(self, remote_node: str):
        self.own.resource_manager.release_remote_protocol(remote_node, self)

    def release_remote_memory(self, remote_node: str, remote_memo: str):
        self.own.resource_manager.release_remote_memory(remote_node, remote_memo)

class EntanglementSwappingB_global(EntanglementProtocol):
    """Entanglement swapping protocol for middle router.

    The entanglement swapping protocol is an asymmetric protocol.
    EntanglementSwappingB should be instantiated on the end nodes, where it waits for swapping results from the middle node.

    Variables:
        EntanglementSwappingB.x_cir (Circuit): circuit that corrects state with an x gate.
        EntanglementSwappingB.z_cir (Circuit): circuit that corrects state with z gate.
        EntanglementSwappingB.x_z_cir (Circuit): circuit that corrects state with an x and z gate.

    Attributes:
        own (QuantumRouter): node that protocol instance is attached to.
        name (str): name of protocol instance.
        memory (Memory): memory to swap.
        remote_protocol_name (str): name of another protocol to communicate with for swapping.
        remote_node_name (str): name of node hosting the other protocol.
    """

    x_cir = Circuit(1)
    x_cir.x(0)

    z_cir = Circuit(1)
    z_cir.z(0)

    x_z_cir = Circuit(1)
    x_z_cir.x(0)
    x_z_cir.z(0)

    def __init__(self, own: "Node", name: str, hold_memo: "Memory"):
        """Constructor for entanglement swapping B protocol.

        Args:
            own (Node): node protocol instance is attached to.
            name (str): name of protocol instance.
            hold_memo (Memory): memory entangled with a memory on middle node.
        """

        EntanglementProtocol.__init__(self, own, name)

        self.memories = [hold_memo]
        self.memory = hold_memo
        self.remote_protocol_name = None
        self.remote_node_name = None

    def is_ready(self) -> bool:
        return self.remote_protocol_name is not None

    def set_others(self, protocol: str, node: str, memories: List[str]) -> None:
        """Method to set other entanglement protocol instance.

        Args:
            protocol (str): other protocol name.
            node (str): other node name.
            memories (List[str]): the list of memory names used on other node.
        """
        self.remote_node_name = node
        self.remote_protocol_name = protocol

    def received_message(self, src: str, msg: "EntanglementSwappingMessage") -> None:
        """Method to receive messages from EntanglementSwappingA.

        Args:
            src (str): name of node sending message.
            msg (EntanglementSwappingMesssage): message sent.

        Side Effects:
            Will invoke `update_resource_manager` method.
        """

        log.logger.debug(
            self.own.name + " protocol received_message from node {}, fidelity={}".format(src, msg.fidelity))
        print("######################   src=", src, "remoteNode=", self.remote_node_name, "at ",self.own.name,"#######################################")
        # assert src == self.remote_node_name

        # if msg.fidelity > 0 and self.own.timeline.now() < msg.expire_time:
        #     if msg.meas_res == [1, 0]:
        #         self.own.timeline.quantum_manager.run_circuit(self.z_cir, [self.memory.qstate_key])
        #     elif msg.meas_res == [0, 1]:
        #         self.own.timeline.quantum_manager.run_circuit(self.x_cir, [self.memory.qstate_key])
        #     elif msg.meas_res == [1, 1]:
        #         self.own.timeline.quantum_manager.run_circuit(self.x_z_cir, [self.memory.qstate_key])

        #     self.memory.fidelity = msg.fidelity
        #     self.memory.entangled_memory["node_id"] = msg.remote_node
        #     self.memory.entangled_memory["memo_id"] = msg.remote_memo
        #     self.memory.update_expire_time(msg.expire_time)
        #     self.update_resource_manager(self.memory, "ENTANGLED")
        # else:
        #     self.update_resource_manager(self.memory, "RAW")

    def start(self) -> None:
        log.logger.info(f"{self.own.name} end protocol start with partner {self.remote_node_name}")

    def memory_expire(self, memory: "Memory") -> None:
        """Method to deal with expired memories.

        Args:
            memory (Memory): memory that expired.

        Side Effects:
            Will update memory in attached resource manager.
        """

        self.update_resource_manager(self.memory, "RAW")

    def release(self) -> None:
        self.update_resource_manager(self.memory, "ENTANGLED")


class NodesTracker():
    def __init__( self, _tl, _numRouters, _numGenerations, _light_speed, _endnode_distance ):
        self.tl = _tl
        self.numRouters = _numRouters
        self.cnodeNo = ( self.numRouters + 2 ) // 2
        self.N = _numGenerations
        self.light_speed = _light_speed
        self.endnode_distance = _endnode_distance
        self.genCounter = 0
        self.nodes = []
        self.swapping_params = None
        #self.memosToTrack = []
        self.links = []
        self.linksForNextSession= []
        self.establishedLinks = []
        #self.numLinksToBeUpdatedAtThisSession = None
        self.numTrials = 0
        #self.lastButExpireEventTime = inf
        #self.numGeneratedLinksReported = 0
        self.process = None
        self.result = None
        #self.blockResourceManagerUpdating = True
        self.swpAccDelayDelta = 0
        self.signalingDelay = ( max( self.cnodeNo, _numRouters + 1 - self.cnodeNo )
                                * _endnode_distance / _light_speed / ( _numRouters + 1 ) ) 
        
    def reset( self ):
        #self.memosToTrack.clear()
        #self.entangledMemories.clear()
        self.links.clear()
        #self.numLinksToBeUpdatedAtThisSession = None
        #self.numGeneratedLinksReported = 0
        #self.lastButExpireEventTime = inf
        self.numTrials = 0
        self.process = None
        self.result = None
        self.linksForNextSession.clear()
        self.cnode.cnodeData.reset()
        

    #def resetNodesUpdatesFlags( self ):    
    #    for j in range( self.numRouters + 2 ):
    #       _eg_updates = self.nodes[2*j].resource_manager.eg_updates
    #       _eg_updates[0] = False
    #       _eg_updates[1] = False
        
    def setNodesList( self, _nodes ):
        self.nodes = _nodes
        self.cnode = self.nodes[ 2*self.cnodeNo ]
        self.cnode.cnodeData = ControlNodeData()

    def setSwappingParams( self, _swapping_params ):
        self.swapping_params = _swapping_params

    #def updateTrackingList( self, _memosToTrack ):
    #    self.memosToTrack = _memosToTrack
    #    self.memosToTrackCopy = list( _memosToTrack )

    def updateLinksToEstablish( self, _links ):
        self.links = _links
        self.establishedLinks.clear()
        #self.numLinksToBeUpdatedAtThisSession = len( _links )
        #self.cnode.cnodeData.numLinksToBeUpdatedAtThisSession = len( _links )

    def updateProcessToSchedule( self, _process ):
        self.process = _process

    def checkPathIsConsistent( s ):
        #print( '-----------------------------------------------' )
        #for j in range( 1, s.numRouters + 2 ):
        #    print( s.nodes[2*j-2].right_memo.entangled_memory, '\n',
        #           s.nodes[2*j].left_memo.entangled_memory )
        
        _numHoppings = 0
        _leftNodeNo = 0
        
        while True:
            _rightNodeName = s.nodes[2*_leftNodeNo].right_memo.entangled_memory['node_id']
            if _rightNodeName is None:
                break
            _rightNodeNo = int( _rightNodeName[4:] )
            
            _leftNodeNamePrev = s.nodes[2*_rightNodeNo].left_memo.entangled_memory['node_id']
            if _leftNodeNamePrev is None:
                break
            else:
                if int( _leftNodeNamePrev[4:] ) != _leftNodeNo:
                    break
                
            _numHoppings += 1
            _leftNodeNo = _rightNodeNo

        #print( _numHoppings if _leftNodeNo == s.numRouters + 1 else None )

        return _numHoppings if _leftNodeNo == s.numRouters + 1 else None

    #def resetEntangledMemories( self ):
    #    #for _m in self.entangledMemories:
    #    #    if _m[1] == 0:
    #    #        self.nodes[_m[0]].left_memo.reset()
    #    #    elif _m[1] == 1:
    #    #        self.nodes[_m[0]].right_memo.reset()
    #    #    else:
    #    #        raise ValueError( 'Wrong memo_id: %d' % _m[1] )
    #    #self.entangledMemories.clear()

    def resetAllMemories( self ):
        for j in range( 1, self.numRouters + 2 ):
            self.nodes[ 2*j-2 ].right_memo.reset()
            self.nodes[ 2*j ].left_memo.reset()
        for e in self.tl.events:
            if e.time >= self.tl.now() and not e.is_invalid():
                self.tl.remove_event( e )

    def resetUsedMemory( self ):
        for _m in self.memosToTrackCopy:
            if _m[1] == 0:
                self.nodes[2*_m[0]].left_memo.reset()
            elif _m[1] == 1:
                self.nodes[2*_m[0]].right_memo.reset()
            else:
                raise ValueError( 'Wrong memo_id: %d' % _m[1] )

    #def flushAllPendingEvents( s ):
    #    _numRemoved = 0
    #    for j in range( 1, s.numRouters ):
    #        for event in s.nodes[2*j-2].protocols[1].scheduled_events:
    #            if not event.is_invalid():
    #                s.tl.remove_event( event )
    #                _numRemoved += 1
    #        for event in s.nodes[2*j].protocols[0].scheduled_events:
    #            if not event.is_invalid():
    #                s.tl.remove_event( event )
    #                _numRemoved += 1
    #    #print( 'numRemoved:', _numRemoved )
        
    def flushTotallyAllPendingEvents( s ):
        _now = s.tl.now()
        for e in s.tl.events:
            if not e.is_invalid() and e.time >= _now:
                s.tl.remove_event( e )

    def doGenerationPart( s ):            
        '''if s.numTrials > 1e6:
            s.result = False
            s.resetAllMemories()
            s.flushTotallyAllPendingEvents()
            print( 'max iteration reached' )
            return

        if s.genCounter > s.N or s.tl.now() > 1160e12:
            s.resetAllMemories()
            s.result = False
            s.flushTotallyAllPendingEvents()
            return
            
        s.numTrials += 1
        '''
        if doDubugOutput:
            print( '################ generation ###################' )
            print( 'time:', s.tl.now() )
            for j in range( 1, s.numRouters + 2 ):
                print( s.nodes[2*j].left_memo.entangled_memory,
                       s.nodes[2*j-2].right_memo.entangled_memory )



        #print( s.links )
        '''
        if len( s.links ) == 0:
            #_process = Process( s, 'doSwappingPart', [] )
            #s.updateProcessToSchedule( _process )
            #s.tl.schedule( Event( s.tl.now() + 10, _process ) )
            return
        '''
        #_memosToTrack = []
        for _link in s.links:
            _j1, _j2 = _link.node_left, _link.node_right
            s.nodes[2*_j1].create_eg_protocol( s.nodes[2*_j1+1].name, s.nodes[2*_j2].name, 1 )
            s.nodes[2*_j2].create_eg_protocol( s.nodes[2*_j1+1].name, s.nodes[2*_j1].name, 0 )
            #_memosToTrack += [ ( _j1, 1 ), ( _j2, 0 ), ]
            _link.node_left_memo_updated = False
            _link.node_right_memo_updated = False
            _link.established = False
            _link.genProtocolFinished = False
            _link.memoryExpired = False
            
            #print( 'now pairing:', nodes[2*j-2].name, nodes[2*j-1].name, nodes[2*j].name )
            pair_eg_protocol( s.nodes[2*_j1].protocols[1], s.nodes[2*_j2].protocols[0] )

            #nodes[2*j-2].right_memo.reset()
            #nodes[2*j].left_memo.reset()

        #s.updateTrackingList( _memosToTrack )
        #s.numLinksToBeUpdatedAtThisSession = len( s.links )
        #s.numGeneratedLinksReported = 0
        '''
        s.nodes[ 2*s.cnodeNo ].cnodeData.numLinksToBeUpdatedAtThisSession = len( s.links )
        s.nodes[ 2*s.cnodeNo ].cnodeData.genLinksUpdated = 0
        '''
        s.tl.init()

        for _link in s.links:
            _j1, _j2 = _link.node_left, _link.node_right
            s.nodes[2*_j1].protocols[1].start()
            s.nodes[2*_j2].protocols[0].start()
            
        #s.blockResourceManagerUpdating = False

        s.genCounter += len( s.links )
        #s.updateProcessToSchedule( Process( s, 'doGenerationPart', [] ) )
        #print( 'dbg: gen_attemt_start:',  s.tl.now() * s.light_speed / s.endnode_distance )
        

    def doSwappingPart( s ):   
        if doDubugOutput:
            print( '################# swapping ###################' )
            print( 'time:', s.tl.now() )
            for j in range( 1, s.numRouters + 2 ):
                print( s.nodes[2*j].left_memo.entangled_memory,
                       s.nodes[2*j-2].right_memo.entangled_memory )
        _numHoppings = s.checkPathIsConsistent()
        #print( 'dbg: swapping iteration begin:',  s.tl.now() * s.light_speed / s.endnode_distance )
        #print( _numHoppings )
        if _numHoppings is None:
            #print( 'doSwappingPart returns False' )
            s.result = False
            s.resetAllMemories()
            s.flushTotallyAllPendingEvents()
            #s.resetUsedMemory()
            #print( 'dbg: no need swapping, FAILED' )
            return

        if _numHoppings == 1:
            #print( 'doSwappingPart returns True' )
            s.result = True
            s.resetAllMemories()
            s.flushTotallyAllPendingEvents()
            #s.resetUsedMemory()
            #print( 'dbg: no need swapping any more, ENTANGLED' )
            return

        leftNodeNo = 0
        entangledNodeName = s.nodes[0].right_memo.entangled_memory['node_id']
        midNodeNo = entangledNodeNo = int( entangledNodeName[4:] )

        s.establishedLinks.clear()
        s.links = []
        _success2 = True
        
        s.swpAccDelayDelta = 0.0
        __elemLinkLengthDelay = s.endnode_distance / ( s.numRouters + 1 ) / s.light_speed
        
        while True:
            rightNodeName = s.nodes[2*midNodeNo].right_memo.entangled_memory['node_id']
            if rightNodeName is None:
                print( 'wtf1' )
                _success2 = False
                break
            rightNodeNo = int( rightNodeName[4:] )

            s.nodes[2*leftNodeNo].protocols[ 3 ] = None
            s.nodes[2*rightNodeNo].protocols[ 2 ] = None
            s.nodes[2*midNodeNo].protocols[ 2:4 ] = [ None, None ]
            s.nodes[2*leftNodeNo].create_es_protocol_ends( 1 )
            s.nodes[2*rightNodeNo].create_es_protocol_ends( 0 )
            s.nodes[2*midNodeNo].create_es_protocol_middle( s.swapping_params )

            s.links.append( Link( ( leftNodeNo, rightNodeNo ) ) )            
            #memosToTrack += [ ( leftNodeNo, 1 ), ( rightNodeNo, 0 ),
            #                  #( midNodeNo, 0 ), ( midNodeNo, 1 ),
            #                  ]

            pair_es_protocol(s.nodes[2*leftNodeNo].protocols[3], s.nodes[2*midNodeNo].protocols[2])
            pair_es_protocol(s.nodes[2*rightNodeNo].protocols[2], s.nodes[2*midNodeNo].protocols[2])

            if leftNodeNo <= s.cnodeNo and rightNodeNo >= s.cnodeNo:
                pass
            elif abs( s.cnodeNo - leftNodeNo ) < abs( s.cnodeNo - rightNodeNo ):
                __extraDelayNeeded = abs( s.cnodeNo - leftNodeNo ) * __elemLinkLengthDelay
                if __extraDelayNeeded > s.swpAccDelayDelta:
                    s.swpAccDelayDelta = __extraDelayNeeded
            elif abs( s.cnodeNo - rightNodeNo ) < abs( s.cnodeNo - leftNodeNo ):
                __extraDelayNeeded = abs( s.cnodeNo - rightNodeNo ) * __elemLinkLengthDelay
                if __extraDelayNeeded > s.swpAccDelayDelta:
                    s.swpAccDelayDelta = __extraDelayNeeded
            else:
                raise ValueError( 'leftNode=%d, midNode=%d, rightNode=%d, cnodeNo=%d' % (
                    leftNodeNo, midNodeNo, rightNodeNo, s.cnodeNo ) )                    

            if rightNodeNo == s.numRouters + 1:
                break
                    
            midNodeName = s.nodes[2*rightNodeNo].right_memo.entangled_memory['node_id']
            if midNodeName is None:
                print( 'wtf2' )
                _success2 = False
                break
            midNodeNo = int( midNodeName[4:] )
            if midNodeNo == s.numRouters + 1:
                break

            leftNodeNo = rightNodeNo
                    
            continue

        if not _success2:
            raise ValueError( 'I expected path is consistent' )

        #s.updateTrackingList( memosToTrack )
                
        #s.tl.init()

        leftNodeNo = 0
        midNodeNo = entangledNodeNo
        while True:
            rightNodeName = s.nodes[2*midNodeNo].right_memo.entangled_memory['node_id']
            rightNodeNo = int( rightNodeName[4:] )

            #print( 'left/mid/right:', leftNodeNo, midNodeNo, rightNodeNo )
            #print( 'left:', s.nodes[2*leftNodeNo].protocols[ 2:4 ] )
            #print( 'right:', s.nodes[2*rightNodeNo].protocols[ 2:4 ] )
            #print( 'mid:', s.nodes[2*midNodeNo].protocols[ 2:4 ] )
            s.nodes[2*leftNodeNo].protocols[3].start()
            s.nodes[2*rightNodeNo].protocols[2].start()
            s.nodes[2*midNodeNo].protocols[2].start()

            if rightNodeNo == s.numRouters + 1:
                break
                    
            midNodeName = s.nodes[2*rightNodeNo].right_memo.entangled_memory['node_id']
            midNodeNo = int( midNodeName[4:] )
            if midNodeNo == s.numRouters + 1:
                break

            leftNodeNo = rightNodeNo
                    
            continue


    def doAction(s, __action, n):
        if len(__action) == 0:
            _process2 = Process( s, 'postAction',[])
            _time_to_wait = 1
            s.tl.schedule( Event( s.tl.now() + _time_to_wait, _process2 ) )

        else:
            i = __action[n]
            s.cnode.cnodeData.numSwapsUpdated += 1 
            rightNodeName = s.nodes[2*i].right_memo.entangled_memory['node_id']
            leftNodeName = s.nodes[2*i].left_memo.entangled_memory['node_id']
            print("############# Node", i, 'is entangled with left node:',leftNodeName, "and right node:", rightNodeName  )
            if rightNodeName is not None and leftNodeName is not None:
                rightNodeNo = int(rightNodeName[4:])
                leftNodeNo = int(leftNodeName[4:])
                midNodeNo = i


                s.nodes[2*leftNodeNo].protocols[ 3 ] = None
                s.nodes[2*rightNodeNo].protocols[ 2 ] = None
                s.nodes[2*midNodeNo].protocols[ 2:4 ] = [ None, None ]
                s.nodes[2*leftNodeNo].create_es_protocol_ends( 1 )
                s.nodes[2*rightNodeNo].create_es_protocol_ends( 0 )
                s.nodes[2*midNodeNo].create_es_protocol_middle( s.swapping_params )

                pair_es_protocol(s.nodes[2*leftNodeNo].protocols[3], s.nodes[2*midNodeNo].protocols[2])
                pair_es_protocol(s.nodes[2*rightNodeNo].protocols[2], s.nodes[2*midNodeNo].protocols[2])
                __elemLinkLengthDelay = s.endnode_distance / ( s.numRouters + 1 ) / s.light_speed

                if leftNodeNo <= s.cnodeNo and rightNodeNo >= s.cnodeNo:
                        pass
                elif abs( s.cnodeNo - leftNodeNo ) < abs( s.cnodeNo - rightNodeNo ):
                    __extraDelayNeeded = abs( s.cnodeNo - leftNodeNo ) * __elemLinkLengthDelay
                    if __extraDelayNeeded > s.swpAccDelayDelta:
                        s.swpAccDelayDelta = __extraDelayNeeded
                elif abs( s.cnodeNo - rightNodeNo ) < abs( s.cnodeNo - leftNodeNo ):
                    __extraDelayNeeded = abs( s.cnodeNo - rightNodeNo ) * __elemLinkLengthDelay
                    if __extraDelayNeeded > s.swpAccDelayDelta:
                        s.swpAccDelayDelta = __extraDelayNeeded
                else:
                    raise ValueError( 'leftNode=%d, midNode=%d, rightNode=%d, cnodeNo=%d' % (
                        leftNodeNo, midNodeNo, rightNodeNo, s.cnodeNo ) )  
                
            
                s.nodes[2*leftNodeNo].protocols[3].start()
                s.nodes[2*rightNodeNo].protocols[2].start()
                s.nodes[2*midNodeNo].protocols[2].start()           

            else:
                if s.cnode.cnodeData.numSwapsUpdated == s.cnode.cnodeData.numSwapsToBeUpdatedAtThisSession:
                    s.cnode.cnodeData.numSwapsUpdated = 0
                    _process2 = Process( s, 'postAction',[])
                    _time_to_wait = 1
                    s.tl.schedule( Event( s.tl.now() + _time_to_wait,
                                                            _process2 ) )
                    
                else:
                    _process = Process(s, 'doAction', [s.cnode.cnodeData.current_action, s.cnode.cnodeData.numSwapsUpdated])
                    _time_to_wait = 1
                    s.tl.schedule( Event(s.tl.now() + _time_to_wait, _process))
        
       


    def postAction(s):
        for i in range(len(s.cnode.cnodeData.current_state[:,0])):
            for j in range(len(s.cnode.cnodeData.current_state[:,0])):
                if s.cnode.cnodeData.current_state[i,j] != -1:
                    s.cnode.cnodeData.current_state[i,j] += 1
        
        for i in range(len(s.cnode.cnodeData.current_state[:,0])):
            for j in range(len(s.cnode.cnodeData.current_state[:,0])):
                if (i<j):
                    if s.cnode.cnodeData.current_state[i,j] >= s.cnode.cnodeData.t_cut:
                        s.cnode.cnodeData.current_state[i,j] = s.cnode.cnodeData.current_state[j,i] = -1
                        for k in range(1, abs(i-j) + 1):  
                            s.links.append(Link((i+k-1, i+k)))

        s.cnode.cnodeData.numLinksToBeUpdatedAtThisSession = len( s.links )
        s.cnode.cnodeData.genLinksUpdated = 0
        print("Nlinks_for next", len(s.links))
        print("Finally the current state", s.cnode.cnodeData.current_state)
        for i in range(len(s.cnode.cnodeData.terminal_states)):

            if ((s.cnode.cnodeData.current_state == s.cnode.cnodeData.terminal_states[i]).all()):
                s.result = True
                s.resetAllMemories()
                s.flushTotallyAllPendingEvents()
                #s.resetUsedMemory()
                #print( 'dbg: no need swapping any more, ENTANGLED' )
                print("######################FINISHED#################")
                return
            
        print('DoGeneration recalled')
        process = Process( s, 'doGenerationPart', [] )
        _time_to_wait = s.signalingDelay + 1
        s.tl.schedule( Event( s.tl.now() + _time_to_wait, process ) )
        
def runSimulations( _numRouters, _distance, _light_speed, _memory_params, _detector_params, _qchannel_params, _swapping_params, _repetitions ):
    _t0 = time.time()
    tl = Timeline()
    tl.init()

    nodesTracker = NodesTracker( tl, _numRouters, _repetitions, _light_speed, _distance )
    _fiberPieceLength = _distance / ( 2 * ( _numRouters + 1 ) )
    nodes = [ EntangleGenNode( 'node0', tl, nodesTracker, **_memory_params ) ]
    for i in range( 1, _numRouters + 2 ):
        nodei = EntangleGenNode( 'node%d' % i, tl, nodesTracker, **_memory_params )
        bsmNodei = BSMNode( 'bsm_node%d' % i , tl, [ nodes[-1].name, nodei.name ] )
        #print( nodes[-1].name, nodei.name )

        for name, param in _detector_params.items():
            bsm_name = bsmNodei.name + '.BSM'
            bsmNodei.components[bsm_name].update_detectors_params( name, param )
        
        _qchannel_params[ "distance" ] = _fiberPieceLength
        
        qc1 = QuantumChannel( 'qc%dr' % ( i - 1 ), tl, **_qchannel_params, light_speed = _light_speed )
        qc2 = QuantumChannel( 'qc%dl' % i, tl, **_qchannel_params, light_speed = _light_speed )
        qc1.set_ends( nodes[-1], bsmNodei.name )
        qc2.set_ends( nodei, bsmNodei.name )

        nodes += [ bsmNodei, nodei ]
        
    for i in range( len( nodes ) ):
        for j in range( len( nodes ) ):
            if i == j:
                continue
            cc = ClassicalChannel( 'cc_%s_%s' % (nodes[i].name, nodes[j].name),
                                   tl, abs(i-j)*_fiberPieceLength )
            cc.set_ends( nodes[i], nodes[j].name )

    nodesTracker.setNodesList( nodes )
    nodesTracker.cnode.cnodeData.t_cut = 3
    jsonFile =  open('n5_p0.500_ps0.500_tc2_tol1e-05.json', encoding="utf-8-sig") 
    policy_data = json.load(jsonFile)
    policy_object = policy(policy_data)
    nodesTracker.cnode.cnodeData.addPolicy(policy_object)
    nodesTracker.setSwappingParams( _swapping_params )

    numSuccessCases = 0
    links = []
    delivery_times = []
    for i in range(_repetitions):
        print('repetition number: ',i)
        nodesTracker.reset()
        nodesTracker.cnode.cnodeData.genLinksEstablished = 0
        assert len( links ) == 0
        links.clear()

        for j in range( 1, _numRouters + 2 ):
            links.append( Link( ( j-1, j ) ) )
        
        nodesTracker.updateLinksToEstablish( links )
        nodesTracker.cnode.cnodeData.numLinksToBeUpdatedAtThisSession = len( links )
        nodesTracker.cnode.cnodeData.genLinksUpdated = 0
        process = Process( nodesTracker, 'doGenerationPart', [] )
        nodesTracker.updateProcessToSchedule( process )
        _time_to_wait = nodesTracker.signalingDelay + 1
        tl.schedule( Event( tl.now() + _time_to_wait, process ) )
        tl.init()
        tl.run()
        _time_end_virt = tl.now()-sum(delivery_times)
        print('result:', nodesTracker.result)
        delivery_times.append(_time_end_virt)
        if tl.now() > 1160e12:
            print('time limit reached')
            break
    avg_delivery_time = [x/1e9 for x in delivery_times]
    return  sum(avg_delivery_time)/len(avg_delivery_time)



N = 1 
coherence_time_avg1 = 0.1e-3
coherence_time_avg = 0.5e-3
coherence_time_stdev = 0.2 * coherence_time_avg1
attenuation = 0.0002
mem_eff = 0.9 #9 #3
det_eff = 0.8 #3
swap_probability = 0.5 #5 #9
swap_degradation = 0.95
maxNumRepeaters = 1
max_distance_km = 1
distance_step_km = 1.0
light_speed = 2e-4
_numRepeaters= 3

memory_params = { "fidelity": 0.9, "frequency": 2e6, "efficiency": mem_eff,
                      "coherence_time": coherence_time_avg,
                      "coherence_time_stdev": coherence_time_stdev,
                      "wavelength": 500 }

detector_params = { "efficiency": det_eff, 
                        #"dark_count": 10, 
                        "time_resolution": 150, 
                        "count_rate": 50e7 
                        }
qchannel_params = { "attenuation": attenuation, 
                    "polarization_fidelity": 1.0, 
                    "distance": 0 }
swapping_params = { "success_prob": swap_probability,
                    "degradation": swap_degradation }

print(runSimulations( _numRepeaters, max_distance_km, light_speed, memory_params, detector_params, qchannel_params, swapping_params, N))
