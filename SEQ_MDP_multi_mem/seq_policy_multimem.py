import os
import re
import time
import pickle
import numpy as np
from sequence.utils import log
import logging

from sequence.components.memory import Memory
from sequence.kernel.timeline import Timeline
from sequence.kernel.process import Process
from sequence.kernel.event import Event
from sequence.kernel.timeline import Timeline
from sequence.topology.node import BSMNode
from sequence.components.optical_channel import QuantumChannel, ClassicalChannel
from sequence.topology.node import Node
from sequence.components.memory import MemoryWithRandomCoherenceTime
from sequence.entanglement_management.entanglement_protocol import EntanglementProtocol
from sequence.entanglement_management.generation import EntanglementGenerationA
from sequence.resource_management.resource_manager import ResourceManagerMsgType
from sequence.entanglement_management.generation import GenerationMsgType
from sequence.entanglement_management.swapping import EntanglementSwappingA, EntanglementSwappingB
from sequence.entanglement_management.swapping import EntanglementSwappingMessage, SwappingMsgType, EntanglementSwappingA, EntanglementSwappingB
from typing import Dict, Tuple, List, Any
from multimem_policy import MultiMemPolicy
from enum import Enum, auto
from sequence.message import Message
from sequence.components.circuit import Circuit

# 1) configure the root logger (or the one in sequence.utils)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
                    handlers=[
                        logging.FileHandler("simulation.log"),
                        logging.StreamHandler()   # still prints to console
                    ])

# 2) grab the logger used by sequence
logger = log.logger  # this is a standard python logger
logger.setLevel(logging.INFO)  # or DEBUG to get more detail



from dataclasses import dataclass
from typing import List
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
            print("FIDELITY OF LEFT AND RIGHT", self.left_memo.name, self.right_memo.name, fidelity)
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

@dataclass
class Link:
    """
    Represents exactly one memory‐pair slot between two nodes.

    Attributes:
        node_left (int): Index of the left node.
        node_right (int): Index of the right node.
        mem_idx (int): Index of the memory slot at each node used for this link.
    """
    node_left: int
    node_right: int
    mem_idx: int

    def __repr__(self) -> str:
        return f"Link({self.node_left}, {self.node_right}, mem_idx={self.mem_idx})"

def pair_eg_protocol(p1: EntanglementProtocol, p2: EntanglementProtocol) -> None:
    """
    Pair two EntanglementGenerationA protocols for the same memory slot.

    This sets up their cross‑references so each knows the other's protocol name,
    node, and memory identifier, enabling proper photon exchange and callbacks.

    Args:
        p1 (EntanglementProtocol): The first EGA protocol instance.
        p2 (EntanglementProtocol): The second EGA protocol instance.
    """
    if not isinstance(p1, EntanglementProtocol) or not isinstance(p2, EntanglementProtocol):
        raise ValueError("pair_eg_protocol expects two EntanglementProtocol instances.")

    # Each protocol has a single memory in .memories[0]
    mem1_name = p1.memories[0].name
    mem2_name = p2.memories[0].name

    # Inform p1 about p2
    p1.set_others(
        protocol=p2.name,
        node=p2.own.name,
        memories=[mem2_name]
    )
    # Inform p2 about p1
    p2.set_others(
        protocol=p1.name,
        node=p1.own.name,
        memories=[mem1_name]
    )

class SimpleManager:
    """
    Resource manager for an EntangleGenNode, handling generation callbacks
    and forwarding link‑status updates to the central control node.
    """

    def __init__(self, owner: Any, nodesTracker: Any):
        self.owner = owner
        # extract numeric index from owner.name, e.g. "node3" → 3
        self.nodeNo = int(owner.name.replace("node", ""))
        self.nodesTracker = nodesTracker

    def update(self, protocol: EntanglementProtocol, memory: Any, state: str) -> None:
        """
        Called by protocols when something happens to a memory.
        We only handle generation updates here; swapping will come later.
        """
        # generation callbacks (EGA)
        if protocol.name.split('.')[1] == 'EGA':
            self._update_eg(protocol, memory, state)

        # swapping callbacks (ESA on the middle node)
        elif protocol.name.split('.')[1] == 'ESA':
            self._update_es(protocol, memory, state)

    def _update_eg(self, protocol: EntanglementProtocol, memory: Any, state: str) -> None:
        """
        Handle EntanglementGenerationA callbacks.

        Args:
            protocol: the EGA instance that just finished.
            memory:   the Memory object involved.
            state:    'ENTANGLED' or 'RAW'.
        """
        # Determine success
        established = (state == "ENTANGLED")

        # Parse which memory slot: name ends with ".R{k}" or ".L{k}"
        suffix = memory.name.split(".")[-1]  # e.g. "R2" or "L2"
        mem_idx = int(suffix[1:])

        # Figure out the remote node index
        remote_node_name = protocol.remote_node_name
        remote_no = int(remote_node_name.replace("node", ""))
        # only the "right" endpoint (higher nodeNo) reports to the control node
        if self.nodeNo < remote_no:
            return
        # Build and send the control‑node message
        msg = ControlNodeMessage(
            ControlNodeMsgType.LINK_STATUS_UPDATED,
            "generation",
            established=established,
            left=self.nodeNo,
            right=remote_no,
            mem_idx=mem_idx
        )

        cnode = self.nodesTracker.cnode
        # send from the right node up to the central control node
        if self.nodeNo != self.nodesTracker.cnode_no:
            self.owner.send_message(cnode.name, msg)
        else:
            # if we somehow are the control node, deliver directly
            cnode.receive_message(self.owner.name, msg)
        
    def _update_es(self, protocol: EntanglementProtocol, memory: Any, state: str) -> None:
        """Handle swapping callbacks from ESA.  Only do work once, on the left‐side memory."""
        # only run this logic once per swap: ignore the right‐memory callback
        if memory is not protocol.left_memo:
            return
        print("update_es called by:", self.owner.name,
              "slot-protocol:", protocol.name,
              " state:", state)

        # --- 1) extract slot index and node indices ---
        # protocol.name == "<node>.ESA.mem<k>"
        slot_idx = int(protocol.name.rsplit("mem", 1)[1])

        midNodeNo = self.nodeNo
        leftNodeNo  = int(protocol.left_node .replace("node",""))
        rightNodeNo = int(protocol.right_node.replace("node",""))

        # did the swap succeed?
        established = (state == "SWAPPED")

        # --- 2) reset or transfer memories ---
        if not established:
            # failure: reset both local memories on middle + remote ends
            protocol.left_memo.reset()
            protocol.right_memo.reset()
            self.nodesTracker.nodes[2*leftNodeNo ].right_memos[slot_idx].reset()
            self.nodesTracker.nodes[2*rightNodeNo].left_memos [slot_idx].reset()
        else:
            # success: compute new fidelity & expire_time
            fidelity    = protocol.updated_fidelity(
                               protocol.left_memo.fidelity,
                               protocol.right_memo.fidelity)
            expire_time = min(
                               protocol.left_memo.get_expire_time(),
                               protocol.right_memo.get_expire_time())

            # reset middle memories
            protocol.left_memo.reset()
            protocol.right_memo.reset()

            # apply to remote ends:
            left_rm  = self.nodesTracker.nodes[2*leftNodeNo].right_memos[slot_idx]
            right_rm = self.nodesTracker.nodes[2*rightNodeNo].left_memos[slot_idx]

            left_rm.fidelity = fidelity
            right_rm.fidelity = fidelity
            left_rm.update_expire_time(expire_time)
            right_rm.update_expire_time(expire_time)

            # now they are entangled with each other
            left_rm.entangled_memory ["node_id"] = f"node{rightNodeNo}"
            left_rm.entangled_memory ["memo_id"] = f"R{slot_idx}"
            right_rm.entangled_memory["node_id"] = f"node{leftNodeNo}"
            right_rm.entangled_memory["memo_id"] = f"L{slot_idx}"

        # --- 3) send the swap‐status up to the control node ---
        msg = ControlNodeMessage(
            ControlNodeMsgType.SWAP_STATUS_UPDATED,
            "swapping",
            established=established,
            left=leftNodeNo,
            middle=midNodeNo,
            right=rightNodeNo,
            mem_idx=slot_idx
        )
        cnode = self.nodesTracker.cnode
        if midNodeNo != self.nodesTracker.cnode_no:
            self.owner.send_message(cnode.name, msg)
        else:
            cnode.receive_message(self.owner.name, msg)

class ControlNodeMsgType(Enum):
    """
    Types of messages sent from repeater nodes to the central control node.
    """
    LINK_STATUS_UPDATED = auto()
    SWAP_STATUS_UPDATED = auto()

class ControlNodeMessage(Message):
    """
    Message wrapper carrying link‐generation or swap results to the control node.

    Attributes (for LINK_STATUS_UPDATED):
        origin (str): 'generation' or 'swapping'
        established (bool): True if the link or swap succeeded
        left (int): index of the local node for this event
        right (int): index of the remote node for this event
        mem_idx (int): memory slot index involved

    Attributes (for SWAP_STATUS_UPDATED):
        Similar fields; may include additional swap details.
    """
    def __init__(
        self,
        msg_type: ControlNodeMsgType,
        origin: str,
        **kwargs
    ):
        # Destination is the control node; base class stores msg_type and dst
        super().__init__(msg_type, receiver="ControlNode")
        self.origin = origin

        if msg_type is ControlNodeMsgType.LINK_STATUS_UPDATED:
            self.established = kwargs.get('established', False)
            self.left        = kwargs['left']
            self.right       = kwargs['right']
            self.mem_idx     = kwargs['mem_idx']
        elif msg_type is ControlNodeMsgType.SWAP_STATUS_UPDATED:
            self.established = kwargs.get('established', False)
            self.left        = kwargs['left']
            self.middle      = kwargs['middle']
            self.right       = kwargs['right']
            self.mem_idx     = kwargs['mem_idx']
        else:
            raise ValueError(f"Unknown ControlNodeMsgType: {msg_type}")

class ControlNodeData:
    """
    Holds shared state and counters for the central control node in a multimem simulation.
    """

    def __init__(self, num_nodes: int):
        # Initialize a zeroed N×N state matrix
        self.current_state: np.ndarray = np.zeros((num_nodes, num_nodes), dtype=int)
        self.policy: MultiMemPolicy = None

        # Generation-phase counters
        self.genLinksUpdated: int = 0
        self.numLinksToBeUpdatedAtThisSession: int = 0

        # Swap-phase counters
        self.current_action: List[Tuple[int,int]] = []
        self.numSwapsUpdated: int = 0
        self.numSwapsToBeUpdatedAtThisSession: int = 0

    def add_policy(
        self,
        policy: MultiMemPolicy,
        c_left: List[int],
        c_right: List[int]
    ) -> None:
        """
        Attach the trained policy, re-initialize state & counters, and
        compute the terminal threshold for end-to-end entanglement.

        Args:
            policy:     a MultiMemPolicy instance (loaded via `.load()`).
            c_left:     left-side memory capacities for each node.
            c_right:    right-side memory capacities for each node.
        """
        self.policy = policy

        # Reset state matrix to zeros with correct dimensions
        N = policy.N
        self.current_state = np.full((N, N), np.inf)

        # Reset counters
        self.genLinksUpdated = 0
        self.numLinksToBeUpdatedAtThisSession = 0
        self.current_action = []
        self.numSwapsUpdated = 0
        self.numSwapsToBeUpdatedAtThisSession = 0

        # Compute how many end-to-end links we require before terminating:
        # it's half the bottleneck capacity between node 0’s right ports and
        # node N–1’s left ports.
        cap_end0 = c_right[0]
        cap_endN = c_left[-1]
        self.terminal_threshold = int(min(cap_end0, cap_endN) / 2)

    def reset(self) -> None:
        """Reset state matrix and all counters for a new generation↔swap cycle."""
        self.current_state.fill(0)
        self.genLinksUpdated = 0
        self.numLinksToBeUpdatedAtThisSession = 0
        self.current_action = []
        self.numSwapsUpdated = 0
        self.numSwapsToBeUpdatedAtThisSession = 0

    def find_action(self) -> List[Tuple[int,int]]:
        """Lookup the next swap action from the policy, given the current state."""
        if self.policy is None:
            raise RuntimeError("Policy must be set before calling find_action().")
        action = self.policy.get_action(self.current_state)
        self.current_action = action
        self.numSwapsToBeUpdatedAtThisSession = len(action)
        self.numSwapsUpdated = 0
        return action
    
class EntangleGenNode(Node):
    """
    Extended to support C_left and C_right memories, with protocol mappings by slot index.

    Attributes:
        left_memos (List[MemoryWithRandomCoherenceTime])
        right_memos (List[MemoryWithRandomCoherenceTime])
        protocols (Dict[Tuple[str,int], Any]):
            Maps ('EGA', side, slot_idx) or ('ES', side, slot_idx) to protocol instances.
        resource_manager: manages callbacks including mem_idx in messages.
    """
    def __init__(
        self,
        name: str,
        tl: Timeline,
        nodes_tracker,
        c_left: int,
        c_right: int,
        **memory_params
    ):
        super().__init__(name, tl)
        # Create left and right memory pools
        self.left_memos: List[MemoryWithRandomCoherenceTime] = []
        self.right_memos: List[MemoryWithRandomCoherenceTime] = []
        for k in range(c_left):
            m = MemoryWithRandomCoherenceTime(f"{name}.L{k}", tl, **memory_params)
            m.owner = self; m.add_receiver(self); m.attach(self)
            self.left_memos.append(m)
        for k in range(c_right):
            m = MemoryWithRandomCoherenceTime(f"{name}.R{k}", tl, **memory_params)
            m.owner = self; m.add_receiver(self); m.attach(self)
            self.right_memos.append(m)

        # Protocol dict: keys like ('EGA', , side, slot_idx) or ('ES',side, slot_idx)
        self._protocols: Dict[Tuple[str,str,int], Any] = {}

        # Resource manager expecting mem_idx in callbacks
        self.resource_manager = SimpleManager(self, nodes_tracker)

        self.nodesTracker = self.resource_manager.nodesTracker

    def get(self, photon, **kwargs):
        self.send_qubit(kwargs['dst'], photon)

    def create_eg_protocol(
        self,
        slot_idx: int,
        side: str,
        middle_node: str,
        other_node: str
    ) -> None:
        """
        Instantiate an EntanglementGenerationA protocol for a given memory slot on either the left or right side.

        Args:
            slot_idx (int): Index of the memory slot.
            side (str): 'left' or 'right', determines which memory list to use.
            middle_node (str): Name of the BSM node in between.
            other_node (str): Name of the remote EntangleGenNode partner.
        """
        # Select local memory based on side
        if side == 'right':
            local_mem = self.right_memos[slot_idx]
            proto_suffix = 'r'
        elif side == 'left':
            local_mem = self.left_memos[slot_idx]
            proto_suffix = 'l'
        else:
            raise ValueError(f"Invalid side '{side}'; expected 'left' or 'right'.")

        proto_name = f"{self.name}.EGA.{proto_suffix}.mem{slot_idx}"
        ega = EntanglementGenerationA(
            self,
            proto_name,
            middle_node,
            other_node,
            local_mem
        )
        # Store under a key including slot index
        self._protocols[('EGA', side, slot_idx)] = ega
        self.protocols.append(ega)

    def create_es_protocol(self,
                           slot_idx: int,
                           left_node_name: str,
                           right_node_name: str) -> None:
        """
        On the middle repeater, instantiate the A‐side swapping protocol
        for slot `slot_idx`, paired to the two memories on that slot.
        """
        left_mem  = self.left_memos[slot_idx]
        right_mem = self.right_memos[slot_idx]
        nameA = f"{self.name}.ESA.mem{slot_idx}"
        esa = EntanglementSwappingA_global(
            self,
            nameA,
            left_mem,
            right_mem,
            **self.nodesTracker.swapping_params
        )
        self._protocols[('ESA', slot_idx)] = esa
        self.protocols.append(esa)

    def create_es_b_protocol(self,
                             slot_idx: int,
                             side: str) -> None:
        """
        On one of the ends (left or right), instantiate the B‐side
        swapping protocol for slot `slot_idx`.
        """
        if side == 'left':
            mem  = self.left_memos[slot_idx]
            suffix = 'l'
        elif side == 'right':
            mem  = self.right_memos[slot_idx]
            suffix = 'r'
        else:
            raise ValueError(f"Invalid side '{side}'; must be 'left' or 'right'.")

        nameB = f"{self.name}.ESB.{suffix}.mem{slot_idx}"
        esb = EntanglementSwappingB_global(self, nameB, mem)
        self._protocols[('ESB', side, slot_idx)] = esb
        self.protocols.append(esb)

    def receive_message(self, src: str, msg: Any) -> None:
        # 1) Generation callbacks
        if isinstance(msg.msg_type, GenerationMsgType):
            # special‐case MEAS_RES (no msg.receiver) → fan‐out to all EGAs
            if msg.msg_type is GenerationMsgType.MEAS_RES:
                for proto in self._protocols.values():
                    if isinstance(proto, EntanglementGenerationA):
                        proto.received_message(src, msg)
                return
            # otherwise NEGOTIATE / NEGOTIATE_ACK come with msg.receiver set
            else:    
                dest = msg.receiver    # <— the protocol instance name
                for proto in self._protocols.values():
                    if proto.name == dest:
                        proto.received_message(src, msg)
                        return

        # 2) Swapping callbacks
        if isinstance(msg.msg_type, SwappingMsgType):
            dest = msg.receiver
            for proto in self._protocols.values():
                if proto.name == dest:
                    proto.received_message(src, msg)
                    return

        # 3) Resource manager
        if isinstance(msg.msg_type, ResourceManagerMsgType):
            # unchanged
            self.resource_manager.update(self, msg)
            return

        # 4) Control node messages
        if isinstance(msg, ControlNodeMessage):
            
            cdata = self.nodesTracker.cnode.cnodeData
            # generation results
            if msg.origin == 'generation':
                cdata.genLinksUpdated += 1
                if msg.established:
                    print("__DEBUG__ success link gen for", msg.left, msg.right)
                    l, r = msg.left, msg.right
                    # “first hit” → 1; else increment
                    if np.isinf(cdata.current_state[l, r]):
                        val = 1
                    else:
                        val = cdata.current_state[l, r] + 1

                    # write both halves to keep symmetry
                    cdata.current_state[l, r] = val
                    cdata.current_state[r, l] = val
                else:
                    print("__DEBUG__ failed link gen for", msg.left, msg.right)
                    # always store (small_index, large_index) so left<=right
                    i, j = sorted((msg.left, msg.right))
                    link = Link(i, j, msg.mem_idx)
                    # only append if we haven’t already queued that exact link
                    if link not in self.nodesTracker.links_for_next_session:
                        self.nodesTracker.links_for_next_session.append(link)
                # all received?
                if cdata.genLinksUpdated == cdata.numLinksToBeUpdatedAtThisSession:
                    
                     # existing requeue logic...
                    action    = cdata.find_action()
                    print("cdata.genLinksUpdated == cdata.numLinksToBeUpdatedAtThisSession__DEBUG__")
                    print("current state:", cdata.current_state)
                    print("corresponding action,", action)
                    swap_jobs = self.nodesTracker.build_swap_jobs(action)
                    cdata.swap_jobs = swap_jobs
                    print(f"Swap jobs for this one __DEBUG__, {swap_jobs}")
                    cdata.numSwapsToBeUpdatedAtThisSession = len(swap_jobs)
                    cdata.numSwapsUpdated = 0

                    # start first swap after signaling delay
                    if not swap_jobs:
                        # no swaps to do → cleanup / requeue
                        proc = Process(self.nodesTracker, 'postAction', [])
                        self.timeline.schedule(Event(self.timeline.now() + 1, proc))
                    else:
                        print(f"__DEBUG__ action triggered with swap_jobs={swap_jobs}")
                        proc = Process(self.nodesTracker, 'doAction', [swap_jobs, 0])
                        delay = self.nodesTracker.signaling_delay
                        self.timeline.schedule(Event(self.timeline.now() + delay, proc))
                return

            # swapping results
            if msg.origin == 'swapping':
                cdata.numSwapsUpdated += 1

                l, m, r, k = msg.left, msg.middle, msg.right, msg.mem_idx

                 
                if msg.established:
                    # 1) remove the two “legs” that got consumed
                    for a, b in ((l, m), (m, r)):
                        old = cdata.current_state[a, b]
                        if not np.isinf(old):
                            new = old - 1
                            # 0 links → back to inf (meaning “no link”)
                            if new == 0:
                                new = np.inf
                            cdata.current_state[a, b] = new
                            cdata.current_state[b, a] = new

                    # 2) add the new long link at (l,r)
                    old_lr = cdata.current_state[l, r]
                    val    = 1 if np.isinf(old_lr) else old_lr + 1
                    cdata.current_state[l, r] = val
                    cdata.current_state[r, l] = val

                    if self.nodesTracker.verbose:
                        print(f"[swap SUCCESS] slot {k} between {l}-{r} -> new state[{l},{r}] = {val}")

                     # 3) requeue the two freed elementary‐link slots for next gen wave
                    for a, b in ((l, m), (m, r)):
                        i, j = sorted((a, b))
                        freed = Link(i, j, k)
                        if freed not in self.nodesTracker.links_for_next_session:
                            self.nodesTracker.links_for_next_session.append(freed)
                else:
                    if self.nodesTracker.verbose:
                        print(f"[swap FAIL] slot {k} between {l}-{r} -> requeue gen links {(l,m)} & {(m,r)}")
                    # requeue both constituent gen links for next wave
                    for a, b in ((l, m), (m, r)):
                        i, j = sorted((a, b))
                        link = Link(i, j, k)
                        if link not in self.nodesTracker.links_for_next_session:
                            self.nodesTracker.links_for_next_session.append(link)
                # Decide next step
                if cdata.numSwapsUpdated < cdata.numSwapsToBeUpdatedAtThisSession:
                    next_idx = cdata.numSwapsUpdated
                    proc = Process(self.nodesTracker, 'doAction',
                                [self.nodesTracker.cnode.cnodeData.swap_jobs, next_idx])
                    self.timeline.schedule(Event(self.timeline.now(), proc))
                else:
                    proc = Process(self.nodesTracker, 'postAction', [])
                    self.timeline.schedule(Event(self.timeline.now() + 1, proc))

                return                                          
        else:
            # Otherwise unhandled
            raise ValueError(f"Unexpected message at {self.name}: {msg} \n this is receiver: {msg.receiver}")

    # Further methods (start, receive_message, create_es_protocol) to handle protocols by slot_idx

def pair_es_protocol(p1: EntanglementProtocol, p2: EntanglementProtocol) -> None:
    """
    Pair two swapping protocols so each knows the other’s name and memories.

    - p1 is the EntanglementSwappingB on an end node.
    - p2 is the EntanglementSwappingA on the middle node (or vice-versa).
    """
    if not isinstance(p1, EntanglementProtocol) or not isinstance(p2, EntanglementProtocol):
        raise ValueError("pair_es_protocol expects two EntanglementProtocol instances.")

    # They each hold exactly one memory:
    mem1 = p1.memories[0].name
    mem2 = p2.memories  # for A, this is a list of two memories

    # Tell each about the other
    p1.set_others(
        protocol=p2.name,
        node=p2.own.name,
        memories=[m.name for m in p2.memories]
    )
    p2.set_others(
        protocol=p1.name,
        node=p1.own.name,
        memories=[mem1]
    )

class NodesTracker:
    """
    Manages the timeline, node list, link queues, and scheduling for multimemory entanglement simulations.

    Attributes:
        tl (Timeline): Sequence simulation timeline.
        L (List[float]): Distances for each elementary link between adjacent nodes.
        light_speed (float): Speed of light (distance units per time unit).
        num_edges (int): Number of elementary links (len(L)).
        num_nodes (int): Total number of nodes = num_edges + 1.
        num_repeaters (int): Number of intermediate repeater nodes = num_nodes - 2.
        cnode_no (int): Index of the control node (central repeater).
        nodes (List): All Node and BSM objects in topology.
        cnode: Reference to the control node object for quick access.
        links (List[Link]): Pending Link objects for generation.
        links_for_next_session (List[Link]): Failed links queued for next session.
        swapping_params (dict): Parameters for entanglement swapping.
        signaling_delay (float): Classical signaling delay from farthest node to control node.
        time_steps (List[float]): Timeline times of generation cycles.
        time_debug (List[float]): Durations between events for debugging.
        process: Last scheduled Process.
        result: Boolean success/failure of current iteration.
    """
    def __init__(
        self,
        tl: Timeline,
        L: List[float],
        light_speed: float,
        swapping_params: dict
    ):
        # Core simulation parameters
        self.tl = tl
        self.L = L
        self.light_speed = light_speed
        self.swapping_params = swapping_params
        self.verbose = True
        # Topology counts
        self.num_edges = len(L)
        self.num_nodes = self.num_edges + 1
        self.num_repeaters = self.num_nodes - 2

        # Control node index (central repeater)
        self.cnode_no = self.num_nodes // 2

        # Placeholders for nodes & control node reference
        self.nodes = []
        self.cnode = None

        # Link management
        self.links = []
        self.links_for_next_session = []

        # Scheduling and result tracking
        self.process = None
        self.result = None
        self.gen_counter = 0
        self.num_trials = 0

        # Compute signaling delay: max distance to ends
        dist_left  = sum(self.L[:self.cnode_no])
        dist_right = sum(self.L[self.cnode_no:])
        max_dist   = max(dist_left, dist_right)
        self.signaling_delay = max_dist / self.light_speed

        # Time‑debug logs
        self.time_steps = [0.0]
        self.time_debug = []

    def reset(self) -> None:
        """
        Reset tracker state for a new cycle.
        """
        self.links.clear()
        self.links_for_next_session.clear()
        self.process = None
        self.result = None
        self.gen_counter = 0
        self.num_trials = 0
        self.time_steps = [0.0]
        self.time_debug = []
        if self.cnode and hasattr(self.cnode, 'cnodeData'):
            self.cnode.cnodeData.reset()

    def set_nodes_list(self, nodes: List) -> None:
        """
        Set the topology's node objects, identify the control node, and initialize its data.

        Args:
            nodes (List): Ordered list of Node and BSMNode instances in sequence.
        """
        self.nodes = nodes
        # control node sits at index = 2 * cnode_no in alternating list
        self.cnode = self.nodes[2 * self.cnode_no]
        # initialize control node data
        self.cnode.cnodeData = ControlNodeData(self.num_nodes)
        # pass swapping params
        self.cnode.cnodeData.policy = None  # to be set after loading policy
        self.cnode.cnodeData.t_cut = None

    def update_links_to_establish(self, links: List) -> None:
        """
        Define the next set of Links for which to attempt generation.

        Args:
            links (List[Link]): New pending links for generation.
        """
        # Overwrite pending links and clear failures
        self.links = list(links)
        self.links_for_next_session.clear()

    def reset_all_memories(self) -> None:
        """
        Reset all memories across repeater nodes and remove pending events.
        """
        # Assumes alternating [EntangleGenNode, BSMNode, EntangleGenNode, ...]
        for j in range(1, self.num_repeaters + 2):
            # reset right memory of node j-1
            self.nodes[2*j - 2].right_memo.reset()
            # reset left memory of node j
            self.nodes[2*j].left_memo.reset()
        # clear any future events
        self.flush_totally_all_pending_events()

    def flush_totally_all_pending_events(self) -> None:
        """
        Remove all scheduled events from the timeline at or after current time.
        """
        now = self.tl.now()
        # Copy list to avoid mutation during iteration
        for event in list(self.tl.events):
            if not event.is_invalid() and event.time >= now:
                self.tl.remove_event(event)

    def doGenerationPart(self) -> None:
        """
        Launch entanglement generation on every pending Link(slot).
        Creates, pairs, and starts EGA protocols for each memory slot.
        """
        print(f"[doGenerationPart] wave #{self.gen_counter} starting @ t={self.tl.now():.0f} with {len(self.links)} links")
        self.num_trials += 1
        if self.verbose:
            print(f"[gen] wave #{self.gen_counter} | trying {len(self.links)} links")
        # No links => go straight to postAction (cleanup / new wave)
        if not self.links:
            proc = Process(self, 'postAction', [])
            self.tl.schedule(Event(self.tl.now() + 1, proc))
            return

        # Reset control-node counters
        cdata = self.cnode.cnodeData
        cdata.genLinksUpdated = 0
        cdata.numLinksToBeUpdatedAtThisSession = len(self.links)

        # Instantiate and pair protocols for each link slot
        for link in list(self.links):
            i, j, k = link.node_left, link.node_right, link.mem_idx
            # BSM sits between node i and node j
            middle_name = self.nodes[2*i + 1].name
            left_node  = self.nodes[2*i]
            right_node = self.nodes[2*j]

            # Create generation protocols on both ends
            left_node.create_eg_protocol(k, 'right', middle_name, right_node.name)
            right_node.create_eg_protocol(k, 'left',  middle_name, left_node.name)

            # Pair them so they know each other's protocol identity
            pair_eg_protocol(
                left_node._protocols[('EGA', 'right', k)],
                right_node._protocols[('EGA', 'left', k)]
            )

        # start all generation protocols
        self.tl.init()
        for link in list(self.links):
            i, j, k = link.node_left, link.node_right, link.mem_idx
            left_node  = self.nodes[2*i]
            right_node = self.nodes[2*j]
            left_node._protocols[('EGA', 'right', k)].start()
            right_node._protocols[('EGA', 'left', k)].start()

        # Debug timing
        now = self.tl.now()
        self.time_debug.append(now - self.time_steps[-1])
        self.time_steps.append(now)
        self.gen_counter += 1

    def build_swap_jobs(self, action: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
            """
            From a list of (node_i, num_swaps), rank candidate slots by
            total link length and flatten to a list of (node_i, slot_idx).
            """
            jobs: List[Tuple[int,int]] = []
            for i, num in action:
                node = self.nodes[2*i]  # EntangleGenNode
                # collect (slot_idx, total_dist)
                candidates: List[Tuple[int, float]] = []
                for k, (lm, rm) in enumerate(zip(node.left_memos, node.right_memos)):
                    left_name  = lm.entangled_memory.get('node_id')
                    right_name = rm.entangled_memory.get('node_id')
                    if left_name is None or right_name is None:
                        continue
                    ln = int(left_name.replace('node',''))
                    rn = int(right_name.replace('node',''))
                    # distance from ln->i plus i->rn
                    dist = sum(self.L[ln:i]) + sum(self.L[i:rn])
                    candidates.append((k, dist))
                # sort by dist desc, take top `num`
                candidates.sort(key=lambda x: x[1], reverse=True)
                for k, _ in candidates[:num]:
                    jobs.append((i, k))
            return jobs

    def doAction(self, swap_jobs: List[Tuple[int,int]], n: int = 0) -> None:
        """
        Perform exactly one swap (the nth in swap_jobs). The callback
        in receive_message will schedule the (n+1)th swap when ready.
        """
        if n >= len(swap_jobs):
            return
        i, k = swap_jobs[n]
        if self.verbose:
            print(f"[swap start] job {n+1} of {len(swap_jobs)} at repeater {i}, slot {k}")
        # identify endpoints
        node   = self.nodes[2*i]
        lm = node.left_memos[k]
        rm = node.right_memos[k]
        ln = int(lm.entangled_memory['node_id'].replace('node',''))
        rn = int(rm.entangled_memory['node_id'].replace('node',''))
        left_end  = self.nodes[2*ln]
        right_end = self.nodes[2*rn]
        mid_node  = node

        # create and pair protocols
        left_end.create_es_b_protocol(k, 'right')
        right_end.create_es_b_protocol(k, 'left')
        mid_node.create_es_protocol(k, left_end.name, right_end.name)

        pair_es_protocol(
            left_end._protocols[('ESB','right',k)],
            mid_node._protocols[('ESA',k)]
        )
        pair_es_protocol(
            right_end._protocols[('ESB','left',k)],
            mid_node._protocols[('ESA',k)]
        )

        # start them
        left_end._protocols[('ESB','right',k)].start()
        right_end._protocols[('ESB','left',k)].start()
        mid_node._protocols[('ESA',k)].start() 

    def postAction(self):

        cdata = self.cnode.cnodeData

        # 1) Check whether we've reached the end-to-end target:
        #    current_state[0, N-1] holds number of end-to-end links
        if cdata.current_state[0, self.num_nodes-1] != np.inf:
            if cdata.current_state[0, self.num_nodes-1] >= cdata.terminal_threshold:
                self.result = True
                print(f"[postAction] reached terminal threshold: {cdata.current_state[0, self.num_nodes-1]}")
                return

        # 2) Otherwise, prepare the next generation wave:
        #    take all the slots we queued up during gen+swap failures
        self.links = list(self.links_for_next_session)
        self.links_for_next_session.clear()

        # 3) Reset control-node counters for the new wave
        cdata.genLinksUpdated = 0
        cdata.numLinksToBeUpdatedAtThisSession = len(self.links)

        if self.verbose:
            print(f"[postAction] re-queuing {len(self.links)} links for next gen wave")

        # 4) Schedule the next generation after the signaling delay
        proc = Process(self, 'doGenerationPart', [])
        self.tl.schedule(Event(self.tl.now() + self.signaling_delay, proc))
        
        

def length_2_prob(prob: float) -> float:
    # memE = 0.95 ; detE = 0.95 ; alpha = 0.2 dB/km
    l = (10 / (0.2 * 1.259 * np.log(10))
         * np.log((0.95**2 * 0.95**2) / (2 * prob)))
    assert l > 0
    return l

def run_simulations(
    _light_speed,
    _memory_params,
    _detector_params,
    _qchannel_params,
    _swapping_params,
    _repetitions,
    Tcut,
    mdp_file_path,
    verbose: bool = False
):
    # 1) Extract p_list, ps, C_list from filename
    fname = os.path.basename(mdp_file_path)
    pattern = (
        r"_p([0-9\.]+(?:_[0-9\.]+)*)"
        r"_ps([0-9\.]+)"
        r"_C([0-9_]+)"
        r"(?:_.*)?\.pkl$"
    )
    m = re.search(pattern, fname)
    if not m:
        raise ValueError(f"Filename {fname!r} doesn't match expected pattern")

    p_list = [float(x) for x in m.group(1).split("_")]
    ps_val  = float(m.group(2))
    C_list  = [int(x)   for x in m.group(3).split("_")]

    # override swapping success probability
    _swapping_params["success_prob"] = ps_val

    # sanity check: #nodes = _numRouters+2
    num_nodes = len(C_list)
    _numRouters  = num_nodes - 2

    # 2) Convert link success probs → physical lengths
    L_list = [length_2_prob(p) for p in p_list]

    # 3) Designate left/right memory counts per node
    c_left  = [0] * num_nodes
    c_right = [0] * num_nodes

    # endpoints: all memories face inward
    c_left[0],        c_right[0]       = 0,           C_list[0]
    c_left[-1],       c_right[-1]      = C_list[-1],  0

    # interior nodes
    for i in range(1, num_nodes - 1):
        ratio     = p_list[i] / (p_list[i - 1] + p_list[i])
        c_left[i] = round(ratio * C_list[i])
        c_right[i]= C_list[i] - c_left[i]

    delivery_times = []
    for rep in range(_repetitions):
        # --- fresh timeline + topology setup ---
        tl = Timeline()
        tl.init()

        nodesTracker = NodesTracker(tl, L_list, _light_speed, _swapping_params)
        nodesTracker.verbose = verbose
        # attach side‑capacities for later
        nodesTracker.c_left  = c_left
        nodesTracker.c_right = c_right

        # build nodes & BSMs, assign each node its c_left/c_right
        nodes = []
        for i in range(num_nodes):
            # instantiate with per-node capacity
            node = EntangleGenNode(
                f"node{i}",
                tl,
                nodesTracker,
                c_left[i],
                c_right[i],
                **_memory_params
            )
            nodes.append(node)

            if i > 0:
                # create BSM in between node[i-1] and node[i]
                bsm = BSMNode(f"bsm_node{i}", tl, [nodes[-2].name, node.name])
                # update detector params on the BSM
                for name, val in _detector_params.items():
                    bsm.components[f"{bsm.name}.BSM"].update_detectors_params(name, val)
                nodes.insert(-1, bsm)

                # quantum channel with link‑specific distance
                _qchannel_params["distance"] = L_list[i - 1]
                qc1 = QuantumChannel(f"qc{i-1}r", tl, **_qchannel_params, light_speed=_light_speed)
                qc2 = QuantumChannel(f"qc{i}l",  tl, **_qchannel_params, light_speed=_light_speed)
                qc1.set_ends(nodes[-3], bsm.name)
                qc2.set_ends(node,        bsm.name)

        # classical channels (same as before)
        for src in nodes:
            for dst in nodes:
                if src is dst:
                    continue
                cc = ClassicalChannel(f"cc_{src.name}_{dst.name}", tl,
                                      abs(nodes.index(src) - nodes.index(dst)) * 1.0)
                cc.set_ends(src, dst.name)

        nodesTracker.set_nodes_list(nodes)
        nodesTracker.cnode.cnodeData.t_cut = Tcut + 1

        policy_object = MultiMemPolicy.load(mdp_file_path)
        nodesTracker.cnode.cnodeData.add_policy(policy_object, c_left, c_right)

        # 4) seed initial Link objects: one per memory‐slot index
        initial_links = []
        for j in range(1, num_nodes):
            cap = min(c_right[j - 1], c_left[j])
            for k in range(cap):
                initial_links.append(Link(j - 1, j, k))

        nodesTracker.update_links_to_establish(initial_links)

        # total number of slot‐links to attempt = number of Link objects
        nodesTracker.cnode.cnodeData.numLinksToBeUpdatedAtThisSession = len(initial_links)
        nodesTracker.cnode.cnodeData.genLinksUpdated = 0

        # start the sim
        proc = Process(nodesTracker, "doGenerationPart", [])
        delay = nodesTracker.signaling_delay
        tl.schedule(Event(tl.now() + delay, proc))
        tl.init()
        tl.run()

        delivery_times.append(tl.now())

    # compute avg rate (e.g. ms → s)
    times_ms = [t / 1e6 for t in delivery_times]
    avg_time = sum(times_ms) / len(times_ms)
    rate = (_repetitions / avg_time)  # or use desired e / avg_time
    print(f"Avg entanglement rate: {rate:.2f} per second")
    return rate

if __name__ == "__main__":
    # example parameters (update as needed)
    _light_speed    = 2e-4
    _memory_params  = { "fidelity": 0.99, "frequency": 2e6, "efficiency": 0.75,
                        "coherence_time": -1, "coherence_time_stdev": 0.0, "wavelength": 500 }
    _detector_params= { "efficiency": 0.8, "time_resolution": 150, "count_rate": 50e7 }
    _qchannel_params= { "attenuation": 0.0002, "polarization_fidelity": 1.0, "distance": 0 }
    _swapping_params= { "success_prob": 1.0, "degradation": 0.95 }
    _repetitions    = 1
    Tcut            = 100
    mdp_file_path   = r"C:\Users\png14\SEQ_MDP_-20250303T211238Z-001\SEQ_MDP_multi_mem\training_data\policy_N4_p0.3_0.3_0.1_ps0.5_C4_4_4_4_a0.01_g0.95_e0.1.pkl"

    run_simulations(
        _light_speed,
        _memory_params,
        _detector_params,
        _qchannel_params,
        _swapping_params,
        _repetitions,
        Tcut,
        mdp_file_path,
        verbose = True
    )

