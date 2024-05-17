from sequence.components.photon import Photon
from sequence.network_management.network_manager import ResourceReservationProtocol
from sequence.network_management.reservation import Reservation
from sequence.topology.node import QuantumRouter
from sequence.topology.router_net_topo import RouterNetTopo
from sequence.topology.node import QuantumRouter
#from src.utils.swaping_rules.ResourceReservationProtocol import create_rules
from utils.swaping_rules.ResourceReservationProtocol import create_rules
import numpy as np
import logging

# Configure logging
"""logging.basicConfig(filename='logsNodeiNodej.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
"""

from sequence.app.request_app import RequestApp
import json
import configparser
import os
N = 100
def create_quantum_network(C, output_file, config):
    num_routers = len(C)-1
    # Get memory parameters
    ATTENUATION = float(config.get('qchannel', 'attenuation'))
    DISTANCE = float(config.get('qchannel', 'distance'))
    # Initialize the network structure
    network = {
        "nodes": [],
        "qconnections": [],
        "is_parallel": False,
        "cconnections": []
    }
    # Add first node
    network["nodes"].append({"name": "Nodei", "type": "QuantumRouter", "seed": 1, "memo_size": 500})#int(C[0])})

    # Add routers
    for i in range(1, num_routers + 1):
        router_name = f"r{i}"
        network["nodes"].append({"name": router_name, "type": "QuantumRouter", "seed": 1, "memo_size": 500})#int(C[i-1]+C[i])})
    
    # Add end node
    network["nodes"].append({"name": "Nodej", "type": "QuantumRouter", "seed": 1 , "memo_size": 500})#int(C[num_routers])})
    
    # Add quantum connections
    for i in range(num_routers+1):
        connection = {
            "node1": network["nodes"][i]["name"],
            "node2": network["nodes"][i + 1]["name"],
            "distance": DISTANCE * 1e3,
            "attenuation": ATTENUATION / 1e3,
            "type": "meet_in_the_middle"
        }
        network["qconnections"].append(connection)

    # Add classical connections (mesh configuration)
    for i in range(num_routers + 2):  # Including Nodei and Nodej
        for j in range(i + 1, num_routers + 2):
            # Calculate distance for classical connections between routers
            distance = abs(i - j) * DISTANCE * 1e3
            connection = {
                "node1": network["nodes"][i]["name"],
                "node2": network["nodes"][j]["name"],
                "distance": distance
            }
            network["cconnections"].append(connection)

    # Save the JSON structure to a file
    with open(output_file, 'w') as file:
        json.dump(network, file, indent=2)

def get(self, photon: Photon, **kwargs):
    """Receives photon from last hardware element (in this case, quantum memory)."""
    if hasattr(self, 'eg_attempts'):
        self.eg_attempts += 1
    else:
        self.eg_attempts = 1
    dst = kwargs.get("dst", None)
    if dst is None:
        raise ValueError("Destination should be supplied for 'get' method on QuantumRouter")
    self.send_qubit(dst, photon)

QuantumRouter.get = get

class trackApp:
    def __init__(self, node: "QuantumRouter"):
        self.node = node
        self.node.set_app(self)
    
    def get_memory(self, info: "MemoryInfo"):
        if info.state == "ENTANGLED":
            print("\t{} app received memory {} ENTANGLED with {} at time {}".format(
                self.node.name, info.index, info.remote_node, self.node.timeline.now() * 1e-12))
            #self.node.network_manager.protocol_stack[-1].create_rules(['Nodei', 'r1', 'r2', 'r3', 'Nodej'],self.node.network_manager.protocol_stack[-1].accepted_reservation[0])



class EnranglementRequestApp(RequestApp):
    def __init__(self, node: QuantumRouter, other_node: str):
        super().__init__(node)
        self.accumulated_fidelity = 0
        self.other_node = other_node

    def get_reserve_res(self, reservation: "Reservation", result: bool):
        if result:
            print("Reservation approved at time", self.node.timeline.now() * 1e-12)
        else:
            print("Reservation failed at time", self.node.timeline.now() * 1e-12)

    def get_memory(self, info: "MemoryInfo"):
        #get_router_state(self.node, [0,1,2,3])
        if info.state == "ENTANGLED" and info.remote_node == self.other_node:
            #print("\t{} app received memory {} ENTANGLED at time {}".format(self.node.name, info.index, self.node.timeline.now() * 1e-12))
            self.memory_counter += 1
            self.accumulated_fidelity += info.fidelity
            self.node.resource_manager.update(None, info.memory, "RAW")
            if self.memory_counter >= N:
                self.end_t = self.node.timeline.now()
                self.node.timeline.stop()

    def get_fidelity(self) -> float:
        if self.memory_counter == 0:
            return 0
        else:
            return self.accumulated_fidelity / self.memory_counter
        
    def get_eg_probability(self) -> float:
        if hasattr(self.node, 'eg_attempts'):
            return self.memory_counter / self.node.eg_attempts
        else:
            return 0.0

class ResetApp:
    def __init__(self, node, other_node_name, target_fidelity=0.1):
        self.node = node
        self.node.set_app(self)
        self.other_node_name = other_node_name
        self.target_fidelity = target_fidelity

    def get_other_reservation(self, reservation):
        """called when receiving the request from the initiating node.

        For this application, we do not need to do anything.
        """

        pass

    def get_memory(self, info):
        """Similar to the get_memory method of the main application.

        We check if the memory info meets the request first,
        by noting the remote entangled memory and entanglement fidelity.
        We then free the memory for future use.
        """
        if (info.state == "ENTANGLED" and info.remote_node == self.other_node_name
                and info.fidelity > self.target_fidelity):
            self.node.resource_manager.update(None, info.memory, "RAW")

def set_parameters(topology: RouterNetTopo, config):
    # Get memory parameters
    MEMO_EXPIRE = float(config.get('Memory', 'coherence_time'))
    MEMO_EFFICIENCY = float(config.get('Memory', 'efficiency'))
    MEMO_FIDELITY = float(config.get('Memory', 'fidelity'))
    WAVE_LENGTH = float(config.get('Memory', 'wavelength'))

    # Set memory parameters
    for node in topology.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        memory_array = node.get_components_by_type("MemoryArray")[0]
        memory_array.update_memory_params("coherence_time", MEMO_EXPIRE)
        memory_array.update_memory_params("efficiency", MEMO_EFFICIENCY)
        memory_array.update_memory_params("raw_fidelity", MEMO_FIDELITY)
        memory_array.update_memory_params("wavelength", WAVE_LENGTH)

    # Get detector parameters
    DETECTOR_EFFICIENCY = float(config.get('Detector', 'efficiency'))

    # Set detector parameters
    for node in topology.get_nodes_by_type(RouterNetTopo.BSM_NODE):
        bsm = node.get_components_by_type("SingleAtomBSM")[0]
        bsm.update_detectors_params("efficiency", DETECTOR_EFFICIENCY)

    # Get entanglement swapping parameters
    SWAPPING_SUCCESS_RATE = float(config.get('Swapping', 'success_rate'))

    # Set entanglement swapping parameters
    for node in topology.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        node.network_manager.protocol_stack[1].set_swapping_success_rate(SWAPPING_SUCCESS_RATE)

def get_router_state(r, memo_index=None):
    if memo_index is None:
        memo_index = range(len(r.resource_manager.memory_manager))  # Default: all indexes
        print("====================")
        print("Node: ", r.name)
    print("{:6}\t{:15}\t{:9}\t{}".format("memo index", "remote_node","fidelity", "Eg Time"))
    for i, info in enumerate(r.resource_manager.memory_manager):
        if i in memo_index:
            
            print("{:6}\t{:15}\t{:9}\t{}".format(str(i), str(info.remote_node),
                                                  str(info.fidelity), str(info.entangle_time * 1e-12)))
    
def simulate(network_topology,config,link_capacity=None,swapping_order = None, target_fidelity = 0.1):
    if link_capacity is not None:
        Reservation.link_capacity = link_capacity
        ResourceReservationProtocol.create_rules = create_rules
    if swapping_order is not None: 
        Reservation.swapping_order = swapping_order
        ResourceReservationProtocol.create_rules = create_rules
    
    network_topo = RouterNetTopo(network_topology)
    #set the simulation parametters
    set_parameters(network_topo, config)
    tl = network_topo.get_timeline()
    tl.stop_time = 60e12
    tl.show_progress = False

    start_node_name = "Nodei"
    end_node_name = "Nodej"
    r3_name = "r3"
    node_names = network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
    for router in node_names:
        if router.name == start_node_name:
            node1 = router
        elif router.name == end_node_name:
            node2 = router
        ###
        elif router.name == r3_name:
            r3 = router
    tracker = trackApp(r3)
    ###
    app_node1 = EnranglementRequestApp(node1, end_node_name)
    reset_app = ResetApp(node2, start_node_name)

    start_time = 0.1e12
    end_time = 60e12

    tl.init()
    memory_number = 1
    app_node1.start(node2.name,start_time ,end_time , memory_number, target_fidelity)
    try:
        tl.run()
    except AssertionError as e:
        print(f"Assertion error: {e}")
        return 0.0, 0, 0.0
    print("rate app1: ", app_node1.get_throughput())
    rate = app_node1.get_throughput()
    fidelity = app_node1.get_fidelity()
    eg_probability = app_node1.get_eg_probability()
    
    return {'rate':rate, 'fidelity': fidelity}

def list_to_swapping_order(swap_tree):
    result = []
    for sublist in swap_tree:
        for element in sublist:
            result.append('r' + str(element))
    return result




config_path = os.path.join(os.path.dirname(__file__), 'utils', 'parameters.ini')
config = configparser.ConfigParser()
config.read(config_path)

topology_file = 'temp_topology.json'
C= np.array([1,1,1,1])
swapping_order = ['r1', 'r3', 'r2']
create_quantum_network(C,topology_file, config)
print(simulate(topology_file, config, C, swapping_order))


