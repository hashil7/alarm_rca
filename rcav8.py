import networkx as nx
import configparser
import mysql.connector
import logging
import time
import pickle
from pgmpy.inference import VariableElimination
from datetime import datetime
import math
import re
import torch
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
# Setup logging

class IsolationGNN(torch.nn.Module):
    def __init__(self, num_node_features=5, num_edge_features=2, hidden_channels=32, num_layers=18):
        """
        GNN model to predict node isolation in a ring network with failures.
        
        Args:
            num_node_features: Number of input node features (is_source, is_failed)
            num_edge_features: Number of edge features (is_failed)
            hidden_channels: Size of hidden representations
            num_layers: Number of message passing layers (should be large enough for 
                       messages to reach farthest nodes in the ring)
        """
        super(IsolationGNN, self).__init__()
        
        # Initial node feature transformation
        self.node_encoder = nn.Linear(num_node_features, hidden_channels)
        
        # Edge feature transformation
        self.edge_encoder = nn.Linear(num_edge_features, hidden_channels)
        
        # Message passing layers that can utilize edge features
        from torch_geometric.nn import GCNConv, EdgeConv, GATv2Conv, MessagePassing
        
        # Custom message passing layer that considers edge attributes
        class EdgeAwareConv(MessagePassing):
            def __init__(self, in_channels, out_channels):
                super(EdgeAwareConv, self).__init__(aggr='add')
                self.lin_node = nn.Linear(in_channels, out_channels)
                self.lin_edge = nn.Linear(in_channels, out_channels)
                self.lin_update = nn.Linear(in_channels + out_channels, out_channels)
            
            def forward(self, x, edge_index, edge_attr):
                # Transform edge attributes
                edge_embedding = self.lin_edge(edge_attr)
                
                # Propagate messages
                out = self.propagate(edge_index, x=x, edge_attr=edge_embedding)
                
                # Update node features
                out = self.lin_update(torch.cat([x, out], dim=1))
                return out
            
            def message(self, x_j, edge_attr):
                # Message is a function of source node and edge features
                # Specifically, if edge is failed (edge_attr=1), it will affect the message
                return x_j * (1 - edge_attr) + edge_attr * self.lin_node(x_j)
        
        # Create the message passing layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(EdgeAwareConv(hidden_channels, hidden_channels))
        
        # Final prediction layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Use only the first 2 features as specified
    
        
        # Initial encoding
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # Apply message passing layers
        for conv in self.conv_layers:
            x = F.relu(conv(x, edge_index, edge_attr))
        
        # Final prediction - whether node is isolated (1) or not (0)
        x = self.classifier(x)
        
        return torch.sigmoid(x).squeeze(-1)

global node_failure_model, link_failure_model, block_failure_model, ups_failure_model
node_failure_model = IsolationGNN(num_node_features=5, num_edge_features=2, hidden_channels=32, num_layers=6) 
link_failure_model = IsolationGNN(num_node_features=5, num_edge_features=2, hidden_channels=32, num_layers=6)
ups_failure_model = IsolationGNN(num_node_features=5, num_edge_features=2, hidden_channels=32, num_layers=6)
block_failure_model = IsolationGNN(num_node_features=5, num_edge_features=2, hidden_channels=32, num_layers=6)

# Load the state dictionaries with CPU mapping
device = torch.device('cpu')  # Force CPU usage
node_failure_state = torch.load('models/node_failure_model.pt', map_location=device)
link_failure_state = torch.load('models/link_failure_model.pt', map_location=device)
ups_failure_state = torch.load('models/ups_failure_model.pt', map_location=device)
block_failure_state = torch.load('models/block_failure_model.pt', map_location=device)

node_failure_model.load_state_dict(node_failure_state)
link_failure_model.load_state_dict(link_failure_state)
ups_failure_model.load_state_dict(ups_failure_state)
block_failure_model.load_state_dict(block_failure_state)

# Put the models in evaluation mode
node_failure_model.eval()
link_failure_model.eval()
ups_failure_model.eval()
block_failure_model.eval()


def setup_logging():
    """Configure logging to file and console"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:\n%(message)s\n',
        handlers=[
            logging.FileHandler("rcav8.log", mode="a"),
            #logging.StreamHandler()
        ]
    )

# Database setup
def setup_database():
    """Initialize database connection and create tables if needed"""
    config = configparser.ConfigParser()
    config.read("config.ini")
    db_config = {
        "host": config["mysql"]["host"],
        "user": config["mysql"]["user"],
        "password": config["mysql"]["password"],
        "database": config["mysql"]["database"]
    }
    mydb = mysql.connector.connect(**db_config)
    cursor = mydb.cursor(dictionary=True, buffered=True)
    
    # Create table if not exists
    create_table_query = """
    CREATE TABLE IF NOT EXISTS dummy_cam (
        id INT AUTO_INCREMENT PRIMARY KEY,
        rca_id VARCHAR(255),
        child_id TEXT,
        timestamp TEXT
    )
    """
    cursor.execute(create_table_query)
    mydb.commit()
    
    return mydb, cursor

# Load network topology
def load_topology(cursor):
    """Load and build network topology graph from database"""
    query_log_topo = """
        SELECT 
           aendname, 
           bendname, 
           aendip, 
           bendip, 
           block_name, 
           physicalringname, 
           lrname 
        FROM topology_data_logical
    """
    cursor.execute(query_log_topo)
    topology_logical = cursor.fetchall()
    
    # Create topology graph
    topology_graph = nx.Graph()
    for row in topology_logical:
        aend = row["aendip"]
        bend = row["bendip"]
        aendname = row["aendname"].upper()
        bendname = row["bendname"].upper()
        pr_name = row["physicalringname"].upper()
        lr_name = row["lrname"].upper()
        block_name = row["block_name"].upper()
         
        topology_graph.add_node(aend, name=aendname, lr_name=lr_name, pr_name=pr_name, block_name=block_name)
        topology_graph.add_node(bend, name=bendname, lr_name=lr_name, pr_name=pr_name, block_name=block_name)
        topology_graph.add_edge(aend, bend, lr_name=lr_name, pr_name=pr_name, block_name=block_name)
    
    logging.info("Topology Logical Graph created")
    return topology_graph

# Identify and mark block routers
def identify_block_routers(cursor, topology_graph):
    """Identify block routers and mark them in the graph"""
    block_router_query = """
        SELECT ip 
        FROM network_element 
        WHERE devicetype = 'BLOCK_ROUTER'
    """
    cursor.execute(block_router_query)
    block_routers = cursor.fetchall()
    
    # Create a set of block router IPs for faster lookups
    block_router_ips = {router['ip'] for router in block_routers}
    logging.info(f"Found {len(block_router_ips)} BLOCK_ROUTERs")
    
    return block_router_ips

# Remove inter-block connections
def remove_inter_block_connections(topology_graph, block_router_ips):
    """Remove nodes that create paths between blocks"""
    logging.info("Removing nodes that connect different blocks...")
    
    # Create a working copy of the graph for path analysis
    connection_graph = topology_graph.copy()
    
    # Track nodes to remove
    nodes_to_remove = set()
    
    # Iterative removal process
    prev_removal_count = -1
    current_removal_count = 0
    block_router_list = list(block_router_ips)
    
    while prev_removal_count != current_removal_count:
        prev_removal_count = current_removal_count
        
        # Create a temporary graph with current removals
        temp_graph = connection_graph.copy()
        temp_graph.remove_nodes_from(nodes_to_remove)
        
        # Check all pairs of block routers
        for i in range(len(block_router_list)):
            for j in range(i+1, len(block_router_list)):
                block1 = block_router_list[i]
                block2 = block_router_list[j]
                
                # Skip if either block router was removed
                if (block1 not in temp_graph.nodes or 
                    block2 not in temp_graph.nodes):
                    continue
                    
                # Find a path if one exists
                try:
                    if nx.has_path(temp_graph, block1, block2):
                        path = nx.shortest_path(temp_graph, block1, block2)
                        
                        # Add intermediate nodes to removal list
                        for node in path[1:-1]:
                            if node not in block_router_ips and node not in nodes_to_remove:
                                nodes_to_remove.add(node)
                                logging.info(f"Marking node {node} for removal (connects blocks {block1} and {block2})")
                except nx.NetworkXNoPath:
                    continue
        
        current_removal_count = len(nodes_to_remove)
        logging.info(f"Iteration complete: {current_removal_count} nodes marked for removal")
    
    # Remove nodes from the actual topology graph
    for node in nodes_to_remove:
        if topology_graph.has_node(node):
            topology_graph.remove_node(node)
    
    logging.info(f"Removed {len(nodes_to_remove)} nodes to completely isolate blocks")
    
    # Verify isolation
    components = list(nx.connected_components(topology_graph))
    logging.info(f"Topology now has {len(components)} disconnected components")
    
    return topology_graph

# Mark nodes with their block router
def assign_nodes_to_blocks(topology_graph, block_router_ips):
    """Assign each node to its block router"""
    for ip in block_router_ips:
        if not topology_graph.has_node(ip):
            logging.warning(f"BLOCK_ROUTER {ip} not found in topology graph")
        else:
            # Mark as block router
            topology_graph.nodes[ip]['is_block'] = True
            topology_graph.nodes[ip]['block_ip'] = ip
            
            try:
                # Get all reachable nodes using BFS
                reachable_nodes = list(nx.bfs_tree(topology_graph, ip))
                reachable_nodes.append(ip)  # Include the source node
                
                # Set block_ip for all reachable nodes
                assigned_count = 0
                for node in reachable_nodes:
                    topology_graph.nodes[node]['block_ip'] = ip
                    assigned_count += 1
                
                block_name = topology_graph.nodes[ip].get('block_name', 'Unknown')
                logging.info(f"Block router {ip} ({block_name}): {assigned_count} nodes assigned")
                
            except Exception as e:
                logging.error(f"Error assigning nodes to block router {ip}: {e}")
    
    return topology_graph
def get_ring_node_mapping(graph):
    """
    Create a mapping from (pr_name, lr_name) tuples to their subgraphs,
    ensuring the associated block router is included in each subgraph.

    Parameters:
    -----------
    graph : NetworkX Graph
        The network topology graph with pr_name, lr_name, and block_ip attributes.

    Returns:
    --------
    dict
        Dictionary mapping (pr_name, lr_name) tuples to NetworkX subgraphs.
    """
    logging.info("Starting get_ring_node_mapping...")
    # Temporary structure to hold nodes and the block_ip for each ring
    # Format: {(pr_name, lr_name): {'nodes': {node1, node2,...}, 'block_ip': block_ip_str}}
    ring_data = {}

    # Iterate through all nodes in the graph
    logging.info("Iterating through graph nodes to collect ring data...")
    node_count = 0
    skipped_nodes = 0
    for node, attrs in graph.nodes(data=True):
        if attrs.get('is_block', False):
            continue
        node_count += 1
        # Extract ring names and block IP
        pr_name = attrs.get('pr_name')
        lr_name_original = attrs.get('lr_name')
        block_ip = attrs.get('block_ip') # Get the block_ip assigned earlier

        # Normalize lr_name to extract the number
        lr_name_normalized = None
        if lr_name_original:
            # Enhanced regex to capture numbers after various separators or at the end
            number_match = re.search(r'(?:no\.?|[-\s]|RING-|RING\s*)(\d+)', lr_name_original, re.IGNORECASE)
            if not number_match: # Try finding number at the end if first regex failed
                 number_match = re.search(r'(\d+)$', lr_name_original)

            if number_match:
                lr_name_normalized = str(int(number_match.group(1)))
                # logging.debug(f"Node {node}: Original LR '{lr_name_original}' -> Normalized LR '{lr_name_normalized}'")
            # else:
                # logging.debug(f"Node {node}: Could not normalize LR name '{lr_name_original}'")


        # Skip nodes without required ring information
        if not pr_name or not lr_name_normalized:
            # logging.debug(f"Node {node}: Skipping due to missing PR ('{pr_name}') or normalized LR ('{lr_name_normalized}')")
            skipped_nodes += 1
            continue

        # Create tuple key with normalized ring names
        ring_key = (pr_name, lr_name_normalized)
        # logging.debug(f"Node {node}: Using ring key {ring_key}")


        # Initialize entry for this ring if it doesn't exist
        if ring_key not in ring_data:
            ring_data[ring_key] = {'nodes': set(), 'block_ip': None}
            # logging.debug(f"Initialized ring_data for key {ring_key}")


        # Add the current node to the set for this ring
        ring_data[ring_key]['nodes'].add(node)
        # logging.debug(f"Added node {node} to ring key {ring_key}. Current nodes: {ring_data[ring_key]['nodes']}")


        # Store the block_ip associated with this ring
        # Assuming all nodes in a logical ring belong to the same block
        if block_ip and ring_data[ring_key]['block_ip'] is None:
            ring_data[ring_key]['block_ip'] = block_ip
            # logging.debug(f"Assigned block_ip {block_ip} to ring key {ring_key}")

        elif block_ip and ring_data[ring_key]['block_ip'] != block_ip:
            # This case indicates a potential issue in block assignment logic
            logging.warning(f"Ring {ring_key} found associated with multiple block IPs:"
                            f" Existing: {ring_data[ring_key]['block_ip']}, New: {block_ip} (from node {node}). Using the first one found.")

    logging.info(f"Finished iterating through {node_count} nodes. Skipped {skipped_nodes} nodes without full ring info.")
    logging.info(f"Collected data for {len(ring_data)} unique ring keys.")

    # Now, ensure the block router node is added to each ring's node set
    # and create the final subgraphs
    logging.info("Creating subgraphs for each ring key...")
    ring_graphs = {}
    for ring_key, data in ring_data.items():
        # logging.debug(f"Processing ring key {ring_key} for subgraph creation.")
        nodes_for_subgraph = data['nodes'].copy() # Start with nodes found in the ring
        block_ip_for_ring = data['block_ip']
        # logging.debug(f"  Initial nodes for subgraph: {nodes_for_subgraph}")
        # logging.debug(f"  Block IP associated with this ring: {block_ip_for_ring}")


        # Add the block router node if it exists and isn't already included
        if block_ip_for_ring:
            if graph.has_node(block_ip_for_ring):
                if block_ip_for_ring not in nodes_for_subgraph:
                    nodes_for_subgraph.add(block_ip_for_ring)
                    # logging.debug(f"  Added block node {block_ip_for_ring} to subgraph node set.")
                # else:
                    # logging.debug(f"  Block node {block_ip_for_ring} was already in the node set.")

            else:
                logging.warning(f"Block IP {block_ip_for_ring} for ring {ring_key} not found in the main graph. Cannot add to subgraph.")
        # else:
            # logging.warning(f"No block IP was assigned to ring {ring_key} during node iteration.")


        # Create a subgraph containing all these nodes (ring nodes + block node) and their edges
        if nodes_for_subgraph: # Only create subgraph if there are nodes
            # logging.debug(f"  Final nodes for subgraph creation: {nodes_for_subgraph}")
            try:
                ring_graphs[ring_key] = graph.subgraph(nodes_for_subgraph).copy()
                # logging.debug(f"  Successfully created subgraph for {ring_key} with {ring_graphs[ring_key].number_of_nodes()} nodes and {ring_graphs[ring_key].number_of_edges()} edges.")
            except Exception as e:
                 logging.error(f"  Error creating subgraph for {ring_key} with nodes {nodes_for_subgraph}: {e}")

        else:
             logging.warning(f"No nodes found for ring {ring_key} after processing. Cannot create subgraph.")


    # Log summary
    logging.info(f"Finished creating subgraphs. Found {len(ring_graphs)} unique ring combinations with associated block routers.")
    for (pr, lr), subgraph in ring_graphs.items():
        # Retrieve block_ip from the original ring_data for logging consistency
        block_node_in_subgraph = ring_data.get((pr, lr), {}).get('block_ip', 'N/A')
        logging.info(f"Ring (PR:{pr}, LR:{lr}): {subgraph.number_of_nodes()} nodes (Block: {block_node_in_subgraph}), {subgraph.number_of_edges()} edges")

    logging.info("Exiting get_ring_node_mapping.")
    return ring_graphs

# ... rest of your code ...

# Add UPS devices
def add_ups_devices(cursor, topology_graph):
    """Add UPS devices as nodes connected to their routers"""
    logging.info("Adding UPS devices as nodes...")
    query_ups = """
        SELECT n1.ip as router_ip, n2.ip as ups_ip
        FROM network_element n1 
        JOIN network_element n2 ON n1.location = n2.location 
        WHERE n1.devicetype IN ("GP_ROUTER","BLOCK_ROUTER") 
        AND n2.devicetype IN ("GP_UPS","BLOCK_UPS")
    """
    cursor.execute(query_ups)
    router_ups_mappings = cursor.fetchall()
    
    ups_count = 0
    for mapping in router_ups_mappings:
        router_ip = mapping["router_ip"]
        ups_ip = mapping["ups_ip"]
        
        if not topology_graph.has_node(router_ip):
            logging.warning(f"Router {router_ip} not found in topology graph. Skipping UPS {ups_ip}")
            continue
            
        if not topology_graph.has_node(ups_ip):
            router_name = topology_graph.nodes[router_ip].get('name', f"Router-{router_ip}")
            ups_name = f"UPS-{router_name}"
            block_ip = topology_graph.nodes[router_ip].get('block_ip', None)
            topology_graph.add_node(ups_ip, name=ups_name, devicetype="UPS",block_ip=block_ip, is_ups=True)
            ups_count += 1
        
        topology_graph.add_edge(router_ip, ups_ip, edge_type="router_ups")
        logging.info(f"Added UPS {ups_ip} connected to router {router_ip}")
    
    logging.info(f"Added {ups_count} UPS devices as nodes to the topology graph")
    return topology_graph
def extract_endpoints_from_rfo(rfo):
    """
    Extract endpoint names from RFO string
    
    Parameters:
    -----------
    rfo : str
        Reason For Outage string (format: "... between NODE1 and NODE2")
        
    Returns:
    --------
    tuple or None
        (node1_name, node2_name) if parsing successful, None otherwise
    """
    if not rfo or "between" not in rfo or "and" not in rfo:
        return None
    
    try:
        # Extract node names from RFO (format: "Partial Fiber Cut Detected between RONJE and UPET")
        rfo_parts = rfo.split("between")
        if len(rfo_parts) <= 1:
            return None
            
        endpoint_parts = rfo_parts[1].split("and")
        if len(endpoint_parts) != 2:
            return None
            
        node1_name = endpoint_parts[0].strip().upper()
        node2_name = endpoint_parts[1].strip().upper()
        
        return (node1_name, node2_name)
        
    except Exception as e:
        logging.error(f"Error parsing RFO '{rfo}': {e}")
        return None
    
def log_ring_visualization(graph, ring_subgraph, block_node,
                           failed_link=None, failed_node=None,
                           affected_nodes=None, title=None):
    """
    Create circular ASCII visualization of a ring subgraph for logging,
    ordering main ring nodes sequentially, placing spurs radially,
    marking node status, and providing a detailed legend.
    """
    if not ring_subgraph or ring_subgraph.number_of_nodes() == 0:
        logging.info("No nodes in ring subgraph to visualize")
        return

    original_nodes = list(ring_subgraph.nodes())
    node_count = len(original_nodes)

    if node_count < 2:
        logging.info(f"Not enough nodes ({node_count}) in subgraph to visualize")
        return

    # --- Identify Main Ring Nodes and Spur Nodes ---
    main_ring_nodes_set = set()
    spur_connections = {} # spur_node -> connector_node
    node_degrees = dict(ring_subgraph.degree())

    for node, degree in node_degrees.items():
        if degree >= 2:
            main_ring_nodes_set.add(node)
        elif degree == 1:
            # Find the single neighbor (connector)
            connector = list(ring_subgraph.neighbors(node))[0]
            spur_connections[node] = connector
        else: # degree == 0 (isolated node in subgraph)
             logging.warning(f"Node {node} has degree 0 in the ring subgraph, ignoring for visualization.")

    # Validate spurs are connected to main ring
    valid_spur_connections = {}
    for spur, connector in spur_connections.items():
        if connector in main_ring_nodes_set:
            valid_spur_connections[spur] = connector
        else:
            # This spur is connected to another spur or an isolated node, which is unusual
            logging.warning(f"Spur node {spur} is connected to node {connector}, which is not part of the main ring (degree < 2). Visualization might be inaccurate.")
            # Treat the connector as part of the main ring for positioning purposes, though ordering might be off
            main_ring_nodes_set.add(connector) # Add connector to main ring set for positioning
            valid_spur_connections[spur] = connector # Keep the connection for drawing

    spur_nodes_set = set(valid_spur_connections.keys())
    main_ring_nodes_list = list(main_ring_nodes_set)
    main_ring_count = len(main_ring_nodes_list)

    if main_ring_count < 2 and node_count >= 2:
         logging.warning(f"Subgraph has {node_count} nodes but fewer than 2 main ring nodes (degree >= 2). Visualization may be linear or point-like.")
         # Fallback: treat all nodes as main ring for basic circular layout
         main_ring_nodes_list = original_nodes
         main_ring_nodes_set = set(original_nodes)
         valid_spur_connections = {}
         spur_nodes_set = set()
         main_ring_count = len(main_ring_nodes_list)


    # --- Order Main Ring Nodes Sequentially ---
    ordered_main_ring_nodes = []
    if block_node not in main_ring_nodes_set:
        # If block node is a spur or missing, start ordering from an arbitrary main ring node
        start_node = main_ring_nodes_list[0] if main_ring_nodes_list else None
        logging.warning(f"Block node {block_node} not in main ring nodes. Starting sequence from {start_node}.")
    else:
        start_node = block_node

    if start_node:
        try:
            ordered_main_ring_nodes = [start_node]
            current_node = start_node
            previous_node = None
            visited_main_ring = {start_node}

            while len(ordered_main_ring_nodes) < main_ring_count:
                found_next = False
                # Consider only neighbors that are also in the main ring set
                main_ring_neighbors = [n for n in ring_subgraph.neighbors(current_node) if n in main_ring_nodes_set]
                main_ring_neighbors.sort() # Deterministic order

                for neighbor in main_ring_neighbors:
                    if neighbor != previous_node:
                        if neighbor not in visited_main_ring:
                            ordered_main_ring_nodes.append(neighbor)
                            visited_main_ring.add(neighbor)
                            previous_node = current_node
                            current_node = neighbor
                            found_next = True
                            break
                        # Handle closing the loop
                        elif neighbor == start_node and len(ordered_main_ring_nodes) == main_ring_count -1:
                             pass # Loop condition terminates

                if not found_next and len(ordered_main_ring_nodes) < main_ring_count:
                    logging.warning(f"Could not find next sequential main ring node from {current_node}. Main ring might be disconnected or complex. Current order: {ordered_main_ring_nodes}")
                    # Add remaining unvisited main ring nodes
                    remaining_main = [n for n in main_ring_nodes_list if n not in visited_main_ring]
                    ordered_main_ring_nodes.extend(remaining_main)
                    break
        except Exception as e:
            logging.error(f"Error finding main ring sequence: {e}. Using arbitrary order.")
            ordered_main_ring_nodes = main_ring_nodes_list # Fallback
    else:
         logging.error("No start node found for main ring ordering.")
         ordered_main_ring_nodes = main_ring_nodes_list # Fallback


    # Combine ordered main ring and spurs for ID assignment and legend
    all_ordered_nodes = ordered_main_ring_nodes + sorted(list(spur_nodes_set))

    # --- Assign unique IDs (Block Node = '0') ---
    ids = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    node_ids = {}
    id_counter = 0

    # Assign '0' to block node if it exists
    if block_node in original_nodes: # Check original list as block might be spur
        node_ids[block_node] = ids[id_counter]
        id_counter += 1
    else:
        logging.warning("Block node was not in the original node list for ID assignment.")

    # Assign IDs to remaining nodes (main ring first, then spurs)
    for node in all_ordered_nodes:
        if node not in node_ids: # Assign if not already assigned (i.e., not block node)
            if id_counter >= len(ids):
                logging.warning(f"Ran out of unique IDs ({len(ids)}). Reusing symbols.")
                node_ids[node] = ids[id_counter % len(ids)]
            else:
                node_ids[node] = ids[id_counter]
            id_counter += 1

    id_to_node = {v: k for k, v in node_ids.items()}

    # --- Prepare Canvas ---
    radius = 10
    spur_offset = 3 # How far spurs project radially
    padding = 3 + spur_offset # Increase padding to accommodate spurs
    center_x = center_y = radius + padding
    total_height = total_width = 2 * (radius + padding) + 1
    canvas = [[' ' for _ in range(total_width)] for _ in range(total_height)]

    # --- Compute Node Positions ---
    node_positions = {}
    start_angle_offset = math.pi / 2 # Top

    # 1. Position Main Ring Nodes
    if main_ring_count > 0:
        for i, node in enumerate(ordered_main_ring_nodes):
            # Prevent division by zero if only one main ring node
            angle_step = (2 * math.pi / main_ring_count) if main_ring_count > 1 else 0
            angle = start_angle_offset - (angle_step * i) # Clockwise
            x = int(round(center_x + radius * math.cos(angle)))
            y = int(round(center_y - radius * math.sin(angle)))
            x = max(0, min(total_width - 1, x))
            y = max(0, min(total_height - 1, y))
            node_positions[node] = (x, y)

    # 2. Position Spur Nodes
    for spur_node, connector_node in valid_spur_connections.items():
        if connector_node in node_positions:
            conn_x, conn_y = node_positions[connector_node]
            # Calculate angle from center to connector
            delta_x = conn_x - center_x
            delta_y = conn_y - center_y
            # Use atan2 for correct angle in all quadrants; negate y because canvas y increases downwards
            angle_to_connector = math.atan2(-delta_y, delta_x)

            # Calculate spur position radially outward
            spur_x = int(round(conn_x + spur_offset * math.cos(angle_to_connector)))
            spur_y = int(round(conn_y - spur_offset * math.sin(angle_to_connector))) # Subtract sin because canvas y increases downwards
            spur_x = max(0, min(total_width - 1, spur_x))
            spur_y = max(0, min(total_height - 1, spur_y))

            # Avoid placing spur directly on top of connector if offset is small/zero
            if (spur_x, spur_y) == (conn_x, conn_y):
                 # Add a minimal nudge, direction might not be perfectly radial but avoids overlap
                 spur_x += 1 if delta_x >= 0 else -1
                 spur_y += 1 if delta_y >= 0 else -1 # Nudge based on quadrant relative to center
                 spur_x = max(0, min(total_width - 1, spur_x))
                 spur_y = max(0, min(total_height - 1, spur_y))

            node_positions[spur_node] = (spur_x, spur_y)
        else:
            logging.warning(f"Connector node {connector_node} for spur {spur_node} has no position. Placing spur at default location (0,0).")
            node_positions[spur_node] = (0, 0) # Fallback

    # --- Draw Edges ---
    logging.debug("--- Drawing Edges ---")
    for u, v in ring_subgraph.edges():
        # Check if both nodes have positions calculated
        if u not in node_positions or v not in node_positions:
            logging.debug(f"Skipping edge ({u}, {v}) - node position missing.")
            continue

        logging.debug(f"Processing edge ({u}, {v})")
        is_current_failed_link = (failed_link and
                                  ((u == failed_link[0] and v == failed_link[1]) or
                                   (u == failed_link[1] and v == failed_link[0])))
        # Check main graph for previous failure status
        is_previous_failed_link = graph.has_edge(u,v) and graph.edges[u, v].get('previous_failed', False)

        link_symbol = '-'
        if is_previous_failed_link: link_symbol = 'p'
        if is_current_failed_link: link_symbol = 'x'
        logging.debug(f"  Edge ({u}, {v}): is_current={is_current_failed_link}, is_previous={is_previous_failed_link} -> Symbol='{link_symbol}'")

        x1, y1 = node_positions[u]
        x2, y2 = node_positions[v]
        dx = x2 - x1
        dy = y2 - y1
        steps = max(abs(dx), abs(dy))

        if steps > 0:
            x_incr, y_incr = dx / steps, dy / steps
            x, y = float(x1), float(y1)
            for k in range(int(steps) + 1):
                int_x, int_y = int(round(x)), int(round(y))
                if (0 <= int_y < total_height and 0 <= int_x < total_width):
                    current_char = canvas[int_y][int_x]
                    # Check if this point is the exact location of *any* node
                    is_node_location = False
                    for node_ip, node_pos in node_positions.items():
                        if node_pos == (int_x, int_y): is_node_location = True; break

                    if not is_node_location: # Only draw if not exactly on a node's center
                        if link_symbol == 'x': canvas[int_y][int_x] = 'x'
                        elif link_symbol == 'p' and current_char != 'x': canvas[int_y][int_x] = 'p'
                        elif link_symbol == '-' and current_char == ' ': canvas[int_y][int_x] = '-'
                if k < steps: x += x_incr; y += y_incr
        else:
             logging.debug(f"  Skipping drawing edge ({u}, {v}) - zero steps (nodes might be at same position).")
    logging.debug("--- Finished Drawing Edges ---")

    # --- Draw Nodes (using Status Symbols) ---
    # Draw nodes *after* edges so they overwrite edge paths at their location
    for node, (x, y) in node_positions.items():
         if 0 <= y < total_height and 0 <= x < total_width:
             node_symbol = node_ids.get(node, '?') # Default to '?' if ID somehow missing
             node_attrs = graph.nodes.get(node, {})
             is_prev_failed = node_attrs.get('previous_failed', False)
             is_affected = affected_nodes and node in affected_nodes

             if node == failed_node: node_symbol = 'X'
             elif is_prev_failed: node_symbol = 'P'
             elif node == block_node: node_symbol = 'B'
             elif is_affected: node_symbol = '!'
             # else: keep node_symbol as assigned ID

             canvas[y][x] = node_symbol
         else:
              logging.warning(f"Node {node} position ({x},{y}) outside canvas bounds during drawing.")

    # --- Build ASCII Output String ---
    frame_width = total_width + 4
    vis = ["\n+" + "-" * (frame_width - 2) + "+"]
    vis.append("| " + f"{'RING VISUALIZATION':^{frame_width - 4}}" + " |")
    if title: vis.append("| " + f"{title:<{frame_width - 4}}" + " |")
    vis.append("| " + f"{'Nodes: ' + str(node_count):<{frame_width - 4}}" + " |")
    vis.append("+" + "-" * (frame_width - 2) + "+")
    for row in canvas: vis.append("|  " + ''.join(row) + "  |")
    vis.append("+" + "-" * (frame_width - 2) + "+")

    # --- Plot Legend ---
    plot_legend_width = frame_width
    vis.append("|" + f"{'Plot Legend':^{plot_legend_width - 4}}" + "|")
    vis.append("|" + "-" * (plot_legend_width - 4) + "|")
    vis.append("| " + f"{'X = Current Failed Node':<{plot_legend_width // 2 - 3}}" + f"{'P = Previously Failed Node':<{plot_legend_width // 2 - 3}}" + "|")
    vis.append("| " + f"{'B = Block Node':<{plot_legend_width // 2 - 3}}" + f"{'! = Affected Node':<{plot_legend_width // 2 - 3}}" + "|")
    vis.append("| " + f"{'0-9, A-Z = Normal Node ID':<{plot_legend_width - 4}}" + "|")
    vis.append("| " + f"{'x = Current Failed Edge':<{plot_legend_width // 2 - 3}}" + f"{'p = Previously Failed Edge':<{plot_legend_width // 2 - 3}}" + "|")
    vis.append("| " + f"{'- = Normal Edge':<{plot_legend_width - 4}}" + "|")
    vis.append("+" + "-" * (plot_legend_width - 2) + "+")

    # --- Detailed Node Table Legend ---
    id_w, ip_w, name_w, role_w, status_w, degree_w, prev_fail_w = 3, 15, 18, 6, 9, 6, 18
    total_legend_width = id_w + ip_w + name_w + role_w + status_w + degree_w + prev_fail_w + 15

    vis.append("+" + "-" * (total_legend_width - 2) + "+")
    vis.append("|" + f"{'Node Details (ID refers to Plot Legend for Normal Nodes)':^{total_legend_width - 2}}" + "|")
    vis.append("+" + "=" * (total_legend_width - 2) + "+")
    vis.append(f"| {'ID':<{id_w}} | {'IP Address':<{ip_w}} | {'Name':<{name_w}} | {'Role':<{role_w}} | {'Status':<{status_w}} | {'Degree':<{degree_w}} | {'Prev. Failed':<{prev_fail_w}} |")
    vis.append("+" + "-" * (total_legend_width - 2) + "+")

    # Use all_ordered_nodes for the table rows
    for node in all_ordered_nodes:
        node_id_char = node_ids.get(node, '?')
        node_attrs = graph.nodes.get(node, {})
        name = node_attrs.get('name', 'Unknown')[:name_w]
        role = "BLOCK" if node == block_node else "RING" # Role based on block_node, not degree
        status = "OK"
        if affected_nodes and node in affected_nodes: status = "AFFECTED"
        if node == failed_node: status = "CURR_FAIL"

        # Get degree from the original subgraph degrees
        degree = node_degrees.get(node, 0)

        prev_fail_str = ""
        if node_attrs.get('previous_failed', False): prev_fail_str = "Node "
        prev_fail_edges = []
        # Check neighbors in the original subgraph
        if node in ring_subgraph:
             for neighbor in ring_subgraph.neighbors(node):
                 if graph.has_edge(node, neighbor) and graph.edges[node, neighbor].get('previous_failed', False):
                     if neighbor in node_ids: prev_fail_edges.append(node_ids[neighbor])
        if prev_fail_edges:
            prev_fail_str += "Edge(s) to: " + ",".join(sorted(prev_fail_edges))
        if not prev_fail_str: prev_fail_str = "No"

        vis.append(f"| {node_id_char:<{id_w}} | {node:<{ip_w}} | {name:<{name_w}} | {role:<{role_w}} | {status:<{status_w}} | {degree:<{degree_w}} | {prev_fail_str[:prev_fail_w]:<{prev_fail_w}} |")

    vis.append("+" + "-" * (total_legend_width - 2) + "+")

    # Log the visualization
    logging.info("\n".join(vis))


# Helper functions for alarm processing
def process_link_down(graph, id, ip, ne_time, rfo,root_cause_mapping, ring_nodes,logged_isolated):
    """Process a link_down alarm by identifying the edge to remove from RFO"""
    if not graph.has_node(ip):
        logging.warning(f"Link down alarm for non-existent node {ip}")
        return None
    if graph.nodes[ip].get('is_ups', False):
        logging.warning(f"Node {ip} is an UPS device, skipping link_down processing")
        return None
    
    # Mark the node as having received a link down alarm
    graph.nodes[ip]['link_down_received'] = True
    graph.nodes[ip]['link_down_ne_time'] = ne_time
    logging.info(f"Node {ip} marked with link_down at time {ne_time}")
    
    # Get router neighbors (excluding UPS nodes)
    router_neighbors = [n for n in graph.neighbors(ip) 
                      if not graph.nodes[n].get('is_ups', False)]
    
    # Check if router neighbors have link_down_received
    neighbors_with_link_down = []
    
    for neighbor in router_neighbors:
        if graph.nodes[neighbor].get('link_down_received', False):
            # Calculate time difference
            neighbor_time = graph.nodes[neighbor]['link_down_ne_time']
            
            # Convert to datetime if stored as string

            
            time_diff = abs((neighbor_time - ne_time).total_seconds())
            neighbors_with_link_down.append((neighbor, time_diff))
    
    # If we have neighbors with link_down, process them
    if neighbors_with_link_down:
        # Sort by time difference (smallest first)
        neighbors_with_link_down.sort(key=lambda x: x[1])
        
        # Get neighbor with smallest time difference
        closest_neighbor, time_diff = neighbors_with_link_down[0]
        
        if time_diff > 60:
            logging.warning(f"Time difference between link_down alarms is too large: {time_diff}s")
            return None
        if graph.has_edge(ip, closest_neighbor):

            graph.edges[ip, closest_neighbor]['current_failed'] = True
            graph.edges[ip, closest_neighbor]['alarm_id'] = id
            logging.info(f"Edge removed between {ip} and {closest_neighbor} - Time diff: {time_diff}s")
            block_ip = graph.nodes[ip].get('block_ip')
            isolated_nodes = None
            if block_ip:
                nodes = nx.node_connected_component(graph,block_ip)
                subgraph = graph.subgraph(nodes)
                isolated_nodes = get_isolated_nodes(subgraph,logged_isolated,link_failure_model)
                logging.info(f"Isolated nodes: {isolated_nodes}")
                graph.edges[ip, closest_neighbor]['current_failed'] = False
                graph.edges[ip, closest_neighbor]['previous_failed'] = True
                if isolated_nodes:
                    root_cause_mapping[id] = isolated_nodes
                
                
                    arbitrary_isolated_node = None
                    
                    for node in isolated_nodes:
                        if not graph.nodes[node].get('is_block', False):
                            arbitrary_isolated_node = node
                            logging.info(f"  Using NON-BLOCK node {arbitrary_isolated_node} from isolated set to determine ring.")
                            break 
                    if arbitrary_isolated_node is not None:
                        lr_name = graph.nodes[arbitrary_isolated_node].get('lr_name', 'Unknown')
                        logging.info(f"LR Name: {lr_name}")
                        if lr_name:
        # Use regex to find the number in various ring naming formats
                            number_match = re.search(r'(?:no\.?|[-\s]|RING-|RING\s*)(\d+)', lr_name, re.IGNORECASE)
                            if not number_match: # Try finding number at the end if first regex failed
                                number_match = re.search(r'(\d+)$', lr_name)
                            if number_match:
                                # Convert to integer to remove leading zeros
                                lr_name = str(int(number_match.group(1)))

                        pr_name = graph.nodes[arbitrary_isolated_node].get('pr_name', 'Unknown')
                        logging.info(f"PR Name: {pr_name}")
                        block_node = graph.nodes[ip].get('block_ip', 'Unknown')
                        nodes_in_ring = ring_nodes.get((pr_name, lr_name), set())
                        if nodes_in_ring:
                            title = f"Ring: PR {pr_name}, LR {lr_name}"
                            log_ring_visualization(graph, nodes_in_ring,block_node, failed_link=(ip, closest_neighbor),
                                                failed_node=None, affected_nodes=isolated_nodes,
                                                title=title)
                    
                for node in isolated_nodes:
                    graph.nodes[node]['has_link_down'] = id
                    logging.info(f"Node {node} marked with has_link_down: {id}")
            
            return (ip, closest_neighbor)
        else:        

            logging.warning(f"No valid edge to remove from graph")
            return None

def process_node_down(graph, ip, id, ne_time,root_cause_mapping,ring_nodes,logged_isolated):
    """Process a node_down alarm by removing the node"""
    if graph.nodes[ip].get('is_ups', False):
        logging.warning(f"Node {ip} is an UPS device, skipping node_down processing")
        return None
        

    block_failed = graph.nodes[ip].get('is_block', False)
    
    graph.nodes[ip]['current_failed'] = True
    graph.nodes[ip]['alarm_id'] = id
    logging.info(f"Node {ip} marked as failed in the graph")
    block_ip = graph.nodes[ip].get('block_ip')
    isolated_nodes = None
    if block_ip:
        subgraph = graph.subgraph(nx.node_connected_component(graph,block_ip))
        if block_failed:
            isolated_nodes = get_isolated_nodes(subgraph, logged_isolated, block_failure_model)
        else:
            isolated_nodes = get_isolated_nodes(subgraph,logged_isolated,node_failure_model)
            logging.info(f"Isolated nodes: {isolated_nodes}")

            if isolated_nodes:
                    root_cause_mapping[id] = isolated_nodes
                    if not graph.nodes[ip].get('is_block', False):
                        arbitrary_isolated_node = None
                        
                        for node in isolated_nodes:
                            if not graph.nodes[node].get('is_block', False):
                                arbitrary_isolated_node = node
                                logging.info(f"  Using NON-BLOCK node {arbitrary_isolated_node} from isolated set to determine ring.")
                                break 
                        if arbitrary_isolated_node is not None:
                            lr_name = graph.nodes[arbitrary_isolated_node].get('lr_name', 'Unknown')
                            logging.info(f"LR Name: {lr_name}")
                            if lr_name:
                        # Use regex to find the number in various ring naming formats
                                number_match = re.search(r'(?:no\.?|[-\s]|RING-|RING\s*)(\d+)', lr_name, re.IGNORECASE)
                                if not number_match: # Try finding number at the end if first regex failed
                                    number_match = re.search(r'(\d+)$', lr_name)
                                if number_match:
                                    # Convert to integer to remove leading zeros
                                    lr_name = str(int(number_match.group(1)))

                            pr_name = graph.nodes[arbitrary_isolated_node].get('pr_name', 'Unknown')
                            logging.info(f"PR Name: {pr_name}")
                            block_node = graph.nodes[ip].get('block_ip', 'Unknown')
                            nodes_in_ring = ring_nodes.get((pr_name, lr_name), set())
                            if nodes_in_ring:
                                title = f"Ring: PR {pr_name}, LR {lr_name}"
                                log_ring_visualization(graph, nodes_in_ring,block_node, failed_link=None,
                                                    failed_node=ip, affected_nodes=isolated_nodes,
                                                    title=title)
            
            graph.nodes[ip]['current_failed'] = False
            graph.nodes[ip]['previous_failed'] = True

            for node in isolated_nodes:

                if block_failed:
                    graph.nodes[node]['block_failed'] = id
                else:
                    graph.nodes[node]['has_link_down'] = id
                logging.info(f"Node {node} marked with has_node_down: {id}")
                logging.info(f"Node {node} marked with block_failed: {block_failed}")

    return None

def process_ups_low_battery(graph, ip, ne_time, id):
    """Process a UPS low battery alarm"""
    if not graph.has_node(ip):
        logging.warning(f"UPS low battery alarm for non-existent node {ip}")
        return None
        
    graph.nodes[ip]['ups_low_battery_alarm_received'] = True
    graph.nodes[ip]['ups_low_battery_ne_time'] = ne_time
    logging.info(f"Node {ip} marked with UPS low battery alarm at time {ne_time}")
    block_ip = graph.nodes[ip].get('block_ip')
    subgraph = graph.subgraph(nx.node_connected_component(graph, block_ip))
    
    isolated_nodes = get_isolated_nodes(subgraph,logged_isolated,ups_failure_model)
    for node in isolated_nodes:
        graph.nodes[node]['has_ups_low_battery'] = id
        logging.info(f"Router {node} marked with UPS low battery alarm")


def custom_bfs(graph, source):
    """
    Custom BFS traversal that doesn't traverse through failed nodes or edges
    
    Parameters:
    -----------
    graph : NetworkX Graph
        The graph to traverse
    source : node
        The source node to start traversal from
        
    Returns:
    --------
    set
        Set of nodes reachable from source
    """
    if source not in graph or graph.nodes[source].get('failed', False):
        return set()
        
    visited = {source}
    queue = [source]
    
    while queue:
        current = queue.pop(0)
        
        for neighbor in graph.neighbors(current):
            # Skip failed nodes
            if graph.nodes[neighbor].get('failed', False) or graph.nodes[neighbor].get('is_ups', False):
                continue
                
            # Skip failed edges
            if graph.edges[current, neighbor].get('failed', False):
                continue
                
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                
    return visited

def custom_bfs_without_ups(graph, source):
    """
    Custom BFS traversal that doesn't traverse through failed nodes or edges
    
    Parameters:
    -----------
    graph : NetworkX Graph
        The graph to traverse
    source : node
        The source node to start traversal from
        
    Returns:
    --------
    set
        Set of nodes reachable from source
    """
    if source not in graph:
        return set()
        
    visited = {source}
    queue = [source]
    
    while queue:
        current = queue.pop(0)
        
        for neighbor in graph.neighbors(current):
            # Skip failed nodes
            if graph.nodes[neighbor].get('is_ups', False):
                continue
                
            # Skip failed edges

                
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                
    return visited

def get_isolated_nodes(subgraph, logged_isolated=None, model=None):
    """
    Use GNN model to predict isolated nodes from a subgraph
    
    Args:
        subgraph: NetworkX graph with failures marked in attributes
        logged_isolated: Set of nodes already known to be isolated
        model: GNN model to predict isolation
    
    Returns:
        set: All isolated nodes including newly identified ones
    """
    if model is None:
        logging.warning("No model provided, cannot predict isolated nodes")
        return logged_isolated or set()
    
    # Prepare input for the model
    nodes = list(subgraph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Node features: [is_source, is_failed, is_ups]
    x = torch.zeros((len(nodes), 5), dtype=torch.float)
    
    for i, node in enumerate(nodes):
        # Is source/block node
        x[i, 0] = 1.0 if subgraph.nodes[node].get('is_block', False) else 0.0
        # Is failed node
        x[i, 1] = 1.0 if subgraph.nodes[node].get('current_failed', False) else 0.0

        x[i,2] = 1.0 if subgraph.nodes[node].get('previous_failed', False) else 0.0
        # Is UPS
        x[i, 3] = 1.0 if subgraph.nodes[node].get('is_ups', False) else 0.0

        x[i, 4] = 1.0 if subgraph.nodes[node].get('ups_low_battery_alarm_received', False) else 0.0


    
    # Edge index and attributes
    edge_index = []
    edge_attr = []
    
    for u, v in subgraph.edges():
        # Add both directions for undirected graph
        edge_index.append([node_to_idx[u], node_to_idx[v]])
        edge_index.append([node_to_idx[v], node_to_idx[u]])
        
        # Edge attributes - include failed status
        current_failed = 1.0 if subgraph.edges[u, v].get('current_failed', False) else 0.0
        previous_failed = 1.0 if subgraph.edges[u, v].get('previous_failed', False) else 0.0
        edge_attr.append([current_failed, previous_failed])
        edge_attr.append([current_failed, previous_failed])  # For undirected graph
    
    # Convert to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(nodes)
    )
    
    # Move to device if model is on GPU
    device = next(model.parameters()).device
    data = data.to(device)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        pred = model(data)
        
    # Convert predictions to isolated nodes
    predicted_isolated_idx = (pred > 0.5).nonzero(as_tuple=True)[0].cpu().numpy()
    predicted_isolated = {nodes[idx] for idx in predicted_isolated_idx 
                         if not subgraph.nodes[nodes[idx]].get('is_ups', False)}
    
    # Add newly found isolated nodes to logged_isolated

    
    # If no logged_isolated provided, return all predicted isolated nodes
    return predicted_isolated

def predict_root_cause(model, attrs):
    """Predict root cause using Bayesian model"""
    inference = VariableElimination(model)
    if 'has_ups_low_battery' in attrs:
        raw_ups_low_battery = attrs['has_ups_low_battery']
        logging.info(f"  Raw attribute 'has_ups_low_battery': {raw_ups_low_battery} (Type: {type(raw_ups_low_battery).__name__})")
    else:
        raw_ups_low_battery = False # Default if key is missing

    if 'has_link_down' in attrs:
        raw_link_down = attrs['has_link_down']
        logging.info(f"  Raw attribute 'has_link_down': {raw_link_down} (Type: {type(raw_link_down).__name__})")
    else:
        raw_link_down = False # Default if key is missing

    if 'block_failed' in attrs:
        raw_block_failed = attrs['block_failed']
        logging.info(f"  Raw attribute 'block_failed': {raw_block_failed} (Type: {type(raw_block_failed).__name__})")
    else:
        raw_block_failed = False 

    # Convert to boolean for evidence
    has_ups_low_battery = bool(raw_ups_low_battery)
    has_link_down = bool(raw_link_down)
    block_failed = bool(raw_block_failed)
    
    evidence = {
        'has_ups_low_battery': has_ups_low_battery,
        'has_link_down': has_link_down,
        'block_failed': block_failed
    }
    logging.info(f"Evidence for RCA: {evidence}")
    
    try:
        # Query the model
        result = inference.query(variables=['rca'], evidence=evidence)
        
        # Get probability distribution
        prob_dist = {state: float(prob) for state, prob in 
                    zip(result.state_names['rca'], result.values)}
        
        # Sort probabilities from highest to lowest
        sorted_probs = sorted(prob_dist.items(), key=lambda x: x[1], reverse=True)
        
        # Format log message
        log_message = ["RCA Probability Distribution:"]
        log_message.append("-" * 50)
        log_message.append(f"{'Root Cause':<20} | {'Probability':<15} | {'Rank'}")
        log_message.append("-" * 50)
        
        for i, (cause, prob) in enumerate(sorted_probs):
            log_message.append(f"{cause:<20} | {prob:.6f} | {i+1}")
        
        log_message.append("-" * 50)
        logging.info("\n".join(log_message))
        
        # Find most probable RCA (first item in sorted list)
        most_probable_rca = sorted_probs[0][0]
        logging.info(f"Selected RCA: {most_probable_rca} with probability {sorted_probs[0][1]:.6f}")
        
        rca_id = attrs.get(most_probable_rca, None)
        if rca_id:
            logging.info(f"RCA ID from node: {rca_id}")
        return rca_id
        
    except Exception as e:
        logging.error(f"Error in RCA prediction: {e}")
        return None

# Process start events for different alarm types
def process_start_event(current_graph, alarm, removed_links, model, cursor, mydb,root_cause_mapping, ring_nodes,logged_isolated):
    """Process a start event (new alarm)"""
    logging.info("="*80)
    global matched, total
    total+=1
    alarm_id = alarm['ID']
    prob_cause = alarm['PROB_CAUSE']
    ip = alarm['OBJ_NAME']
    event_time = alarm['NE_TIME']
    rfo = alarm['RFO']
    rca_from_algo = alarm.get('RCA_ID')
    logging.info(f"Processing alarm {alarm_id} for {ip} at {event_time} with prob_cause {prob_cause}")
    # Skip if node doesn't exist in original topology (except for UPS)
    if ip not in topology_logical_graph.nodes and prob_cause != 'ups_low_battery':
        logging.warning(f"Node {ip} not found in topology graph, skipping")
        return
    rca = None
    logging.info(f"RCA from existing algorithm: {rca_from_algo}")
    if prob_cause != 'link_down':
        rca=find_rca(current_graph,ip, alarm_id, model, cursor, mydb)
    if rca:
        if rca == rca_from_algo:
            matched+=1
            logging.info(f"RCA from algorithm matches predicted RCA: {rca}")
        elif rca_from_algo is None:
            logging.info(f"RCA from algorithm is None, but predicted RCA exists")
        else:
            logging.warning(f"RCA from algorithm does not match predicted RCA")
        return
    if rca_from_algo is None:
        matched+=1
    else:
        logging.warning(f"Predicted RCA is None, but algorithm RCA exists")
    # Process based on alarm type
    if prob_cause == 'link_down':
        failed_link = process_link_down(current_graph, alarm_id, ip, event_time,rfo, root_cause_mapping,ring_nodes,logged_isolated)
        if failed_link:
            removed_links[alarm_id] = failed_link
    
    elif prob_cause == 'node_not_reachable':
        if current_graph.has_node(ip):
            # Save node attributes before removing

            
            process_node_down(current_graph, ip, alarm_id, event_time,root_cause_mapping,ring_nodes,logged_isolated)
    
    elif prob_cause == 'ups_low_battery':
        process_ups_low_battery(current_graph, ip, event_time, alarm_id)
        
    
    # After each alarm, look for isolated nodes that need RCA
    

# Process clear events for different alarm types


def find_rca(current_graph, ip, alarm_id, model, cursor, mydb):
    """Check if the current node needs RCA for the specific alarm"""
    if not current_graph.has_node(ip):
        logging.warning(f"Cannot check RCA for non-existent node {ip}")
        return

    root_cause_id = predict_root_cause(model, current_graph.nodes[ip])
    
    if root_cause_id:
        current_graph.nodes[ip]['rca_done'] = True
        current_graph.nodes[ip]['root_cause_id'] = root_cause_id
        
        # Store RCA result in database
        insert_query = """
            INSERT INTO dummy_cam (rca_id, child_id, timestamp) 
            VALUES (%s, %s, %s)
        """
        cursor.execute(insert_query, (root_cause_id, alarm_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        mydb.commit()
        logging.info(f"RCA recorded: Node {ip} with root cause {root_cause_id}")
        return root_cause_id
    return None

def main():
    """Main function that orchestrates the RCA process"""
    try:
        
        setup_logging()
        mydb, cursor = setup_database()
        
        # Load network topology
        global topology_logical_graph  # Declare we're using the global variable
        topology_logical_graph = load_topology(cursor)
        
        # Identify block routers
        block_router_ips = identify_block_routers(cursor, topology_logical_graph)
        
        # Isolate blocks by removing inter-block connections
        topology_logical_graph = remove_inter_block_connections(topology_logical_graph, block_router_ips)
        
        # Assign nodes to their blocks
        topology_logical_graph = assign_nodes_to_blocks(topology_logical_graph, block_router_ips)
        
        ring_nodes = get_ring_node_mapping(topology_logical_graph)
        # Add UPS devices
        topology_logical_graph = add_ups_devices(cursor, topology_logical_graph)
        
        global logged_isolated
        logged_isolated = set()


        # Create working copy of the graph
 # {alarm_id: (source_ip, target_ip)}
        # Load Bayesian model
        with open('bayesian_rca_model.pkl', 'rb') as f:
            model = pickle.load(f)
        global matched, total
        
        while(True):
            try:
                logging.info("Starting new cycle of alarm processing at " + str(datetime.now()))
                truncate_query = "DELETE FROM dummy_cam"
                cursor.execute(truncate_query)
                mydb.commit()
                current_graph = topology_logical_graph.copy()

                root_cause_mapping = {}
                removed_links = {} 
                query = """SELECT * FROM alarm ORDER BY NE_TIME ASC, CASE OBJ_TYPE WHEN 'BLOCK_ROUTER' THEN 1 ELSE 2 END ASC,
                CASE PROB_CAUSE WHEN 'ups_low_battery' THEN 1 ELSE 2 END ASC"""
                
                cursor.execute(query)
                alarms = cursor.fetchall()
                mydb.commit()
        
        # Create timeline of events

        
        # Sort events chronologically

        
        # Process each event
                for alarm in alarms:
                    process_start_event(current_graph, alarm, removed_links, model, cursor, mydb,root_cause_mapping, ring_nodes,logged_isolated)


                logging.info(f"Finished Processing current alarms at {datetime.now()}")
                logging.info(f"Matched RCA_IDs: {matched} out of {total} ({(matched/total)*100:.2f}%)")
                time.sleep(360)
            except Exception as e:
                logging.error(f"Error in while loop: {e}", exc_info=True)
                time.sleep(60)
                continue
    except Exception as e:
        logging.error(f"Error in main function: {e}", exc_info=True)
def process_clear_event(current_graph, alarm, removed_links, topology_graph,root_cause_mapping):
    """Process a clear event (alarm cleared)"""
    alarm_id = alarm['ID']
    ip = alarm['OBJ_NAME']
    prob_cause = alarm['PROB_CAUSE']

    
    # Restore links
    if prob_cause == 'link_down' and alarm_id in removed_links:
        source, target = removed_links[alarm_id]
        logging.info(f"Clearing attributes for edge between {source} and {target} for alarm {alarm_id}")
        
        if 'failed' in current_graph.edges[source, target]:
            current_graph.edges[source, target].pop('failed', None)
        if 'alarm_id' in current_graph.edges[source, target]:
            current_graph.edges[source, target].pop('alarm_id', None)
                        # Remove link_down flags from nodes that were affected by this link
        for node in root_cause_mapping[alarm_id] if alarm_id in root_cause_mapping else []:
            if current_graph.nodes[node].get('has_link_down') == alarm_id:
                current_graph.nodes[node].pop('has_link_down', None)
                logging.info(f"Cleared has_link_down flag from node {node}")
        
        # Remove from our tracking dict
        del removed_links[alarm_id]
    
    # Restore nodes
    elif prob_cause == 'node_down':
        
        logging.info(f"Clearing failed atttributes for node {ip}")
        current_graph.nodes['ip'].pop('failed', None)
        current_graph.nodes['ip'].pop('alarm_id', None)
        
        # Add the node back with its attributes
        
        
        # Restore connections based on original topology

        
        # Remove node_down flags from affected nodes
        for node in root_cause_mapping[alarm_id] if alarm_id in root_cause_mapping else []:
            if current_graph.nodes[node].get('has_link_down') == alarm_id:
                current_graph.nodes[node].pop('has_link_down', None)
                current_graph.nodes[node].pop('block_failed', None)
                logging.info(f"Cleared has_node_down flag from node {node}")
        
    
    # Clear UPS alarms
    elif prob_cause == 'ups_low_battery':
        
        logging.info(f"Clearing UPS alarm for {ip} for alarm {alarm_id}")
        
        if current_graph.has_node(ip):
            # Clear UPS flags
            if 'ups_low_battery_alarm_received' in current_graph.nodes[ip]:
                current_graph.nodes[ip].pop('ups_low_battery_alarm_received', None)
            if 'ups_low_battery_ne_time' in current_graph.nodes[ip]:
                current_graph.nodes[ip].pop('ups_low_battery_ne_time', None)
            
            # Clear has_ups_low_battery flag from connected routers
            for neighbor in current_graph.neighbors(ip):
                if current_graph.nodes[neighbor].get('has_ups_low_battery') == alarm_id:
                    current_graph.nodes[neighbor].pop('has_ups_low_battery', None)
                    logging.info(f"Cleared has_ups_low_battery flag from node {neighbor}")
    if alarm_id in root_cause_mapping:    
        del root_cause_mapping[alarm_id]


def local_main():
    try:
        # Setup
        setup_logging()
        mydb, cursor = setup_database()
        
        # Load network topology
        global topology_logical_graph,total,matched  # Declare we're using the global variable
        topology_logical_graph = load_topology(cursor)
        
        # Identify block routers
        block_router_ips = identify_block_routers(cursor, topology_logical_graph)
        
        # Isolate blocks by removing inter-block connections
        topology_logical_graph = remove_inter_block_connections(topology_logical_graph, block_router_ips)
        
        # Assign nodes to their blocks
        topology_logical_graph = assign_nodes_to_blocks(topology_logical_graph, block_router_ips)
        
        ring_nodes = get_ring_node_mapping(topology_logical_graph)
        # Add UPS devices
        topology_logical_graph = add_ups_devices(cursor, topology_logical_graph)
        
        # Create working copy of the graph
        current_graph = topology_logical_graph.copy()
        
        # Prepare tracking dictionaries
        root_cause_mapping = {}
        removed_links = {}  # {alarm_id: (source_ip, target_ip)}
        
        # Load Bayesian model
        with open('bayesian_rca_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Fetch alarms from database
        query = """SELECT * FROM alarm_hist where ne_time > '2025-04-20 13:39:17'"""
        cursor.execute(query)
        alarms = cursor.fetchall()
        mydb.commit()
        
        # Create timeline of events
        timeline_events = []
        for alarm in alarms:
            # Add start event
            timeline_events.append({
                'time': alarm['NE_TIME'],
                'type': 'start',
                'alarm': alarm
            })
            
            # Add clear event if available
            if alarm['CLEARED_TIME']:
                timeline_events.append({
                    'time': alarm['CLEARED_TIME'],
                    'type': 'clear',
                    'alarm': alarm
                })
        
        # Sort events chronologically
        timeline_events.sort(key=lambda x: x['time'])
        logging.info(f"Processing {len(timeline_events)} alarm events in chronological order")
        
        # Process each event
        for event in timeline_events:
            event_type = event['type']
            alarm = event['alarm']
            logging.info(f"Processing {event_type} event for alarm {alarm['ID']} at {event['time']}: {alarm['PROB_CAUSE']} on {alarm['OBJ_NAME']}")
            
            if event_type == 'start':
                process_start_event(current_graph, alarm, removed_links, model, cursor, mydb,root_cause_mapping, ring_nodes)
            elif event_type == 'clear':
                process_clear_event(current_graph, alarm, removed_links, topology_logical_graph,root_cause_mapping)
        
        # Final analysis
        logging.info("Timeline processing complete")
        logging.info(f"Remaining uncleared links: {len(removed_links)}")


        
    except Exception as e:
        logging.error(f"Error in main function: {e}", exc_info=True)       

if __name__ == "__main__":
    # Make topology_logical_graph accessible globally
    # This preserves the existing behavior of sharing the original topology
    # across various functions
    topology_logical_graph = None
    matched,total = 0,0
    main()