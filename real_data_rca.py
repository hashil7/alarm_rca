import networkx as nx
import configparser
import mysql.connector
import logging
import time
import pickle
from pgmpy.inference import VariableElimination
from datetime import datetime

# Setup logging
def setup_logging():
    """Configure logging to file and console"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:\n%(message)s\n',
        handlers=[
            logging.FileHandler("real_data_rca.log", mode="a"),
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
            topology_graph.add_node(ups_ip, name=ups_name, devicetype="UPS", is_ups=True)
            ups_count += 1
        
        topology_graph.add_edge(router_ip, ups_ip, edge_type="router_ups")
        logging.info(f"Added UPS {ups_ip} connected to router {router_ip}")
    
    logging.info(f"Added {ups_count} UPS devices as nodes to the topology graph")
    return topology_graph

# Helper functions for alarm processing
def process_link_down(graph, id, ip, ne_time, rfo):
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
    
    # Parse RFO to extract the other endpoint
    if rfo and "between" in rfo and "and" in rfo:
        try:
            # Extract node names from RFO (format: "Partial Fiber Cut Detected between RONJE and UPET")
            rfo_parts = rfo.split("between")
            if len(rfo_parts) > 1:
                endpoint_parts = rfo_parts[1].split("and")
                if len(endpoint_parts) == 2:
                    node1_name = endpoint_parts[0].strip().upper()
                    node2_name = endpoint_parts[1].strip().upper()
                    
                    # Get the name of our current node (the one that sent the alarm)
                    current_node_name = graph.nodes[ip].get('name', '').upper()
                    
                    # Determine which name in the RFO corresponds to the other endpoint
                    other_node_name = None
                    if current_node_name == node1_name:
                        other_node_name = node2_name
                    elif current_node_name == node2_name:
                        other_node_name = node1_name
                    else:
                        logging.warning(f"Alarm node {ip} ({current_node_name}) does not match either endpoint in RFO: {node1_name} or {node2_name}")
                        return None
                    
                    logging.info(f"Looking for neighbor named {other_node_name} connected to {ip} ({current_node_name})")
                    
                    # Check direct neighbors only
                    for neighbor_ip in graph.neighbors(ip):
                        neighbor_name = graph.nodes[neighbor_ip].get('name', '').upper()
                        if neighbor_name == other_node_name:
                            # Found the matching neighbor
                            previous_graph = graph.copy()
                            graph.remove_edge(ip, neighbor_ip)
                            logging.info(f"Edge removed between {ip} ({current_node_name}) and {neighbor_ip} ({neighbor_name}) based on RFO")
                            
                            # Process impact
                            isolated_nodes = get_isolated_nodes(graph, previous_graph, ip)
                            logging.info(f"Isolated nodes: {isolated_nodes}")
                            
                            for node in isolated_nodes:
                                graph.nodes[node]['has_link_down'] = id
                                logging.info(f"Node {node} marked with has_link_down: {id}")
                            
                            return (ip, neighbor_ip)
                    
                    logging.warning(f"No neighbor of {ip} matches the name {other_node_name} from RFO")
        except Exception as e:
            logging.error(f"Error parsing RFO '{rfo}': {e}")
    
    logging.warning(f"No valid RFO information to determine link to remove or failed to parse RFO: {rfo}")
    return None

def process_node_down(graph, ip, id, ne_time):
    """Process a node_down alarm by removing the node"""
    if graph.nodes[ip].get('is_ups', False):
        logging.warning(f"Node {ip} is an UPS device, skipping node_down processing")
        return None
        
    previous_graph = graph.copy()
    block_failed = graph.nodes[ip].get('is_block', False)
    graph.remove_node(ip)
    logging.info(f"Node {ip} removed from the graph")
    
    isolated_nodes = get_isolated_nodes(graph, previous_graph, ip, "node_down")
    logging.info(f"Isolated nodes: {isolated_nodes}")
    
    for node in isolated_nodes:
        graph.nodes[node]['has_link_down'] = id
        if block_failed:
            graph.nodes[node]['block_failed'] = id
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
    
    try:
        router = next(graph.neighbors(ip))
    except StopIteration:
        logging.warning(f"UPS {ip} has no connected router")
        return None
        
    graph.nodes[router]['has_ups_low_battery'] = id
    logging.info(f"Router {router} marked with UPS low battery alarm")

def get_isolated_nodes(graph, previous_graph, ip, prob_cause="link_down"):
    """Analyze impact of failures by comparing node reachability"""
    if ip is None or not previous_graph.has_node(ip):
        logging.warning(f"Node {ip} not found in graph while analyzing impact")
        return set()
    
    logging.info(f"Analyzing impact of failure at {ip}")
    block_ip = previous_graph.nodes[ip].get('block_ip')
    if not block_ip:
        logging.warning(f"Node {ip} has no assigned block_ip, cannot analyze impact")
        return set()
        
    logging.info(f"Block IP for {ip}: {block_ip}")
    if block_ip not in previous_graph.nodes:
        logging.warning(f"Block IP {block_ip} not found in graph while analyzing impact")
        return set()
        
    if previous_graph is None:
        logging.info("Previous graph not available in get_isolated_nodes function")
        return set()
        
    if ip == block_ip:
        isolated_nodes = set(nx.bfs_tree(previous_graph, block_ip))
    else:    
        # Find currently reachable nodes through operational edges
        current_reachable = set(nx.bfs_tree(graph, block_ip))
        logging.info(f"Current reachable nodes from {block_ip}: {current_reachable}")
        
        # Find previously reachable nodes
        previous_reachable = set(nx.bfs_tree(previous_graph, block_ip))
        logging.info(f"Previous reachable nodes from {block_ip}: {previous_reachable}")
        isolated_nodes = previous_reachable - current_reachable
        
    if isolated_nodes is None:
        return set()
        
    if prob_cause == "node_down" and ip in isolated_nodes:
        isolated_nodes.remove(ip)
        
    ups_nodes = set()
    for node in isolated_nodes:
        if node in previous_graph.nodes and previous_graph.nodes[node].get('is_ups', False):
            ups_nodes.add(node)
            logging.info(f"Removing UPS device {node} from isolated nodes list")
    
    # Remove UPS devices from isolated nodes
    isolated_nodes -= ups_nodes  
    return isolated_nodes

def predict_root_cause(model, attrs):
    """Predict root cause using Bayesian model"""
    inference = VariableElimination(model)
    has_ups_low_battery = bool(attrs.get('has_ups_low_battery', False))
    has_link_down = bool(attrs.get('has_link_down', False))
    block_failed = bool(attrs.get('block_failed', False))
    
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
        logging.info(f"RCA ID from node: {rca_id}")
        return rca_id
        
    except Exception as e:
        logging.error(f"Error in RCA prediction: {e}")
        return None

# Process start events for different alarm types
def process_start_event(current_graph, alarm, removed_links, removed_nodes, ups_alarms, model, cursor, mydb):
    """Process a start event (new alarm)"""
    alarm_id = alarm['ID']
    prob_cause = alarm['PROB_CAUSE']
    ip = alarm['OBJ_NAME']
    event_time = alarm['NE_TIME']
    rfo = alarm['RFO']
    
    # Skip if node doesn't exist in original topology (except for UPS)
    if ip not in topology_logical_graph.nodes and prob_cause != 'ups_low_battery':
        logging.warning(f"Node {ip} not found in topology graph, skipping")
        return
    rca = None
    if prob_cause != 'link_down':
        rca=find_rca(current_graph,ip, alarm_id, model, cursor, mydb)

    # Process based on alarm type
    if prob_cause == 'link_down':
        failed_link = process_link_down(current_graph, alarm_id, ip, event_time,rfo)
        if failed_link:
            removed_links[alarm_id] = failed_link
    
    elif prob_cause == 'node_not_reachable':
        if current_graph.has_node(ip):
            # Save node attributes before removing
            node_attrs = dict(current_graph.nodes[ip])
            removed_nodes[alarm_id] = (ip, node_attrs)
            process_node_down(current_graph, ip, alarm_id, event_time)
    
    elif prob_cause == 'ups_low_battery':
        process_ups_low_battery(current_graph, ip, event_time, alarm_id)
        ups_alarms[alarm_id] = ip
    
    # After each alarm, look for isolated nodes that need RCA
    

# Process clear events for different alarm types
def process_clear_event(current_graph, alarm, removed_links, removed_nodes, ups_alarms, topology_graph):
    """Process a clear event (alarm cleared)"""
    alarm_id = alarm['ID']
    
    # Restore links
    if alarm_id in removed_links:
        source, target = removed_links[alarm_id]
        logging.info(f"Restoring link between {source} and {target} for alarm {alarm_id}")
        
        # Add the edge back
        if current_graph.has_node(source) and current_graph.has_node(target):
            current_graph.add_edge(source, target)
            
            # Remove link_down flags from nodes that were affected by this link
            for node in current_graph.nodes:
                if current_graph.nodes[node].get('has_link_down') == alarm_id:
                    current_graph.nodes[node].pop('has_link_down', None)
                    logging.info(f"Cleared has_link_down flag from node {node}")
        
        # Remove from our tracking dict
        del removed_links[alarm_id]
    
    # Restore nodes
    if alarm_id in removed_nodes:
        node_ip, node_attrs = removed_nodes[alarm_id]
        logging.info(f"Restoring node {node_ip} for alarm {alarm_id}")
        
        # Add the node back with its attributes
        current_graph.add_node(node_ip, **node_attrs)
        
        # Restore connections based on original topology
        for neighbor in topology_graph.neighbors(node_ip):
            if current_graph.has_node(neighbor):
                edge_attrs = topology_graph.get_edge_data(node_ip, neighbor)
                current_graph.add_edge(node_ip, neighbor, **edge_attrs)
        
        # Remove node_down flags from affected nodes
        for node in current_graph.nodes:
            if current_graph.nodes[node].get('has_node_down') == alarm_id:
                current_graph.nodes[node].pop('has_node_down', None)
                current_graph.nodes[node].pop('block_failed', None)
                logging.info(f"Cleared has_node_down flag from node {node}")
        
        # Remove from tracking dict
        del removed_nodes[alarm_id]
    
    # Clear UPS alarms
    if alarm_id in ups_alarms:
        ups_ip = ups_alarms[alarm_id]
        logging.info(f"Clearing UPS alarm for {ups_ip} for alarm {alarm_id}")
        
        if current_graph.has_node(ups_ip):
            # Clear UPS flags
            if 'ups_low_battery_alarm_received' in current_graph.nodes[ups_ip]:
                current_graph.nodes[ups_ip].pop('ups_low_battery_alarm_received', None)
            if 'ups_low_battery_ne_time' in current_graph.nodes[ups_ip]:
                current_graph.nodes[ups_ip].pop('ups_low_battery_ne_time', None)
            
            # Clear has_ups_low_battery flag from connected routers
            for neighbor in current_graph.neighbors(ups_ip):
                if current_graph.nodes[neighbor].get('has_ups_low_battery') == alarm_id:
                    current_graph.nodes[neighbor].pop('has_ups_low_battery', None)
                    logging.info(f"Cleared has_ups_low_battery flag from node {neighbor}")
        
        # Remove from tracking dict
        del ups_alarms[alarm_id]

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
        # Setup
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
        
        # Add UPS devices
        topology_logical_graph = add_ups_devices(cursor, topology_logical_graph)
        
        # Create working copy of the graph
        current_graph = topology_logical_graph.copy()
        
        # Prepare tracking dictionaries
        removed_links = {}  # {alarm_id: (source_ip, target_ip)}
        removed_nodes = {}  # {alarm_id: (node_ip, node_attributes)}
        ups_alarms = {}     # {alarm_id: ups_ip}
        
        # Load Bayesian model
        with open('bayesian_rca_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Fetch alarms from database
        query = """SELECT * FROM alarm_hist where ne_time > '2024-9-28 10:59:58'"""
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
                process_start_event(current_graph, alarm, removed_links, removed_nodes, ups_alarms, model, cursor, mydb)
            elif event_type == 'clear':
                process_clear_event(current_graph, alarm, removed_links, removed_nodes, ups_alarms, topology_logical_graph)
        
        # Final analysis
        logging.info("Timeline processing complete")
        logging.info(f"Remaining uncleared links: {len(removed_links)}")
        logging.info(f"Remaining uncleared nodes: {len(removed_nodes)}")
        logging.info(f"Remaining uncleared UPS alarms: {len(ups_alarms)}")
        
    except Exception as e:
        logging.error(f"Error in main function: {e}", exc_info=True)

if __name__ == "__main__":
    # Make topology_logical_graph accessible globally
    # This preserves the existing behavior of sharing the original topology
    # across various functions
    topology_logical_graph = None
    main()