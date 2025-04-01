import networkx as nx
import configparser
import mysql.connector
import logging
import time
import pickle
from pgmpy.inference import VariableElimination
from datetime import datetime

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:\n%(message)s\n',
    handlers=[
        logging.FileHandler("rcav1.log", mode="a"),
        #logging.StreamHandler()
    ]
)
config = configparser.ConfigParser()
config.read("config.ini")
db_config = {
    "host": config["mysql"]["host"],
    "user": config["mysql"]["user"],
    "password": config["mysql"]["password"],
    "database": config["mysql"]["database"]
}
mydb = mysql.connector.connect(**db_config)
cursor = mydb.cursor(dictionary=True,buffered=True)
# (Place the table creation snippet here, ideally once at the start)
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

if 'topology_logical' not in globals():
    query_log_topo  = """
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
    mydb.commit()
    topology_logical_graph = nx.Graph()
    for row in topology_logical:
        aend = row["aendip"]
        bend = row["bendip"]
        aendname = row["aendname"].upper()
        bendname = row["bendname"].upper()
        pr_name = row["physicalringname"].upper()
        lr_name = row["lrname"].upper()
        block_name = row["block_name"].upper()
        topology_logical_graph.add_node(aend, name=aendname,lr_name=lr_name,pr_name = pr_name,block_name=block_name)
        topology_logical_graph.add_node(bend, name=bendname,lr_name=lr_name,pr_name = pr_name,block_name=block_name)
        topology_logical_graph.add_edge(aend, bend,lr_name=lr_name,pr_name = pr_name,block_name=block_name)
    logging.info("Topology Logical Graph created")
    # for edge in topology_logical_graph.edges(data=True):
    #     node1,node2,attributes = edge
    #     logging.info(f"Node: {node1} is connected to {node2} | Physical Ring: {attributes.get('pr_name', 'N/A')} | LR Ring: {attributes.get('lr_name', 'N/A')}")
block_router_query = """
    SELECT ip 
    FROM network_element 
    WHERE devicetype = 'BLOCK_ROUTER'
"""
cursor.execute(block_router_query)
block_routers = cursor.fetchall()
mydb.commit()

# # Create a set of block router IPs for faster lookups
block_router_ips = {router['ip'] for router in block_routers}
logging.info(f"Found {len(block_router_ips)} BLOCK_ROUTERs")

# First mark each block router
for ip in block_router_ips:
    if not topology_logical_graph.has_node(ip):
        logging.warning(f"BLOCK_ROUTER {ip} not found in topology graph")
    else:
        # Mark as block router
        topology_logical_graph.nodes[ip]['is_block'] = True
        
        # Also set its own block_ip to itself
        topology_logical_graph.nodes[ip]['block_ip'] = ip
        
        # Find all nodes reachable from this block router using BFS
        try:
            # Get all reachable nodes (bfs_tree excludes the source node by default)
            reachable_nodes = list(nx.bfs_tree(topology_logical_graph, ip))
            
            # Include the source node itself
            reachable_nodes.append(ip)
            
            # Set block_ip for all reachable nodes
            assigned_count = 0
            for node in reachable_nodes:
                    
                # Assign this node to the current block router
                topology_logical_graph.nodes[node]['block_ip'] = ip
                assigned_count += 1
            
            # Log the assignment
            block_name = topology_logical_graph.nodes[ip].get('block_name', 'Unknown')
            logging.info(f"Block router {ip} ({block_name}): {assigned_count} nodes assigned")
            
        except Exception as e:
            logging.error(f"Error assigning nodes to block router {ip}: {e}")

# Add UPS devices as spur nodes
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
mydb.commit()

# Count of UPS nodes added
ups_count = 0

# Add each UPS as a spur node connected to its router
for mapping in router_ups_mappings:
    router_ip = mapping["router_ip"]  # n1.ip
    ups_ip = mapping["ups_ip"]     # n2.ip
    
    # Skip if router doesn't exist in graph
    if not topology_logical_graph.has_node(router_ip):
        logging.warning(f"Router {router_ip} not found in topology graph. Skipping UPS {ups_ip}")
        continue
        
    # Add UPS node if it doesn't exist
    if not topology_logical_graph.has_node(ups_ip):
        router_name = topology_logical_graph.nodes[router_ip].get('name', f"Router-{router_ip}")
        ups_name = f"UPS-{router_name}"
        topology_logical_graph.add_node(ups_ip, name=ups_name, 
                                      devicetype="UPS", is_ups=True)
        ups_count += 1
    
    # Connect UPS to router
    topology_logical_graph.add_edge(router_ip, ups_ip, edge_type="router_ups")
    logging.info(f"Added UPS {ups_ip} connected to router {router_ip}")

logging.info(f"Added {ups_count} UPS devices as nodes to the topology graph")
for edge in topology_logical_graph.edges(data=True):
        node1,node2,attributes = edge
        logging.info(f"Node: {node1} is connected to {node2} | {attributes}")

def process_link_down(graph, id, ip, ne_time):
    """
    Process a link_down alarm for a node by removing the edge to the neighbor
    with the smallest time difference between link_down events.
    
    Parameters:
    -----------
    graph : NetworkX Graph
        The current network graph
    ip : str
        IP address of the node with link_down alarm
    ne_time : datetime or str
        Timestamp of the link_down alarm
        
    Returns:
    --------
    tuple or None
        (source_ip, target_ip) of removed edge, or None if no edge was removed
    """
    if not graph.has_node(ip):
        logging.warning(f"Link down alarm for non-existent node {ip}")
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
        
        # Remove the edge to the closest neighbor
        if graph.has_edge(ip, closest_neighbor):
            previous_graph = graph.copy()
            graph.remove_edge(ip, closest_neighbor)
            logging.info(f"Edge removed between {ip} and {closest_neighbor} - Time diff: {time_diff}s")
            isolated_nodes = get_isolated_nodes(graph,previous_graph,ip)
            logging.info(f"Isolated nodes: {isolated_nodes}")

            for node in isolated_nodes:
                graph.nodes[node]['has_link_down'] = id
                logging.info(f"Node {node} marked with has_link_down: {id}")
            return (ip, closest_neighbor)
    
    return None
def process_node_down(graph, ip,id, ne_time):
    previous_graph = graph.copy()
    graph.remove_node(ip)
    logging.info(f"Node {ip} removed from the graph")
    isolated_nodes = get_isolated_nodes(graph,previous_graph,ip)
    logging.info(f"Isolated nodes: {isolated_nodes}")
    block_failed = graph.nodes[ip].get('is_block', False)
    for node in isolated_nodes:
        graph.nodes[node]['has_node_down'] = id
        graph.nodes[node]['block_failed'] = block_failed
        logging.info(f"Node {node} marked with has_node_down: {id}")

    return None
def predict_root_cause(model,attrs):
    inference = VariableElimination(model)
    has_ups_low_battery = attrs.get('has_ups_low_battery', False)
    has_link_down = attrs.get('has_link_down', False)
    block_failed = attrs.get('block_failed', False)
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
        
        # Format nicely for logging
        log_message = ["RCA Probability Distribution:"]
        log_message.append("-" * 50)
        log_message.append(f"{'Root Cause':<20} | {'Probability':<15} | {'Rank'}")
        log_message.append("-" * 50)
        
        for i, (cause, prob) in enumerate(sorted_probs):
            log_message.append(f"{cause:<20} | {prob:.6f} | {i+1}")
        
        log_message.append("-" * 50)
        
        # Log the formatted probability distribution
        logging.info("\n".join(log_message))
        
        # Find most probable RCA (first item in sorted list)
        most_probable_rca = sorted_probs[0][0]
        
        # Log the final prediction
        logging.info(f"Selected RCA: {most_probable_rca} with probability {sorted_probs[0][1]:.6f}")
        rca_id = attrs.get(most_probable_rca,None)
        logging.info(f"RCA ID from node: {rca_id}")
        return rca_id
        
    except Exception as e:
        logging.error(f"Error in RCA prediction: {e}")
        return None

def process_ups_low_battery(graph, ip, ne_time,id):
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

def get_isolated_nodes(graph, previous_graph, ip):
    """
    Analyze impact of failures by comparing node reachability
    
    Parameters:
    -----------
    graph : NetworkX Graph
        Current network graph
    previous_graph : NetworkX Graph or str, optional
        Previous graph state or node IP
    ip : str, optional
        IP address of node to analyze
        
    Returns:
    --------
    set
        Set of nodes that became unreachable
    """
    # Handle case where previous_graph is the IP (backward compatibility)

    # Basic validation
    if ip is None or not graph.has_node(ip):
        logging.warning(f"Node {ip} not found in graph while analyzing impact")
        return set()
    
    logging.info(f"Analyzing impact of failure at {ip}")
    block_ip = graph.nodes[ip].get('block_ip')
    if not block_ip:
        logging.warning(f"Node {ip} has no assigned block_ip, cannot analyze impact")
        return set()
    logging.info(f"Block IP for {ip}: {block_ip}")
    # Create operational graph (without failed edges)
   
    # Add only non-failed edges

    
    # Find currently reachable nodes through operational edges
    current_reachable = set(nx.bfs_tree(graph, block_ip))
    logging.info(f"Current reachable nodes from {block_ip}: {current_reachable}")
    # If we don't have previous graph, we can only report current state
    if previous_graph is None:
        logging.info("Previous graph not available in get_isolated_nodes function")
        return None
    
    # Find previously reachable nodes
    previous_reachable = set(nx.bfs_tree(previous_graph, block_ip))
    logging.info(f"Previous reachable nodes from {block_ip}: {previous_reachable}")
    # Return nodes that were reachable before but not now
    return previous_reachable - current_reachable
    
           
def main():
    while True:
        try:
            truncate_query = "DELETE FROM dummy_cam"
            cursor.execute(truncate_query)
            mydb.commit()
            current_graph = topology_logical_graph.copy()
            query = """SELECT * FROM berla_alarms where prob_cause = "link_down" ORDER BY NE_TIME"""
            cursor.execute(query)
            mydb.commit()
            alarms = cursor.fetchall()
            for alarm in alarms:
                id = alarm["ID"]
                prob_cause = alarm["PROB_CAUSE"]
                ip = alarm["OBJ_NAME"]
                ne_time = alarm["NE_TIME"]
                with open('bayesian_rca_model.pkl', 'rb') as f:
                    model = pickle.load(f)
                root_cause = predict_root_cause(model,current_graph.nodes[ip])
                if root_cause is None:
                    root_cause = id
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                insert_query = "INSERT INTO dummy_cam (rca_id, child_id, timestamp) VALUES (%s, %s, %s)"
                cursor.execute(insert_query, (root_cause, id, timestamp))
                mydb.commit()               
                if prob_cause == "link_down":
                    process_link_down(current_graph,id, ip, ne_time)
                elif prob_cause == "node_not_reachable":
                    process_node_down(current_graph, ip,id, ne_time)
                elif prob_cause == "ups_low_battery":
                    process_ups_low_battery(current_graph, ip, ne_time,id)
               
            time.sleep(300)            
        except Exception as e:
            logging.error(f"Unhandled exception in main loop: {e}")
            time.sleep(60)
     
if __name__ == "__main__":
    main()