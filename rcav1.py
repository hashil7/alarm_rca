import networkx as nx
import configparser
import mysql.connector
import logging
import time


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
        pr_name = row["physicalringname"]
        lr_name = row["lrname"]
        block_name = row["block_name"]
        topology_logical_graph.add_node(aend, name=aendname,lr_name=lr_name,pr_name = pr_name,block_name=block_name)
        topology_logical_graph.add_node(bend, name=bendname,lr_name=lr_name,pr_name = pr_name,block_name=block_name)
        topology_logical_graph.add_edge(aend, bend,lr_name=lr_name,pr_name = pr_name,block_name=block_name)
    logging.info("Topology Logical Graph created")
    # for edge in topology_logical_graph.edges(data=True):
    #     node1,node2,attributes = edge
    #     logging.info(f"Node: {node1} is connected to {node2} | Physical Ring: {attributes.get('pr_name', 'N/A')} | LR Ring: {attributes.get('lr_name', 'N/A')}")

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

def process_link_down(graph, ip, ne_time):
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
            graph.remove_edge(ip, closest_neighbor)
            logging.info(f"Edge removed between {ip} and {closest_neighbor} - Time diff: {time_diff}s")
            return (ip, closest_neighbor)
    
    return None
def main():
    while True:
        try:
            truncate_query = "DELETE FROM dummy_cam"
            cursor.execute(truncate_query)
            mydb.commit()
            current_graph = topology_logical_graph.copy()
            query = "SELECT * FROM alarm ORDER BY NE_TIME"
            cursor.execute(query)
            mydb.commit()
            alarms = cursor.fetchall()
            for alarm in alarms:
                prob_cause = alarm["PROB_CAUSE"]
                ip = alarm["OBJ_NAME"]
                ne_time = alarm["NE_TIME"]
                
                if prob_cause == "link_down":
                    process_link_down(current_graph, ip, ne_time)
            time.sleep(300)            
        except Exception as e:
            logging.error(f"Unhandled exception in main loop: {e}")
            time.sleep(60)
     
if __name__ == "__main__":
    main()