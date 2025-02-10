import time
import configparser
import mysql.connector
from datetime import datetime
import logging
import networkx as nx
import uuid

# Configure logging to file and console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("final.log", mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Read configuration and establish database connection.
config = configparser.ConfigParser()
config.read("config.ini")
db_config = {
    "host": config["mysql"]["host"],
    "user": config["mysql"]["user"],
    "password": config["mysql"]["password"],
    "database": config["mysql"]["database"]
}
mydb = mysql.connector.connect(**db_config)
cursor = mydb.cursor(dictionary=True)

# Construct the global physical graph from topology data.
if 'persistent_physical_graph' not in globals():
    query_phys_topo = "SELECT aendname, bendname, aendip, bendip, block_name FROM topology_data_logical"
    cursor.execute(query_phys_topo)
    phys_rows = cursor.fetchall()
    persistent_physical_graph = nx.Graph()
    for row in phys_rows:
        aend = row['aendname']
        bend = row['bendname']
        aendip = row.get('aendip')
        bendip = row.get('bendip')
        persistent_physical_graph.add_node(aend, ip=aendip)
        persistent_physical_graph.add_node(bend, ip=bendip)
        persistent_physical_graph.add_edge(aend, bend)
    print("Global physical graph constructed.")

if 'logged_isolated_nodes' not in globals():
    logged_isolated_nodes = set()
    print(f"Logged isolated initially: {logged_isolated_nodes}")

# Starting timestamp for processing alarms.
last_processed_time = datetime.strptime("2025-02-06 09:53:00", "%Y-%m-%d %H:%M:%S")

def insert_dummy_node_not_reachable_alarm(obj_name):
    """
    Inserts a dummy alarm into the alarm table for a node not reachable.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dummy_id = f"DUMMY_{uuid.uuid4()}"
    query = """
    INSERT INTO alarm (ID, NOTIF_ID, NE_TIME, OBJ_NAME, PROB_CAUSE, CATEGORY)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    params = (dummy_id, dummy_id, now, obj_name, "node_not_reachable", "EQUIPMENT ALARMS")
    cursor.execute(query, params)
    mydb.commit()
    return dummy_id

def mark_node_down_by_ip(node_ip):
    """
    Removes the node from the persistent graph based on its IP.
    """
    for node, data in list(persistent_physical_graph.nodes(data=True)):
        if data.get("ip") == node_ip:
            persistent_physical_graph.remove_node(node)
            logging.info(f"Marked node {node} with IP {node_ip} as down (removed from graph).")
            return node
    logging.warning(f"Node with IP {node_ip} not found in graph.")
    return None

def restore_link(aend, bend):
    """
    Restores a link (edge) between two nodes in the graph if both endpoints are valid.
    """
    if aend is None or bend is None:
        logging.warning(f"Cannot restore link, invalid endpoints: aend={aend}, bend={bend}")
        return
    persistent_physical_graph.add_edge(aend, bend)
    logging.info(f"Restored link {aend}-{bend} in the graph.")

while True:
    # 1. Process fixed alarms from the history table.
    query_hist = """
        SELECT * FROM alarm_hist_dummy 
        WHERE NE_TIME > %s 
        ORDER BY NE_TIME
    """
    cursor.execute(query_hist, (last_processed_time,))
    fixed_results = cursor.fetchall()
    
    if fixed_results:
        logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing fixed alarms from history...")
        for row in fixed_results:
            prob_cause = row.get("PROB_CAUSE", "").lower()
            if prob_cause == "node_not_reachable":
                node_ip = row.get("OBJ_NAME")
                # Restore node in the graph if previously marked as isolated.
                if node_ip in logged_isolated_nodes:
                    logged_isolated_nodes.remove(node_ip)
                    logging.info(f"Restored node with IP {node_ip} after fixed alarm resolution.")
            elif prob_cause == "link_down":
                # For fixed link_down alarms, restore the link in the graph.
                aend = row.get("AENDNAME")
                bend = row.get("BENDNAME")
                if not persistent_physical_graph.has_edge(aend, bend):
                    restore_link(aend, bend)
    
    # 2. Process new (active) alarms from the alarm table.
    query_new = """
        SELECT * FROM alarm 
        WHERE NE_TIME > %s 
        ORDER BY NE_TIME
    """
    cursor.execute(query_new, (last_processed_time,))
    new_results = cursor.fetchall()
    
    if new_results:
        logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing new alarms...")
        for row in new_results:
            prob_cause = row.get("PROB_CAUSE", "").lower()
            if prob_cause == "node_not_reachable":
                node_ip = row.get("OBJ_NAME")
                marked = mark_node_down_by_ip(node_ip)
                if marked:
                    logged_isolated_nodes.add(node_ip)
            elif prob_cause == "link_down":
                aend = row.get("AENDNAME")
                bend = row.get("BENDNAME")
                if persistent_physical_graph.has_edge(aend, bend):
                    persistent_physical_graph.remove_edge(aend, bend)
                    logging.info(f"Removed link {aend}-{bend} from graph due to new alarm.")
    
    # 3. Reanalyze network connectivity.
    try:
        # Use "BERLA" as the reference root node.
        connected_component = nx.node_connected_component(persistent_physical_graph, "BERLA")
    except nx.NetworkXError:
        # If "BERLA" is missing, treat all nodes as isolated.
        connected_component = set()
        logging.warning("Reference node 'BERLA' not found in the graph. All nodes considered isolated.")
    
    isolated_nodes_after = set(persistent_physical_graph.nodes()) - connected_component
    
    if isolated_nodes_after:
        logging.warning(f"Nodes isolated after processing new alarms: {isolated_nodes_after}")
        child_alarms = []
        for node in isolated_nodes_after:
            ip = persistent_physical_graph.nodes[node].get("ip")
            dummy_id = insert_dummy_node_not_reachable_alarm(ip)
            child_alarms.append(dummy_id)
        log_msg = (
            "\n[ALARM ALERT] New issues detected due to active alarms:\n"
            f"    Isolated Nodes : {', '.join(isolated_nodes_after)}\n"
            f"    Generated Alarm IDs: {', '.join(child_alarms)}\n"
            "---------------------------------------------------------------------"
        )
        logging.info(log_msg)
    else:
        logging.info("✅ Network connectivity intact after processing alarms.")
    
    # 4. Update the last processed time and pause before next cycle.
    last_processed_time = datetime.now()
    time.sleep(300)