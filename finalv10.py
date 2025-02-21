import os
import time
import configparser
import mysql.connector
from datetime import datetime
from collections import defaultdict
from itertools import combinations
import logging
import networkx as nx
import uuid
# Initialize root_cause_mapping based on log file
root_cause_mapping = {}

node_edges_backup = {}  # Store edges when removing nodes

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:\n%(message)s\n',
    handlers=[
        logging.FileHandler("finalv10.log", mode="a"),
        #logging.StreamHandler()
    ]
)
# Read config and setup DB connection
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
CREATE TABLE IF NOT EXISTS child_alarm_mapping (
    id INT AUTO_INCREMENT PRIMARY KEY,
    root_cause_alarm_id VARCHAR(255),
    child_alarm_ids TEXT,
    child_alarm_timestamps TEXT
)
"""
cursor.execute(create_table_query)
mydb.commit()
if 'persistent_physical_graph' not in globals():
    query_phys_topo = "SELECT aendname, bendname, aendip, bendip,block_name FROM topology_data_logical"
    cursor.execute(query_phys_topo)
    mydb.commit()
    phys_rows = cursor.fetchall()
    persistent_physical_graph = nx.Graph()
    for row in phys_rows:
        aend = row['aendname'].upper()	
        bend = row['bendname'].upper()
        aendip = row.get('aendip')
        bendip = row.get('bendip')
        # Add nodes with an 'ip' attribute so we can later locate them by IP.
        persistent_physical_graph.add_node(aend, ip=aendip)
        persistent_physical_graph.add_node(bend, ip=bendip)
        persistent_physical_graph.add_edge(aend, bend)
    logging.info("Global graph constructed with the following nodes.")
    for node in persistent_physical_graph.nodes():
        logging.info(f"Node {node} with IP {persistent_physical_graph.nodes[node].get('ip')}")


if 'logged_isolated_nodes' not in globals():
    logged_isolated_nodes = set()

# Initialize separate timestamps for each table
last_processed_alarm_time = datetime.now()
last_processed_hist_time = datetime.now()

def get_connection_info(result_conn):
    """
    Extract a connection name (formatted as 'aendname-bendname') and lr_ring from the query result.
    Assumes result_conn is a list of dictionaries with keys 'aendname', 'bendname', and 'lrname'.
    """
    if result_conn:
        physical_ring = result_conn[0]['physicalringname']
        connection_name = f"{result_conn[0]['aendname']}-{result_conn[0]['bendname']}"
        lr_ring = result_conn[0]['lrname']
        block = result_conn[0]['block_name']
        district = result_conn[0]['district_name']
        physical_segments = result_conn[0]['physicalsegments']
        return connection_name, lr_ring, block, district,physical_ring, physical_segments
    return None, None, None, None, None, None

def get_num_lrings(physical_ring):
  query_count = """
    SELECT COUNT(DISTINCT lrname) AS logical_ring_count
    FROM topology_data_logical
    WHERE physicalringname = %s
  """
  cursor.execute(query_count, (physical_ring,))
  mydb.commit()
  result_count = cursor.fetchone()
  count = result_count["logical_ring_count"]
  return count

def get_physical_connection(physicalsegment, physicalring):
    """
    Get physical connection details for a given segment and ring.
    
    Args:
        physicalsegment: The segment ID to look up
        physicalring: The physical ring name
    Returns:
        String formatted as 'aendname-bendname' or None if not found
    """
    query_phys = """
      SELECT aendname, bendname FROM topology_data_physical
      WHERE ringname = %s AND segmentid = %s
    """
    cursor.execute(query_phys, (physicalring, physicalsegment))
    mydb.commit()
    r = cursor.fetchone()
    if r:
        return f"{r['aendname']}-{r['bendname']}"
    return None

def get_common_segments(segment_list):
    """
    Given a list of strings, where each string is a comma-separated list of segments,
    return the set of common segments across all elements.
    """
    # Convert each comma-separated string into a set of trimmed segment strings.
    segment_sets = [set(s.strip() for s in segments.split(',')) for segments in segment_list]
    # Use set intersection to find common segments among all sets.
    return set.intersection(*segment_sets) if segment_sets else set()

def mark_node_down_by_name(node_name):
    """Store node's edges and IP before removing it."""
    node_name = node_name.upper()
    if node_name in persistent_physical_graph.nodes():
        # Backup edges and node IP before removing node
        node_ip = persistent_physical_graph.nodes[node_name].get('ip')
        node_edges_backup[node_name] = {
            'edges': list(persistent_physical_graph.edges(node_name)),
            'ip': node_ip
        }
        persistent_physical_graph.remove_node(node_name)
        logging.info(f"Node {node_name} with IP {node_ip} removed from the graph")
        return node_name
    return None

def get_block_from_ip(ip):
    """
    Fetch the block name from network_element table based on IP address.
    Returns only the alphabetical part of the block name or None as fallback.
    
    Example: "Block : 106:::BERLA" -> Returns "BERLA"
    """
    query = """
        SELECT locationText
        FROM network_element
        WHERE ip = %s
    """
    cursor.execute(query, (ip,))
    result = cursor.fetchone()
    mydb.commit()
    
    if result and result['locationText']:
        location_text = result['locationText']
        
        # Ensure 'Block :' exists before splitting
        if "Block :" in location_text:
            try:
                # Extract the block part
                block_part = location_text.split("Block :")[1]
                block_raw = block_part.split(";")[0].strip()
                
                # Extract only alphabetical part (handles cases like "106:::BERLA")
                block_name = ''.join(c for c in block_raw.split(":")[-1] if c.isalpha())
                
                if block_name:
                    logging.info(f"Block from db: {block_name}")
                    return block_name.upper()
                    
            except IndexError:
                logging.warning(f"Could not parse block from locationText for IP {ip}")
    
    logging.warning(f"Could not find block name from locationText for IP {ip}")
    return None

def fetch_alarm_id_by_ip_and_time(ip, ne_time, prob_cause="node_not_reachable"):
    """
    Fetches the alarm ID from the alarm table by matching the given IP (OBJ_NAME), PROB_CAUSE,
    and NE_TIME (which should be the same as the linked down alarm time).
    Returns the ID if found, or None otherwise.
    """
    query = """
        SELECT ID FROM alarm
        WHERE OBJ_NAME = %s AND PROB_CAUSE = %s AND NE_TIME >= %s
        ORDER BY NE_TIME DESC LIMIT 1
    """
    cursor.execute(query, (ip, prob_cause, ne_time))
    mydb.commit()
    result = cursor.fetchone()
    if result:
        return result["ID"]
    return None

def get_node_from_ip(ip):
    """
    Fetch the node name from network_element table based on IP address.
    Returns the last alphabetical text component from locationText or None as fallback.
    
    Example: 
    "Cluster : A2; District : BEMETARA; Block : 106:::BERLA; GP : KIRITPUR" -> Returns "KIRITPUR"
    """
    query = """
        SELECT locationText
        FROM network_element
        WHERE ip = %s
    """
    cursor.execute(query, (ip,))
    result = cursor.fetchone()
    mydb.commit()
    
    if result and result['locationText']:
        location_text = result['locationText']
        try:
            # Split by semicolon and get the last non-empty part
            parts = [part.strip() for part in location_text.split(';') if part.strip()]
            if not parts:
                return None
                
            # Get the last part and extract text after the last colon
            last_part = parts[-1]
            if ':' in last_part:
                # Extract only alphabetical characters after the last colon
                node_name = ''.join(c for c in last_part.split(':')[-1].strip() if c.isalpha())
                if node_name:
                    logging.info(f"Node name from locationText: {node_name}")
                    return node_name.upper()
                    
        except (IndexError, AttributeError):
            logging.warning(f"Could not parse node name from locationText for IP {ip}")
    
    logging.warning(f"Could not find node name from locationText for IP {ip}")
    return None

# Group link_down alarms with time tolerance
def are_times_within_tolerance(time1, time2, tolerance_seconds=5):  # 5 minutes tolerance
    """Check if two timestamps are within the specified tolerance."""
    time_diff = abs((time1 - time2).total_seconds())
    return time_diff <= tolerance_seconds

# Modified grouping logic
all_groups = defaultdict(list)
current_group_time = None

def insert_child_alarm_mapping(root_cause_id, child_id):
    """
    Insert a child alarm mapping into the database.
    
    Args:
        root_cause_id: ID of the root cause alarm
        child_id: ID of the child alarm
    """
    current_time = datetime.now()
    insert_query = """
        INSERT INTO child_alarm_mapping 
        (root_cause_alarm_id, child_alarm_ids, child_alarm_timestamps)
        VALUES (%s, %s, %s)
    """
    try:
        cursor.execute(insert_query, (
            str(root_cause_id),
            str(child_id),
            str(current_time)
        ))
        mydb.commit()
        logging.info(f"Inserted child alarm mapping: Root={root_cause_id}, Child={child_id}, Time={current_time}")
    except mysql.connector.Error as err:
        logging.error(f"Error inserting child alarm mapping: {err}")
        mydb.rollback()

while True:
    try:
        print("\n=== Debug Information ===")
        print(f"Current last_processed_hist_time: {last_processed_hist_time}")
        logging.info("Analysis Start")
        logging.info(f"Start Time for Processing: {last_processed_hist_time}")
        
        query_hist = """
            SELECT * FROM alarm_hist
            WHERE NE_TIME > %s 
            ORDER BY NE_TIME
        """
        logging.info(f"Trying to fetch fixed alarms from alarm_hist with query{query_hist}")
        # Debug the actual SQL being executed
        cursor.execute(query_hist, (last_processed_hist_time,))
        mydb.commit()
        logging.info(f"Executed query with parameter: {last_processed_hist_time}")
        
        fixed_results = cursor.fetchall()
        logging.info(f"Number of results returned after query into alarm_hist: {len(fixed_results)}")
        
        
        # Update last_processed_hist_time if there were results
        if fixed_results:
            last_processed_hist_time = max(row['NE_TIME'] for row in fixed_results)
            logging.info(f"Alarm hist wil be fetched from this time in next iter : {last_processed_hist_time}")
            

        # Group fixed alarms by NE_TIME and type
        fixed_groups = defaultdict(list)
        current_group_time = None
            
        # Sort fixed alarms by time first
        sorted_fixed_results = sorted(fixed_results, key=lambda x: x['NE_TIME'])

        for row in sorted_fixed_results:
            prob_cause = row.get("PROB_CAUSE", "").lower()
            ne_time = row["NE_TIME"]
            
            # Start a new group if this is the first alarm or if outside tolerance of current group
            if current_group_time is None or not are_times_within_tolerance(ne_time, current_group_time):
                current_group_time = ne_time
            
            if prob_cause == "link_down":
                fixed_groups[current_group_time].append(("link_down", row))
                logging.info(f"Fixed link_down alarm grouped at {current_group_time}: {row}")
            elif prob_cause == "node_not_reachable":
                fixed_groups[current_group_time].append(("node_not_reachable", row))
                logging.info(f"Fixed node_not_reachable alarm grouped at {current_group_time}: {row}")

        # Process fixed alarms
        if fixed_results:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing fixed alarms:")
            for ne_time in sorted(fixed_groups.keys()):
                group = fixed_groups[ne_time]
                logging.info(f"Processing fixed alarms NE_TIME group: {ne_time}")
                print(f"\nFixed Alarms NE_TIME Group: {ne_time}")
                
                # Process fixed node_not_reachable alarms
                node_down_group = [alarm for (atype, alarm) in group if atype == "node_not_reachable"]
                
                if node_down_group:
                    for alarm in node_down_group:
                        node_ip = alarm.get("OBJ_NAME")
                        alarm_id = alarm.get("ID")
                        node_name = None
                        if alarm_id in root_cause_mapping:
                            mapping = root_cause_mapping[alarm_id]
                            node_name = mapping.get('root node_not_reachable', None)
                            
                            # Restore the node in the physical graph
                            if node_name in node_edges_backup:
                                backup_data = node_edges_backup[node_name]
                                # First add back the node with its IP
                                persistent_physical_graph.add_node(node_name, ip=backup_data['ip'])
                                logging.info(f"Restored node {node_name} with IP {backup_data['ip']}")
                                
                                # Then restore all its edges
                                for edge in backup_data['edges']:
                                    neighbor = edge[1]  # Get the neighbor node
                                    persistent_physical_graph.add_edge(node_name, neighbor)
                                    logging.info(f"Restored edge between {node_name} and {neighbor}")
                                
                                del node_edges_backup[node_name]
                                logging.info(f"Removed {node_name} from node_edges_backup after restoring")
                            else:
                                logging.info(f"No backup edges found for node {node_name}")
                            logging.info(f"the received fixed alarm {alarm_id} was a RC alarm. Clearing associated isolated nodes.")
                            for iso_node in mapping.get('isolated_nodes', []):
                                if iso_node in logged_isolated_nodes:
                                    logged_isolated_nodes.remove(iso_node)
                                    logging.info(f"Cleared isolated status for node: {iso_node}")
                            
                            del root_cause_mapping[alarm_id]
                            continue
                        
                        # Get node name from database
                        node_name = get_node_from_ip(node_ip)
                        
                        
                        if not node_name:
            
                            logging.warning(f"Could not find block name for IP {node_ip} from network_element table.")
                            continue
                                
                        if node_name in logged_isolated_nodes:
                            logging.info(f"Clearing isolated status for node {node_name}")
                            logged_isolated_nodes.remove(node_name)
                        elif node_name in node_edges_backup:
                            backup_data = node_edges_backup[node_name]
                            # First add back the node with its IP
                            persistent_physical_graph.add_node(node_name, ip=backup_data['ip'])
                            # Then restore all its edges
                            for edge in backup_data['edges']:
                                neighbor = edge[1]  # Get the neighbor node
                                persistent_physical_graph.add_edge(node_name, neighbor)
                                logging.info(f"Restored edge between {node_name} and {neighbor}")
                            del node_edges_backup[node_name]
                            logging.info(f"Removed {node_name} from node_edges_backup after restoring")
                        else:
                            logging.info(f"Node {node_name} was not previously marked as isolated")
                # Process fixed link_down alarms
                link_down_group = [alarm for (atype, alarm) in group if atype == "link_down"]
                if link_down_group:
                    logging.info(f"Processing Fixed link_down alarms: {link_down_group}")
                    lr_rings_in_group = []
                    physical_segments_in_group = []
                    physicalring = None
                    
                    # Check all possible combinations in the current group
                    for alarm1, alarm2 in combinations(link_down_group, 2):
                        ip1, ifIndex1 = alarm1.get("OBJ_NAME"), alarm1.get("interface")
                        ip2, ifIndex2 = alarm2.get("OBJ_NAME"), alarm2.get("interface")
                        id1, id2 = alarm1.get("ID"), alarm2.get("ID")
                        
                        if not (ip1 and ifIndex1 and ip2 and ifIndex2):
                            continue
                        
                        query_conn = """
                            SELECT physicalringname, aendname, bendname, lrname, physicalsegments, district_name, block_name 
                            FROM topology_data_logical
                            WHERE (aendip = %s AND aendifIndex = %s AND bendip = %s AND bendifIndex = %s)
                               OR (aendip = %s AND aendifIndex = %s AND bendip = %s AND bendifIndex = %s)
                        """
                        cursor.execute(query_conn, (ip1, ifIndex1, ip2, ifIndex2, ip2, ifIndex2, ip1, ifIndex1))
                        result_conn = cursor.fetchall()
                        
                        connection_name, lr_ring, block, district, curr_physicalring, physicalsegments = get_connection_info(result_conn)
                        if connection_name:
                            logging.info(f"Found valid connection from topology for fixed link_down: {connection_name}")
                            # Find nodes by their IP attributes
                            aend, bend = connection_name.split('-')  # Use the connection name which has the correct node names
                            
                            # Restore edge in the graph if both endpoints exist
                            if aend in persistent_physical_graph.nodes() and bend in persistent_physical_graph.nodes():
                                persistent_physical_graph.add_edge(aend, bend)
                                logging.info(f"Restored edge in graph: {aend} <--> {bend}")
                
                            # Then check and log RCA details
                            root_cause_key = f"{min(id1, id2)}_{max(id1, id2)}"
                            if root_cause_key in root_cause_mapping:
                                child_alarm_ids = root_cause_mapping[root_cause_key]['ids']
                                logging.info(f"Child alarm IDs stored for this fixed link_down {root_cause_key}: {child_alarm_ids}")
                                fixed_node_alarms = {alarm.get("ID") for alarm in node_down_group}
                                logging.info(f"Fixed node alarms in alarm_hist before processing this fixed link_down: {fixed_node_alarms}")
                                fixed_children = child_alarm_ids & fixed_node_alarms
                                remaining_children = child_alarm_ids - fixed_node_alarms
                    
                                
                                root_cause_names = root_cause_mapping[root_cause_key]['root_cause_names']
                                isolated_nodes = root_cause_mapping[root_cause_key]['isolated_nodes']
                                
                                # Get node names for fixed and remaining alarms
                                fixed_nodes = []
                                remaining_nodes = []
                                for alarm in node_down_group:
                                    alarm_id = alarm.get("ID")
                                    node_ip = alarm.get("OBJ_NAME")
                                    if alarm_id in fixed_children:
                                        for node, attrs in persistent_physical_graph.nodes(data=True):
                                            if attrs.get('ip') == node_ip:
                                                logging.info(f"Found fixed Child alarm node {node} with IP {node_ip}")
                                                fixed_nodes.append(node)
                                                break
                                
                                # Get remaining nodes from isolated_nodes list that aren't fixed
                                remaining_nodes = [node for node in isolated_nodes if node not in fixed_nodes]
                                
                                log_message = (
                                    "="*80 + "\n"
                                    "FIXED LINK DOWN DETECTED\n" +
                                    "="*80 + "\n"
                                    f"Alarm IDs: {id1} and {id2}\n"
                                    f"Root Cause Connections: {', '.join(root_cause_names)}\n"
                                    f"Originally Isolated Nodes: {', '.join(isolated_nodes)}\n\n"
                                    "Connection Details:\n"
                                    f"   Endpoints: {ip1}:{ifIndex1} <--> {ip2}:{ifIndex2}\n"
                                    f"   Connection Name: {connection_name}\n"
                                    f"   LR Ring: {lr_ring}\n"
                                    f"   Physical Ring: {curr_physicalring}\n"
                                    f"   Physical Segments: {physicalsegments}\n"
                                    f"   Block: {block}\n"
                                    f"   District: {district}\n"
                                    f"   Graph Status: Edge restored\n" +
                                    "="*80
                                )
                                logging.info(log_message)
                                
                                if remaining_children:
                                    logging.warning(
                                        "="*80 + "\n"
                                        "WARNING: INCOMPLETE RESTORATION\n" +
                                        "="*80 + "\n"
                                        f"Root Cause Alarms ({root_cause_key}) fixed but some alarms remain:\n\n"
                                        "Fixed:\n"
                                        f"    Alarm IDs: {', '.join(map(str, fixed_children))}\n"
                                        f"    Nodes: {', '.join(fixed_nodes)}\n\n"
                                        "Remaining:\n"
                                        f"    Alarm IDs: {', '.join(map(str, remaining_children))}\n"
                                        f"    Nodes: {', '.join(remaining_nodes)}\n"
                                        f"Note: Possible power failure for remaining nodes\n" +
                                        "="*80
                                    )
                                else:
                                    logging.info(
                                        "="*80 + "\n"
                                        "COMPLETE RESTORATION\n" +
                                        "="*80 + "\n"
                                        f"Root Cause Connection: {connection_name}\n"
                                        f"Root Cause Alarms: {root_cause_key}\n"
                                        "All alarms resolved:\n"
                                        f"    Alarm IDs: {', '.join(map(str, child_alarm_ids))}\n"
                                        f"    Nodes: {', '.join(isolated_nodes)}\n" +
                                        "="*80
                                    )
                                
                                del root_cause_mapping[root_cause_key]
                            else:
                                logging.info(f"There is no previous RCA mapping for this fixed link_down {root_cause_key}")

                            
                            lr_rings_in_group.append(lr_ring)
                            physical_segments_in_group.append(physicalsegments)
                            if physicalring is None:
                                physicalring = curr_physicalring
                    
                    if len(lr_rings_in_group) > 0:
                        distinct_rings = set(lr_rings_in_group)
                        if physicalring:
                            total_lrings = get_num_lrings(physicalring)
                            if len(distinct_rings) == total_lrings:
                                unique_physical_segment = get_common_segments(physical_segments_in_group)
                                if unique_physical_segment:
                                    physical_cut = get_physical_connection(unique_physical_segment.pop(), physicalring)
                                    logging.info(
                                        "="*80 + "\n"
                                        "PHYSICAL CUT RESTORED\n" +
                                        "="*80 + "\n"
                                        f"Time: {ne_time}\n"
                                        f"Physical Ring: {physicalring}\n"
                                        f"Connection: {physical_cut}\n"
                                        
                                        "="*80
                                    )

        logging.info(f"Now cheching alarm table from {last_processed_alarm_time}")
        query = "SELECT * FROM alarm WHERE NE_TIME > %s ORDER BY NE_TIME"
        cursor.execute(query, (last_processed_alarm_time,))
        mydb.commit()
        results = cursor.fetchall()
        
        # Update last_processed_alarm_time if there were results
        if results:
            last_processed_alarm_time = max(row['NE_TIME'] for row in results)
            logging.info(f"Alarm table will be fetched from this time in next iter: {last_processed_alarm_time}")
            print(f"last_processed_alarm_time: {last_processed_alarm_time}")
        # Group link_down alarms by NE_TIME
        all_groups = defaultdict(list)
        current_group_time = None

        # Sort results by NE_TIME first
        sorted_results = sorted(results, key=lambda x: x['NE_TIME'])

        for row in sorted_results:
            prob_cause = row.get("PROB_CAUSE", "").lower()
            ne_time = row["NE_TIME"]
            
            # Start a new group if this is the first alarm or if outside tolerance of current group
            if current_group_time is None or not are_times_within_tolerance(ne_time, current_group_time):
                current_group_time = ne_time
            
            if prob_cause == "link_down":
                all_groups[current_group_time].append(("link_down", row))
                logging.info(f"Link down alarm grouped at {current_group_time}: {row}")
            elif prob_cause == "node_not_reachable":
                all_groups[current_group_time].append(("node_not_reachable", row))
                logging.info(f"Node not reachable alarm grouped at {current_group_time}: {row}")

        if results:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] New alarms:")
            for ne_time in sorted(all_groups.keys()):
              group = all_groups[ne_time]
              logging.info(f"Processing NE_TIME group: {ne_time}")
              print(f"\nNE_TIME Group: {ne_time}")
              node_down_group = [alarm for (atype, alarm) in group if atype == "node_not_reachable"]
              

              
              link_down_group = [alarm for (atype, alarm) in group if atype == "link_down"]
            
              
              if link_down_group:
                print(f"link_down_group: {link_down_group}")
                lr_rings_in_group = []
                physcical_segments_in_group = []
                physicalring = None
                  # Check all possible combinations in the current group
                for alarm1, alarm2 in combinations(link_down_group, 2):
                    ip1, ifIndex1 = alarm1.get("OBJ_NAME"), alarm1.get("interface")
                    ip2, ifIndex2 = alarm2.get("OBJ_NAME"), alarm2.get("interface")
                    id1, id2 = alarm1.get("ID"), alarm2.get("ID")
                    ne_time = alarm1.get("NE_TIME")
                    print(f"ip1: {ip1}, ifIndex1: {ifIndex1}, ip2: {ip2}, ifIndex2: {ifIndex2}")
                    if not (ip1 and ifIndex1 and ip2 and ifIndex2):
                        continue
                    
                    query_conn = """
                        SELECT physicalringname, aendname, bendname, lrname, physicalsegments, district_name, block_name FROM topology_data_logical
                        WHERE (aendip = %s AND aendifIndex = %s AND bendip = %s AND bendifIndex = %s)
                          OR (aendip = %s AND aendifIndex = %s AND bendip = %s AND bendifIndex = %s)
                    """
                    cursor.execute(query_conn, (ip1, ifIndex1, ip2, ifIndex2, ip2, ifIndex2, ip1, ifIndex1))
                    mydb.commit()
                    result_conn = cursor.fetchall()
                    
                    connection_name, lr_ring, block,district, curr_physicalring, physicalsegments = get_connection_info(result_conn)
                    if connection_name:
                        log_message = (
                          f"Due to the below alarms a logical cut has happened:\n"
                          f"   Alarm IDs: {id1} and {id2}\n"
                          f"   Endpoints: {ip1}:{ifIndex1} <--> {ip2}:{ifIndex2}\n"
                          f"   Connection Name: {connection_name}\n"
                          f"   LR Ring: {lr_ring}\n"
                          f"   Physical Ring: {curr_physicalring}\n"
                          f"   Physical Segments: {physicalsegments}\n"
                          f"   Block: {block}\n"
                          f"   District: {district}\n"
                          "----------------------------------------")

                        logging.info(log_message)
                        aend, bend = connection_name.split('-')
                        if persistent_physical_graph.has_edge(aend, bend):
                            persistent_physical_graph.remove_edge(aend, bend)
                            
                            print(f"Edge {aend}-{bend} removed from global physical graph.")
                        else:
                            
                            print(f"Edge {aend}-{bend} not found in global physical graph.")

                        # Check for isolated nodes from a chosen start node (e.g. 'BERLA')
                        start_node = block.upper()
                        
                        logging.info(f"Block Node from topology data: {start_node}")
                        if start_node in persistent_physical_graph.nodes:
                            # Get all nodes connected to start_node (all possible nodes we could reach)
                            connected_nodes = set(nx.node_connected_component(persistent_physical_graph, start_node))
                            
                            # Get nodes actually reachable from start_node
                            reachable = set(nx.single_source_shortest_path_length(persistent_physical_graph, start_node).keys())
                            
                            # Isolated nodes are those that should be connected but aren't reachable
                            current_isolated_nodes = connected_nodes - reachable
                            
                            logging.info(f"Connected nodes: {connected_nodes}")
                            logging.info(f"Reachable nodes: {reachable}")
                            logging.info(f"Isolated nodes: {current_isolated_nodes}")
                        else:
                            logging.info(f"The block node {start_node} is not in the graph")
                            current_isolated_nodes = set()
                            continue
                        new_isolated = current_isolated_nodes - logged_isolated_nodes
                        child_alarms = []
                        if new_isolated:
                            for node in new_isolated:
                                ip = persistent_physical_graph.nodes[node].get('ip')
                                print(f"Passed ne_time: {ne_time}")
                                dummy_id = fetch_alarm_id_by_ip_and_time(ip, ne_time)
                                if dummy_id:
                                    child_alarms.append(dummy_id)
                                    insert_child_alarm_mapping(alarm_id, dummy_id)
                                    logging.info(f"Child alarm entry detected at same time as root alarm - Node: {node}, IP: {ip}, Child ID: {dummy_id}, Parent ID: {alarm_id}, Time: {ne_time}")
                                
                            
                            # Store root cause and child alarm mapping with additional information
                            root_cause_key = f"{min(id1, id2)}_{max(id1, id2)}"  # Consistent ordering
                            root_cause_mapping[root_cause_key] = {
                                'ids': set(child_alarms),
                                'root_cause_names': [connection_name],
                                'isolated_nodes': list(new_isolated)
                            }
                            
                            root_cause_alarm_ids = f"{id1} and {id2}"
                            logging.info(
                                "="*80 + "\n"
                                f"Isolated Nodes: {new_isolated}\n"
                                f"Root Cause Alarms: {root_cause_alarm_ids}\n"
                                f"Child Alarm IDs: {', '.join(map(str, child_alarms))}\n"
                                f"Root Cause Connection: {connection_name}\n"
                                f"Root Cause Mapping Key: {root_cause_key}\n" +
                                "="*80
                            )
                            
                            print("Isolated nodes after logical cut:", new_isolated)
                            logged_isolated_nodes.update(new_isolated)
                            print(f"Logged Isolated after logical cut {connection_name}: {logged_isolated_nodes}")
                        else:
                            logging.info("No isolated nodes after the cut.")
                        print(f"✅ Connection exists between {ip1}:{ifIndex1} and {ip2}:{ifIndex2}")
                        print(f"   Connection Name: {connection_name}")
                        print(f"   LR Ring: {lr_ring}")
                        lr_rings_in_group.append(lr_ring)
                        physcical_segments_in_group.append(physicalsegments)
                        if physicalring is None:
                          physicalring = curr_physicalring
                    else:
                        
                        logging.info(f"❌ No connection found between {ip1}:{ifIndex1} and {ip2}:{ifIndex2} for alarms with {id1} and {id2}")   


                if len(lr_rings_in_group)>0:
                    distinct_rings = set(lr_rings_in_group)
                    logging.info(f"Distinct LR Rings in the link down alarms at the same time: {distinct_rings}")
                    total_lrings  = get_num_lrings(physicalring)
                    logging.info(f"Physical Ring for the alarm: {physicalring}")
                    logging.info(f"Total lr rings for the physical ring: {total_lrings}")
                    print(distinct_rings)
                    if len(distinct_rings)==total_lrings:
                        logging.info(
                        f"All logical cuts cover all logical rings in the physical ring...Might be a physical cut"
                        )
                    
                        unique_physical_segment = get_common_segments(physcical_segments_in_group)
                        logging.info(f"Unique Segment found for all logical rings: {unique_physical_segment}")
                        print(f"Physical Segments: {physcical_segments_in_group}")
                        if unique_physical_segment:
                            physical_cut = get_physical_connection(unique_physical_segment.pop(), physicalring)
                            logging.info(
                                f"All the above alarms at {ne_time} have occurred due to an OFC.\n"
                                f"Connection: {physical_cut} in Physical Ring: {physicalring}"
                            )
                    else:
                        logging.info("The logical cuts does not cover all logical rings...cannot be a physical cut")
                    if len(distinct_rings) > 1:
                        logging.info("⚠️ Logical cuts detected in different rings:", distinct_rings)
                    elif len(distinct_rings) == 1 and distinct_rings:
                        ring = distinct_rings.pop()
                        logging.info("All logical cuts in this group belong to the same ring: %s", ring)
                    else:
                        logging.info("No logical connections were found in this group.")    
                # Update last_processed_time to the latest NE_TIME from the results
              if node_down_group:

                for alarm in node_down_group:
                  
                    logging.info(f"New node down alarm processing: {alarm}")
                    node_ip = alarm.get("OBJ_NAME")
                    node_name = None
                    for n, attrs in persistent_physical_graph.nodes(data=True):
                        if attrs.get("ip") == node_ip:
                            node_name = n
                            logging.info(f"Found node {node_name} with IP {node_ip} in the graph")
                            break
                    for key, mapping in root_cause_mapping.items():
                        if node_name and node_name in mapping.get('isolated_nodes', []):
                            alarm_id = alarm.get("ID")
                            mapping['ids'].add(alarm_id)
                            logging.info(f"Found the child alarm ID {alarm_id} to a previous root cause key {key}")
                            insert_child_alarm_mapping(key, alarm_id)
                            continue
                    if not node_name:
                        logging.info(f"Node {node_name} not found in the graph")
                        continue
                    if  node_name in logged_isolated_nodes:
                        logging.info(f"Node {node_name} already logged as isolated; skipping node_down alarm.")
                        continue
                    alarm_id = alarm.get("ID")
                    
                    ne_time = alarm.get("NE_TIME")
                        # Mark the node down in the global physical graph via its IP.
                    mark_node_down_by_name(node_name)

                    start_node = get_block_from_ip(node_ip).upper()
                    if not start_node:
                        logging.warning(f"Could not find block name for IP {node_ip} from network_element table.")  
                        continue
                    logging.info(f"Block Node found from the db: {start_node}")
                    current_isolated_nodes = set()
                    if start_node == node_name:
                        logging.info(f"The Block Node itself got down...Isolating all nodes in the block")
                        # Query location_data to get all nodes (GPs) in the block
                        query = """
                            SELECT NAME FROM location_data 
                            WHERE BLOCK_NAME = %s AND LOCATION_TYPE = 'GP'
                        """
                        cursor.execute(query, (start_node,))
                        mydb.commit()
                        block_nodes = cursor.fetchall()
                        
                        if block_nodes:
                            for row in block_nodes:
                                gp_name = row['NAME']
                                if gp_name in persistent_physical_graph.nodes():
                                    current_isolated_nodes.add(gp_name)
                                    logging.info(f"Added GP node {gp_name} from block {start_node} to isolated nodes")
                        else:
                            logging.warning(f"No GP nodes found in location_data for block {start_node}")
                    elif start_node in persistent_physical_graph.nodes:
                        logging.info(f"Start node {start_node} found in the graph")
                        # Get all nodes connected to start_node (all possible nodes we could reach)
                        connected_nodes = set(nx.node_connected_component(persistent_physical_graph, start_node))
                        
                        # Get nodes actually reachable from start_node
                        reachable = set(nx.single_source_shortest_path_length(persistent_physical_graph, start_node).keys())
                        
                        # Isolated nodes are those that should be connected but aren't reachable
                        current_isolated_nodes = connected_nodes - reachable
                        
                        logging.info(f"Connected nodes: {connected_nodes}")
                        logging.info(f"Reachable nodes: {reachable}")
                        logging.info(f"Isolated nodes: {current_isolated_nodes}")
                    else:
                        logging.info(f"The block node {start_node} is not there in the graph")
                        current_isolated_nodes = set()
                        continue
                    logging.info(f"Total Current Isolated Nodes: {current_isolated_nodes}")
                    logging.info(f"Total Logged Isolated Nodes: {logged_isolated_nodes}")
                    new_isolated = current_isolated_nodes - logged_isolated_nodes
                    logging.info(f"New Isolated Nodes after the current alarm: {new_isolated}")
                    child_alarms =  []
                    if new_isolated:
                        for node in new_isolated:
                            ip = persistent_physical_graph.nodes[node].get('ip')
                            dummy_id=fetch_alarm_id_by_ip_and_time(ip, ne_time)
                            if dummy_id:
                                child_alarms.append(dummy_id)
                                insert_child_alarm_mapping(alarm_id, dummy_id)
                            log_msg = (
                            "\n[ALARM ALERT] New isolated nodes detected due to node_not_reachable alarms:\n"
                            f"    Root Cause Alarm IDs : {alarm_id}\n"
                            f"    Isolated Nodes       : {', '.join(new_isolated)}\n"
                            f"    Child Alarm IDs      : {', '.join(child_alarms)}\n"
                            "---------------------------------------------------------------------"
                        )
                        logging.info(log_msg)
                        root_cause_mapping[alarm_id] = {
                            'ids': set(child_alarms),
                            'root node_not_reachable': node_name,
                            'isolated_nodes': list(new_isolated)
                        }
                        print(f"New isolated nodes due to node_not_reachable alarms {alarm_id}: {new_isolated}")
                        logged_isolated_nodes.update(new_isolated)
                    else:
                        logging.info(f"No new isolated nodes detected after node_not_reachable alarms {alarm_id}.")
                print(f"Root cause mapping: {root_cause_mapping}")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No new alarms.")
            print(f"Root cause mapping: {root_cause_mapping}")
        
        # Log summary of parent alarm IDs, link down connections, and children alarm IDs
        logging.info("="*80)
        logging.info(f"Root Cause Mapping Dictionary: {root_cause_mapping}")
        logging.info("Summary of Alarm Mapping after one iteration:")
        for parent_key, mapping in root_cause_mapping.items():
                if "root node_not_reachable" in mapping:
                    logging.info("============ Root Node Not Reachable Alarm Mapping ============")
                    logging.info(f"Root Alarm ID: {parent_key}")
                    logging.info("Root node_not_reachable: %s" % mapping.get("root nod_not_reachable"))
                    logging.info("Isolated Nodes: %s" % ", ".join(mapping.get("isolated_nodes", [])))
                    logging.info("Child Alarm IDs: %s" % ", ".join(str(cid) for cid in mapping.get("ids", [])))
                    logging.info("==============================================================")
                else:
                    parent_alarm_ids = parent_key  # Parent alarm key is built from the parent alarm IDs
                    link_down_connections = mapping.get("root_cause_names", [])
                    child_alarm_ids = mapping.get("ids", set())
                    
                    logging.info(f"Parent Alarm IDs: {parent_alarm_ids}")
                    logging.info(f"Link Down Connections: {', '.join(link_down_connections) if link_down_connections else 'N/A'}")
                    logging.info(f"Child Alarm IDs: {', '.join(str(cid) for cid in child_alarm_ids) if child_alarm_ids else 'N/A'}")
                logging.info("="*80)
        logging.info(f"node_edges_backup Dictionary: {node_edges_backup}")
        logging.info(f"One iteration completed at {datetime.now()}")
        print("One iteration completed at", datetime.now())
        time.sleep(300)
    except Exception as e:
        logging.exception("Unhandled exception in main loop: %s", e)
        time.sleep(60)

