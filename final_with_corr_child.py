import time
import configparser
import mysql.connector
from datetime import datetime
from collections import defaultdict
from itertools import combinations
import logging
import networkx as nx
import uuid

root_cause_mapping = {}
node_edges_backup = {}  # Store edges when removing nodes

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("final.log", mode="a"),
        logging.StreamHandler()
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
cursor = mydb.cursor(dictionary=True)

if 'persistent_physical_graph' not in globals():
    query_phys_topo = "SELECT aendname, bendname, aendip, bendip,block_name FROM topology_data_logical"
    cursor.execute(query_phys_topo)
    phys_rows = cursor.fetchall()
    persistent_physical_graph = nx.Graph()
    for row in phys_rows:
        aend = row['aendname']
        bend = row['bendname']
        aendip = row.get('aendip')
        bendip = row.get('bendip')
        # Add nodes with an 'ip' attribute so we can later locate them by IP.
        persistent_physical_graph.add_node(aend, ip=aendip)
        persistent_physical_graph.add_node(bend, ip=bendip)
        persistent_physical_graph.add_edge(aend, bend)
    print("Global physical graph constructed.")

if 'logged_isolated_nodes' not in globals():
  logged_isolated_nodes = set()
  print(f"Logged isolated initially: {logged_isolated_nodes}")
last_processed_time = datetime.strptime("2025-02-10 15:45:00", "%Y-%m-%d %H:%M:%S")

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
  result_count = cursor.fetchone()
  count = result_count["logical_ring_count"]
  return count

def get_physical_connection(physicalsegment):

    query_phys = """
      SELECT aendname, bendname FROM topology_data_physical
      WHERE ringname = %s AND segmentid = %s
    """
    cursor.execute(query_phys, (physicalring, physicalsegment))
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

def mark_node_down_by_ip(node_ip):
    """Store node's edges before removing it."""
    for node, data in list(persistent_physical_graph.nodes(data=True)):
        if data.get("ip") == node_ip:
            # Backup edges before removing node
            node_edges_backup[node] = list(persistent_physical_graph.edges(node))
            persistent_physical_graph.remove_node(node)
            return node
    return None

def get_block_from_ip(ip):
    """
    Fetch the block name from network_element table based on IP address.
    Returns the block name or 'BERLA' as fallback.
    """
    query = """
        SELECT locationText
        FROM network_element
        WHERE ip = %s
    """
    cursor.execute(query, (ip,))
    result = cursor.fetchone()
    if result and result['locationText']:
        location_text = result['locationText']
        
        # Ensure 'Block :' exists before splitting
        if "Block :" in location_text:
            try:
                # Extract the block name
                block_part = location_text.split("Block :")[1]
                block_name = block_part.split(";")[0].strip()
                print(f"Block from db: {block_name}")  # Ensure we get only the block name
                return block_name
            except IndexError:
                logging.warning(f"Could not parse block from locationText for IP {ip}")
        else:
            logging.warning(f"Block information not found in locationText for IP {ip}")

    logging.warning(f"Using fallback block 'BERLA' for IP {ip}")
    return "BERLA"   

def insert_dummy_node_not_reachable_alarm(obj_name):

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Create dummy IDs (using uuid for uniqueness)
    dummy_id = f"DUMMY_{uuid.uuid4()}"
    dummy_notif_id = dummy_id

    # Prepare the SQL insert.
    query = """
    INSERT INTO alarm
      (ID, NOTIF_ID, NE_TIME, OBJ_NAME, OBJ_TYPE, RES_NAME, EMS_TIME,
       PROB_CAUSE, PERC_SEVERITY, NMS_SEVERITY, LOG_TIME, LOCATION_ID,
       PROCESS_FLAG, REMARKS, CATEGORY)
    VALUES
      (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    # Set dummy values as appropriate.
    params = (
        dummy_id,               # ID
        dummy_notif_id,         # NOTIF_ID
        now,                    # NE_TIME
        obj_name,               # OBJ_NAME
        "GP_ROUTER",            # OBJ_TYPE (example value)
        obj_name,               # RES_NAME (using obj_name as resource name)
        now,                    # EMS_TIME
        "node_not_reachable",   # PROB_CAUSE
        1,                      # PERC_SEVERITY (example)
        1,                      # NMS_SEVERITY (example)
        now,                    # LOG_TIME
        "0",                    # LOCATION_ID (example)
        "3",                    # PROCESS_FLAG (example)
        "Dummy alarm for testing device down",  # REMARKS
        "EQUIPMENT ALARMS"      # CATEGORY
    )
    cursor.execute(query, params)
    mydb.commit()
    print(f"Inserted dummy node_not_reachable alarm for {obj_name} with ID {dummy_id}.")
    return dummy_id

while True:
    # First check alarm_hist_dummy table for any new entries
    query_hist = """
        SELECT * FROM alarm_hist_dummy 
        WHERE NE_TIME > %s 
        ORDER BY NE_TIME
    """
    cursor.execute(query_hist, (last_processed_time,))
    fixed_results = cursor.fetchall()
    print(f"Fixed results: {fixed_results}")
    # Group fixed alarms by NE_TIME and type
    fixed_groups = defaultdict(list)
    for row in fixed_results:
        prob_cause = row.get("PROB_CAUSE", "").lower()
        ne_time = row["NE_TIME"]
        if prob_cause == "link_down":
            fixed_groups[ne_time].append(("link_down", row))
        elif prob_cause == "node_not_reachable":
            print(f"Node not reachable alarm: {row}")
            fixed_groups[ne_time].append(("node_not_reachable", row))
    
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
                    
                    # Restore node and its edges in the graph
                    if node_ip in logged_isolated_nodes:
                        logging.info(f"Restoring node {node_ip} in the graph")
                        logged_isolated_nodes.remove(node_ip)
                        persistent_physical_graph.add_node(node_ip)
                        
                        # Restore backed up edges
                        if node_ip in node_edges_backup:
                            for edge in node_edges_backup[node_ip]:
                                if edge[1] in persistent_physical_graph:  # Only restore if other endpoint exists
                                    persistent_physical_graph.add_edge(*edge)
                            del node_edges_backup[node_ip]
                        
                        logging.info(f"Fixed node_not_reachable alarm for node {node_ip} (Alarm ID: {alarm_id})")
            
            # Process fixed link_down alarms
            link_down_group = [alarm for (atype, alarm) in group if atype == "link_down"]
            if link_down_group:
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
                        # Find nodes by their IP attributes
                        aend, bend = connection_name.split('-')  # Use the connection name which has the correct node names
                        
                        # Restore edge in the graph if both endpoints exist
                        if aend in persistent_physical_graph.nodes() and bend in persistent_physical_graph.nodes():
                            persistent_physical_graph.add_edge(aend, bend)
                            logging.info(f"Restored edge in graph: {aend} <--> {bend}")
                        
                        # First log connection details
                        log_message = (
                            f"Fixed link_down detected:\n"
                            f"   Alarm IDs: {id1} and {id2}\n"
                            f"   Endpoints: {ip1}:{ifIndex1} <--> {ip2}:{ifIndex2}\n"
                            f"   Connection Name: {connection_name}\n"
                            f"   LR Ring: {lr_ring}\n"
                            f"   Physical Ring: {curr_physicalring}\n"
                            f"   Physical Segments: {physicalsegments}\n"
                            f"   Block: {block}\n"
                            f"   District: {district}\n"
                            f"   Graph updated: Edge restored\n"
                            "----------------------------------------"
                        )
                        logging.info(log_message)
                        
                        # Then check and log RCA details
                        root_cause_key = f"{min(id1, id2)}_{max(id1, id2)}"
                        if root_cause_key in root_cause_mapping:
                            child_alarm_ids = root_cause_mapping[root_cause_key]
                            fixed_node_alarms = {alarm.get("ID") for alarm in node_down_group}
                            
                            fixed_children = child_alarm_ids & fixed_node_alarms
                            remaining_children = child_alarm_ids - fixed_node_alarms
                            
                            if remaining_children:
                                logging.warning(
                                    f"Root cause analysis for {connection_name}:\n"
                                    f"    Root cause alarms ({root_cause_key}) fixed but some child alarms remain:\n"
                                    f"    Fixed children: {', '.join(map(str, fixed_children))}\n"
                                    f"    Remaining children: {', '.join(map(str, remaining_children))}\n"
                                    f"    Possible power failure for remaining nodes\n"
                                    "----------------------------------------"
                                )
                            else:
                                logging.info(
                                    f"Root cause analysis for {connection_name}:\n"
                                    f"    Root cause alarms ({root_cause_key}) and all child alarms resolved:\n"
                                    f"    All children fixed: {', '.join(map(str, child_alarm_ids))}\n"
                                    "----------------------------------------"
                                )
                            
                            del root_cause_mapping[root_cause_key]
                        
                        log_message = (
                            f"Fixed link_down detected:\n"
                            f"   Alarm IDs: {id1} and {id2}\n"
                            f"   Endpoints: {ip1}:{ifIndex1} <--> {ip2}:{ifIndex2}\n"
                            f"   Connection Name: {connection_name}\n"
                            f"   LR Ring: {lr_ring}\n"
                            f"   Physical Ring: {curr_physicalring}\n"
                            f"   Physical Segments: {physicalsegments}\n"
                            f"   Block: {block}\n"
                            f"   District: {district}\n"
                            f"   Graph updated: Edge restored\n"
                            "----------------------------------------"
                        )
                        logging.info(log_message)
                        
                        lr_rings_in_group.append(lr_ring)
                        physical_segments_in_group.append(physicalsegments)
                        if physicalring is None:
                            physicalring = curr_physicalring
                
                # Check if this was a physical cut that got fixed
                distinct_rings = set(lr_rings_in_group)
                if physicalring:
                    total_lrings = get_num_lrings(physicalring)
                    if len(distinct_rings) == total_lrings:
                        unique_physical_segment = get_common_segments(physical_segments_in_group)
                        if unique_physical_segment:
                            physical_cut = get_physical_connection(unique_physical_segment.pop())
                            logging.info(
                                f"All the fixed alarms at {ne_time} indicate a restored OFC.\n"
                                f"Connection: {physical_cut} in Physical Ring: {physicalring}"
                            )

    # Then proceed with checking active alarms (existing code)
    query = "SELECT * FROM alarm WHERE NE_TIME > %s ORDER BY NE_TIME"
    cursor.execute(query, (last_processed_time,))
    results = cursor.fetchall()
    
    # Group link_down alarms by NE_TIME
    all_groups = defaultdict(list)
    for row in results:
        prob_cause = row.get("PROB_CAUSE", "").lower()
        ne_time = row["NE_TIME"]
        if prob_cause == "link_down":
            all_groups[ne_time].append(("link_down",row))
        elif prob_cause == "node_not_reachable":
          all_groups[ne_time].append(("node_not_reachable", row))
    
    if results:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] New alarms:")
        for ne_time in sorted(all_groups.keys()):
          group = all_groups[ne_time]
          logging.info(f"Processing NE_TIME group: {ne_time}")
          print(f"\nNE_TIME Group: {ne_time}")
          node_down_group = [alarm for (atype, alarm) in group if atype == "node_not_reachable"]
          
          if node_down_group:
            alarm_ids_causing_down = []
            for alarm in node_down_group:
              
              logging.info(alarm)
              node_ip = alarm.get("OBJ_NAME")
              alarm_id = alarm.get("ID")
              alarm_ids_causing_down.append(alarm_id)
                # Mark the node down in the global physical graph via its IP.
              mark_node_down_by_ip(node_ip)
              start_node = get_block_from_ip(node_ip).upper()
            # After processing these alarms, check for new isolated nodes.
            if start_node in persistent_physical_graph.nodes:
                reachable = set(nx.single_source_shortest_path_length(persistent_physical_graph, start_node).keys())
                current_isolated_nodes = set(persistent_physical_graph.nodes()) - reachable
                print(f"Isolated nodes after node down: {current_isolated_nodes}")
            else:
                current_isolated_nodes = set(persistent_physical_graph.nodes())
            print(current_isolated_nodes)
            print(logged_isolated_nodes)    
            new_isolated = current_isolated_nodes - logged_isolated_nodes
            child_alarms =  []
            if new_isolated:
                for node in new_isolated:
                   ip = persistent_physical_graph.nodes[node].get('ip')
                   dummy_id=insert_dummy_node_not_reachable_alarm(ip)
                   child_alarms.append(dummy_id) 
                   logging.info("")
                log_msg = (
                    "\n[ALARM ALERT] New isolated nodes detected due to node_not_reachable alarms:\n"
                    f"    Root Cause Alarm IDs : {', '.join(str(aid) for aid in alarm_ids_causing_down)}\n"
                    f"    Isolated Nodes       : {', '.join(new_isolated)}\n"
                    f"    Child Alarm IDs      : {', '.join(child_alarms)}\n"
                    "---------------------------------------------------------------------"
                )
                logging.info(log_msg)
                print(f"New isolated nodes due to node_not_reachable alarms {alarm_ids_causing_down}: {new_isolated}")
                logged_isolated_nodes.update(new_isolated)
            else:
                logging.info(f"No new isolated nodes detected after node_not_reachable alarms {alarm_ids_causing_down}.")
          
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
                print(f"ip1: {ip1}, ifIndex1: {ifIndex1}, ip2: {ip2}, ifIndex2: {ifIndex2}")
                if not (ip1 and ifIndex1 and ip2 and ifIndex2):
                    continue
                
                query_conn = """
                    SELECT physicalringname, aendname, bendname, lrname, physicalsegments, district_name, block_name FROM topology_data_logical
                    WHERE (aendip = %s AND aendifIndex = %s AND bendip = %s AND bendifIndex = %s)
                      OR (aendip = %s AND aendifIndex = %s AND bendip = %s AND bendifIndex = %s)
                """
                cursor.execute(query_conn, (ip1, ifIndex1, ip2, ifIndex2, ip2, ifIndex2, ip1, ifIndex1))
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
                    
                    print(f"Start node: {start_node}")
                    if start_node in persistent_physical_graph.nodes:
                        reachable = set(nx.single_source_shortest_path_length(persistent_physical_graph, start_node).keys())
                        print(f"Reachable Nodes: {reachable}")
                        current_isolated_nodes = set(persistent_physical_graph.nodes()) - reachable

                    else:
                        print("Could not find start node in graph")
                        current_isolated_nodes = set(persistent_physical_graph.nodes())
                    new_isolated = current_isolated_nodes - logged_isolated_nodes
                    child_alarms = []
                    if new_isolated:
                        for node in new_isolated:
                            ip = persistent_physical_graph.nodes[node].get('ip')
                            dummy_id = insert_dummy_node_not_reachable_alarm(ip)
                            child_alarms.append(dummy_id)
                            print("Inserted alarm")
                        
                        # Store root cause and child alarm mapping
                        root_cause_key = f"{min(id1, id2)}_{max(id1, id2)}"  # Consistent ordering
                        root_cause_mapping[root_cause_key] = set(child_alarms)
                        
                        root_cause_alarm_ids = f"{id1} and {id2}"
                        logging.info(f"Isolated nodes after logical cut: {new_isolated}\n"
                                    f"Root Cause alarms: {root_cause_alarm_ids}\n"
                                    f"Child alarm IDs: {', '.join(map(str, child_alarms))}\n"
                                    f"Added to root cause mapping with key: {root_cause_key}")
                        
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
                    
                    print(f"❌ No connection found between {ip1}:{ifIndex1} and {ip2}:{ifIndex2}")



            distinct_rings = set(lr_rings_in_group)
            total_lrings  = get_num_lrings(physicalring)
            print(f"Physical Ring: {physicalring}")
            print(total_lrings)
            print(distinct_rings)
            if len(distinct_rings)==total_lrings:
              logging.info(
                f"All logical cuts cover all logical rings in the physical ring...Might be a physical cut"
              )
              print(f"Segments: {physcical_segments_in_group}")
              unique_physical_segment = get_common_segments(physcical_segments_in_group)
              print(f"Unique Segment: {unique_physical_segment}")
              print(f"Physical Segments: {physcical_segments_in_group}")
              if unique_physical_segment:
                physical_cut = get_physical_connection(unique_physical_segment.pop())
                print(f"Physical cut: {physical_cut}")
                logging.info(
                      f"All the above alarms at {ne_time} have occurred due to an OFC.\n"
                      f"Connection: {physical_cut} in Physical Ring: {physicalring}"
                    )
            else:
              logging.info("The logical cuts does not cover all logical rings...cannot be a physical cut")
            if len(distinct_rings) > 1:
              print("⚠️ Logical cuts detected in different rings:", distinct_rings)
            elif len(distinct_rings) == 1 and distinct_rings:
              ring = distinct_rings.pop()
              print("All logical cuts in this group belong to the same ring:", ring)
            else:
              print("No logical connections were found in this group.")    
        # Update last_processed_time to the latest NE_TIME from the results
        print(f"Root cause mapping: {root_cause_mapping}")
        last_processed_time = datetime.now()
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No new alarms.")
    
    time.sleep(300)