import time
import configparser
import mysql.connector
from datetime import datetime
from collections import defaultdict
from itertools import combinations
import logging
import networkx as nx



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
    query_phys_topo = "SELECT aendname, bendname FROM topology_data_logical"
    cursor.execute(query_phys_topo)
    phys_rows = cursor.fetchall()
    persistent_physical_graph = nx.Graph()
    for row in phys_rows:
        persistent_physical_graph.add_edge(row['aendname'], row['bendname'])
    print("Global physical graph constructed.")

if 'logged_isolated_nodes' not in globals():
  logged_isolated_nodes = set()
last_processed_time = datetime.strptime("2025-02-06 09:53:00", "%Y-%m-%d %H:%M:%S")

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


while True:
    query = "SELECT * FROM alarm WHERE NE_TIME > %s ORDER BY NE_TIME"
    cursor.execute(query, (last_processed_time,))
    results = cursor.fetchall()
    
    # Group link_down alarms by NE_TIME
    link_down_groups = defaultdict(list)
    for row in results:
        prob_cause = row.get("PROB_CAUSE", "").lower()
        if prob_cause == "link_down":
            ne_time = row["NE_TIME"]
            link_down_groups[ne_time].append(row)
    
    if results:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] New alarms:")
        
        # Process each group of link_down alarms at the same NE_TIME
        if link_down_groups:
            print("\nLink Down Alarm Groups:")
            for ne_time, group in link_down_groups.items():
                logging.info(f"Processing NE_TIME group: {ne_time}")
                for alarm in group:
                  logging.info(alarm)
                print(f"\nNE_TIME Group: {ne_time}")
                lr_rings_in_group = []
                physcical_segments_in_group = []
                physicalring = None
                # Check all possible combinations in the current group
                for alarm1, alarm2 in combinations(group, 2):
                    ip1, ifIndex1 = alarm1.get("OBJ_NAME"), alarm1.get("interface")
                    ip2, ifIndex2 = alarm2.get("OBJ_NAME"), alarm2.get("interface")
                    id1, id2 = alarm1.get("ID"), alarm2.get("ID")
                    # Continue only if both alarms have valid IP and ifIndex
                    if not (ip1 and ifIndex1 and ip2 and ifIndex2):
                        continue
                    
                    query_conn = """
                        SELECT physicalringname, aendname, bendname, lrname, physicalsegments, district_name, block_name FROM topology_data_logical
                        WHERE (aendip = %s AND aendifIndex = %s AND bendip = %s AND bendifIndex = %s)
                           OR (aendip = %s AND aendifIndex = %s AND bendip = %s AND bendifIndex = %s)
                    """
                    cursor.execute(query_conn, (ip1, ifIndex1, ip2, ifIndex2, ip2, ifIndex2, ip1, ifIndex1))
                    result_conn = cursor.fetchall()
                    
                    connection_name, lr_ring, district, block, curr_physicalring, physicalsegments = get_connection_info(result_conn)
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
                        start_node = 'BERLA'
                        if start_node in persistent_physical_graph.nodes:
                            reachable = set(nx.single_source_shortest_path_length(persistent_physical_graph, start_node).keys())
                            current_isolated_nodes = set(persistent_physical_graph.nodes()) - reachable
                        else:
                            # If the start node is not present, consider all nodes as isolated.
                            current_isolated_nodes = set(persistent_physical_graph.nodes())
                        new_isolated = current_isolated_nodes - logged_isolated_nodes
                        if new_isolated:
                            root_cause_alarm_ids = f"{id1} and {id2}"
                            logging.info(f"Isolated nodes after logical cut: {new_isolated}\n"
                                         f"Root Cause alarms: {root_cause_alarm_ids}")
                            print("Isolated nodes after logical cut:", new_isolated)
                            logged_isolated_nodes.update(new_isolated)
                        else:
                            logging.info("No isolated nodes after the cut.")
                            print("No isolated nodes after physical cut.")
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

        last_processed_time = max(row["NE_TIME"] for row in results)
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No new alarms.")
    
    time.sleep(300)