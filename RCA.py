import uuid
import stomp
from datetime import datetime
import pandas as pd
import random
import mysql.connector
import logging
from logging.handlers import RotatingFileHandler
import json
from collections import defaultdict
from datetime import datetime
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

HOST = "localhost"
PORT = 61613
USERNAME = "admin"
PASSWORD = "admin"
TOPIC_NAME = "/topic/karthick"

def publish_link_down_alarms(connection_str):
    """
    Publishes two alarm messages to a STOMP topic for a given connection string.
    Each alarm is generated based on the input connection.
    
    :param connection_str: A string in the format 'ip1-ip2' indicating the connection that went down.
    """
    connection_str = connection_str.lower()
    input_ips = connection_str.split('-')
    if len(input_ips) != 2:
        raise ValueError("Connection string must be in the format 'ip1-ip2'.")
    
    # STOMP Connection
    conn = stomp.Connection([(HOST, PORT)])
    conn.connect(USERNAME, PASSWORD, wait=True)
    
    # Generate alarms for each IP
    for index, ip in enumerate(input_ips):
        obj_type = "BLOCK_ROUTER" if ip == "10.128.0.16" else "GP_ROUTER"
        current_time = datetime.now()
        alarm_id = f"LIN_{current_time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:12]}"
        notif_id = f"RAI_{alarm_id}"
        ifIndex = 14 - index
        addi_info = f"IF_DESCR=GigabitEthernet0/0/{ifIndex};IF_TYPE=6;REASON=administratively down;ENGINE_ID=80:00:00:09:03:00:f8:0f:6f:05:91:80;SYS_TIME=8000"
        
        
        # Construct alarm message
        alarm_message = {
            "ID": alarm_id,
            "NOTIF_ID": notif_id,
            "NE_TIME": current_time.isoformat(),
            "OBJ_NAME": ip,
            "OBJ_TYPE": obj_type,
            "RES_NAME": ip,
            "EMS_TIME": current_time.isoformat(),
            "PROB_CAUSE": "link_down",
            "PERC_SEVERITY": 1,
            "NMS_SEVERITY": 1,
            "ADDI_INFO": addi_info,
            "RCA_INDICATOR": 1,
            "RCA_ID": None,
            "VISIBLE": 0,
            "TT_ID": None,
            "LOG_TIME": current_time.isoformat(),
            "LOCATION_ID": 1195,
            "PROCESS_FLAG": 2,
            "REMARKS": "Interface Admin Status is DOWN",
            "CATEGORY": "LINK ALARMS",
            "interface": ifIndex,
            "RFO": None,
            "FIBER_DETAILS": None,
            "LATEST_RFO": None,
            "PREVIOUS_TT_ID": None,
            "PREVIOUS_RCA_ID": None,
            "UPS_BATTERY_PERCENT": None
        }
        
        # Send message
        conn.send(destination=TOPIC_NAME, body=str(alarm_message), headers={})
        print("Message sent:", alarm_message)
    
    conn.disconnect()

def get_random_aendip_bendip(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Select a random row from the DataFrame
    random_row = df.sample(n=1).iloc[0]
    
    # Extract the aendip and bendip from the random row
    aendip = random_row['aendip']
    bendip = random_row['bendip']
    
    # Return the aendip-bendip pair as a formatted string
    return f"{aendip}-{bendip}"

# Example usage:
csv_file_path = 'BERLA_Blue.csv'  # Replace with the path to your CSV file
#connection_str = get_random_aendip_bendip(csv_file_path)

for i in range(3):
    connection_str = get_random_aendip_bendip('BERLA_Blue.csv')
    print(connection_str)
    publish_link_down_alarms(connection_str)

db_config = {
    "host": "192.168.30.15",
    "user": "nms",
    "password": "Nms@1234",
    "database": "cnmsip"
}
mydb = mysql.connector.connect(**db_config)
cursor = mydb.cursor(dictionary=True)

# Define last processed timestamp
last_processed = datetime.strptime("2025-02-05 16:55:00", "%Y-%m-%d %H:%M:%S")


alarms = []
try:
    with open("received_messages.json", "r") as file:
        for line in file:
            data = json.loads(line.strip())  # Load full JSON array
            try:
                raw_message = data["message"]
                    
                    # Convert single quotes to double quotes and None to null
                formatted_message = (
                    raw_message.replace("'", '"')
                    .replace(" None", " null")
                )
                    
                    # Parse the formatted JSON string inside "message"
                alarm_data = json.loads(formatted_message)
                alarms.append(alarm_data)

            except json.JSONDecodeError as e:
                print(f"Skipping malformed entry: {raw_message} | Error: {e}")

except FileNotFoundError:
    print("No message file found.")

# Filter new alarms based on 'PROB_CAUSE' and 'NE_TIME'
for i in range(len(alarms)):
    new_alarms = [
        alarm for alarm in alarms
        if alarm.get("PROB_CAUSE") == "link_down" and
        datetime.strptime(alarm["NE_TIME"], "%Y-%m-%dT%H:%M:%S.%f") > last_processed
    ]

if new_alarms:
    # Group alarms by NE_TIME
    groups = defaultdict(list)
    for alarm in new_alarms:
        groups[alarm["NE_TIME"]].append(alarm)

    # Store grouped alarms in a variable for further processing
    grouped_alarms = {ne_time: group for ne_time, group in groups.items()}

    # Print alarms
    for ne_time, group in grouped_alarms.items():
        print(f"Processing alarms for NE_TIME: {ne_time}")
        for alarm in group:
            print(alarm)

    # Update last processed time
    last_processed = max(datetime.strptime(a["NE_TIME"], "%Y-%m-%dT%H:%M:%S.%f") for a in new_alarms)
else:
    print("No new alarms")

# Configure rotating log handler (5MB per file, keep 3 backups)
handler = RotatingFileHandler("app.log", maxBytes=5*1024*1024, backupCount=3)

logging.basicConfig(
    handlers=[handler],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
for i in grouped_alarms:
    logging.info(f"{grouped_alarms[i]}")

def extract_ip_ifindex(grouped_alarms):
    extracted_data = []
    
    for ne_time, group in grouped_alarms.items():
        for alarm in group:
            ip = alarm.get("OBJ_NAME")
            ifIndex = alarm.get("interface")

            if ip and ifIndex:
                extracted_data.append((ip, ifIndex))

    return extracted_data

# Example usage
extracted_values = extract_ip_ifindex(grouped_alarms)
print("Extracted IPs and ifIndexes:", extracted_values)

def check_connection(ip_ifindex_pairs):
        
        # Extract IPs and ifIndexes
        if len(ip_ifindex_pairs) %2 != 0:
            print("❌ Invalid input: Need in multiple of 2 IP-ifIndex pairs.")
            return False

        results = []
        names = []
        lrResult = []
        for i in range(0, len(ip_ifindex_pairs), 2):

            (ip1, ifIndex1), (ip2, ifIndex2) = ip_ifindex_pairs[i], ip_ifindex_pairs[i+1]

            # Query to check connection in both directions
            query = """
                SELECT aendname, bendname, lrname FROM topology_data_logical
                WHERE (aendip = %s AND aendifIndex = %s AND bendip = %s AND bendifIndex = %s)
                OR (aendip = %s AND aendifIndex = %s AND bendip = %s AND bendifIndex = %s)
            """
            
            cursor.execute(query, (ip1, ifIndex1, ip2, ifIndex2, ip2, ifIndex2, ip1, ifIndex1))
            result = cursor.fetchall()
            print(result)


            if result:
                print(f"✅ Connection exists between {ip1}:{ifIndex1} and {ip2}:{ifIndex2}")
                results.append((ip1, ifIndex1, ip2, ifIndex2, True))
                names.append(f"{result[0]['aendname']}-{result[0]['bendname']}")
                lrResult.append(result[0]['lrname'])
            else:
                print(f"❌ No connection found between {ip1}:{ifIndex1} and {ip2}:{ifIndex2}")
                results.append((ip1, ifIndex1, ip2, ifIndex2, False))
            
        return lrResult,names

# Example usage
lr_rings,namess = check_connection(extracted_values)

print(namess)
print(lr_rings)

logging.info(f"logical cuts at nodes: {namess}")
logging.info(f"logical cuts at rings: {lr_rings}")

def find_common_physical_connections_from_files(logical_connections_file, physical_connections_file, input_connections):
    """
    Finds common physical connections based on shared segments in logical connections,
    reading data from CSV files.

    Args:
        logical_connections_file: Path to the logical connections CSV file.
        physical_connections_file: Path to the physical connections CSV file.
        input_connections: A list of strings representing logical connections (e.g., ['Kumhi-Berla', 'Kharra-Bharda']).

    Returns:
        A dictionary where keys are the common segment IDs and values are dictionaries
        containing the corresponding physical connection data or None if no common segment found.
    """

    # Load the data from CSV files
    logical_df = pd.read_csv(logical_connections_file)
    physical_df = pd.read_csv(physical_connections_file)
    
    common_physical_connections = {}
    all_physical_segments = None
    
    for input_connection in input_connections:
         aend_name, bend_name = input_connection.split('-')
         print(aend_name)
         # Find the logical connection
         logical_connection = logical_df[((logical_df['aendname'] == aend_name) & (logical_df['bendname'] == bend_name)) |
                                          ((logical_df['aendname'] == bend_name) & (logical_df['bendname'] == aend_name))]
         
         if not logical_connection.empty:
             physical_segments_str = logical_connection['physicalsegments'].iloc[0]
          
             if isinstance(physical_segments_str, str) and physical_segments_str: #check if not empty and is not Nan value
                 physical_segments = set(int(segment.strip()) for segment in physical_segments_str.split(','))  # Convert to set

                 print(physical_segments)
                 if all_physical_segments is None:
                    all_physical_segments = physical_segments  # Initialize with the first set
                 else:
                    all_physical_segments &= physical_segments      
                 
    
    # all_physical_segments = set(all_physical_segments)
    

    if not all_physical_segments:
        return None
    
    # Find matching physical segments
    for segment_id in all_physical_segments:
         physical_connection = physical_df[physical_df['segmentid'] == segment_id]
         if not physical_connection.empty:
            common_physical_connections[segment_id] = physical_connection.to_dict(orient='records')[0]
         else:
            common_physical_connections[segment_id] = None
    return common_physical_connections


result = None

if len(lr_rings)==3:
    if lr_rings[0] != lr_rings[1] != lr_rings[2]:
        
        
        logical_connections_file = 'BERLA_Blue.csv'
        data = pd.read_csv(logical_connections_file)  # Replace with your file path
        physical_connections_file = 'topology_data_physical.csv'  # Replace with your file path



        # List of fiber cuts

        common_connections = find_common_physical_connections_from_files(
            logical_connections_file, physical_connections_file, namess
        )

        if common_connections:
            segment_key = next(iter(common_connections))  # Gets the first key (127)
            segment_info = common_connections[segment_key]

        # Fetch and format "aendname-bendname"
            result = f"{segment_info['aendname']}-{segment_info['bendname']}"
            for segment_id, connection in common_connections.items():
                if connection:
                    print(f"Common segment ID: {segment_id}")
                    print("Physical Connection Details:")
                    for key, value in connection.items():
                        print(f"  {key}: {value}")
                        print("-" * 30)
                else:
                    print(f"Common segment ID: {segment_id}")
                    print("No Physical Connection Details found")
        else:
            print("No physical cut.")
    else:
        None


logging.info(f"physical cut result: {result}")

if result:
    logging.info("This phy cut is the RCA for the isolated nodes ")

class NetworkAnalyzer:
    def __init__(self, filepath):
        # Load CSV data
        if os.path.exists(filepath):
            self.df = pd.read_csv(filepath)
        else:
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Build the original graph using friendly node names.
        # Assumes CSV has columns 'aendname' and 'bendname'
        self.original_graph = nx.Graph()
        for _, row in self.df.iterrows():
            aend = row['aendname']
            bend = row['bendname']
            self.original_graph.add_edge(aend, bend)
        
        # Records for failures
        self.applied_cuts = []    # fiber cuts (edges removed)
        self.down_nodes = set()   # nodes marked as down

    def apply_cut(self, cut_aend, cut_bend):
        """Simulate a fiber cut (edge removal) and record it."""
        if not self.original_graph.has_edge(cut_aend, cut_bend):
            print(f"Warning: Edge {cut_aend}-{cut_bend} does not exist in the network.")
            return False
        
        # Avoid duplicate cuts (the graph is undirected)
        if (cut_aend, cut_bend) not in self.applied_cuts and (cut_bend, cut_aend) not in self.applied_cuts:
            self.applied_cuts.append((cut_aend, cut_bend))
            print(f"Edge {cut_aend}-{cut_bend} has been cut.")
            return True
        else:
            print(f"Warning: Edge {cut_aend}-{cut_bend} has already been cut.")
            return False

    def mark_node_down(self, node_name):
        """Mark a node as down. A down node is considered nonfunctional and will be removed from the working graph."""
        if node_name in self.original_graph.nodes:
            self.down_nodes.add(node_name)
            # Optionally mark the node in the graph for visualization.
            self.original_graph.nodes[node_name]['down'] = True
            print(f"Node {node_name} marked as down.")
        else:
            print(f"Node {node_name} not found in the network.")
    
    def check_isolated(self, start_node='BERLA'):
        """
        Check which nodes (that are not down) become isolated from the start node
        after applying fiber cuts and marking nodes as down.
        """
        # Create a working copy of the network graph.
        G_work = self.original_graph.copy()
        
        # Remove fiber cuts (simulate failed edges)
        for a, b in self.applied_cuts:
            if G_work.has_edge(a, b):
                G_work.remove_edge(a, b)
        
        # Remove nodes that are down.
        G_work.remove_nodes_from(self.down_nodes)
        
        # Check if the start node is still present.
        if start_node not in G_work.nodes:
            print(f"Start node '{start_node}' is not in the network after removals.")
            return [f"Start node '{start_node}' not found in the pruned network."]
        
        # Compute the reachable nodes from the start node.
        try:
            reachable = set(nx.single_source_shortest_path_length(G_work, start_node).keys())
        except nx.NetworkXError as e:
            print("Error during traversal:", e)
            reachable = set()
        
        # Isolated nodes are those (that are not down) and are not reachable.
        non_down_nodes = set(n for n in self.original_graph.nodes if n not in self.down_nodes)
        isolated_nodes = list(non_down_nodes - reachable)
        return isolated_nodes if isolated_nodes else None

    def display_graph(self):
        """
        Visualize the original graph.
        Down nodes will be colored black; all other nodes will be light blue.
        Note: The visualization does not remove the fiber cuts.
        """
        node_colors = []
        for node in self.original_graph.nodes:
            if node in self.down_nodes or self.original_graph.nodes[node].get('down', False):
                node_colors.append('black')
            else:
                node_colors.append('lightblue')
        
        plt.figure(figsize=(12, 12))
        nx.draw(self.original_graph, with_labels=True, node_color=node_colors,
                font_weight='bold', node_size=500)
        plt.title("Network Graph (Down Nodes in Black)")
        plt.show()





filepath = "BERLA_Blue.csv"
analyzer = NetworkAnalyzer(filepath)


input_connections = namess
for connection in input_connections:
    aend, bend = connection.split('-')
    analyzer.apply_cut(aend, bend)


# --- Check for Isolation ---
# This will determine which nodes (that are not down) are isolated from "BERLA"
isolated_nodes = analyzer.check_isolated(start_node='BERLA')
print("Isolated nodes:", isolated_nodes)

logging.info(f"isolated nodes: {isolated_nodes}")

if isolated_nodes:
    logging.info(f"Root cause of the isolated nodes: {result}")








