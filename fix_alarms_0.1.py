import sys
import networkx as nx
from final_with_corr_child import persistent_physical_graph

def fetch_log_by_timestamp(log_file, timestamp):
    """
    Reads the log file line by line and returns all relevant information for the given timestamp.
    """
    log_info = {
        'timestamp_line': '',
        'root_cause': '',
        'child_alarms': '',
        'connection_details': []
    }
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if timestamp in line:
                log_info['timestamp_line'] = line.strip()
                
                # Look for connection details in previous lines
                j = i - 1
                while j >= 0 and j >= i - 10:  # Check up to 10 lines before
                    prev_line = lines[j].strip()
                    if "Due to the below alarms a logical cut has happened:" in prev_line:
                        # Collect all details until current line
                        connection_details = lines[j:i]
                        log_info['connection_details'] = [l.strip() for l in connection_details]
                        break
                    j -= 1
                
                # Check next lines for root cause and child alarms
                if i + 1 < len(lines):
                    root_line = lines[i + 1]
                    if 'Root Cause alarms:' in root_line:
                        log_info['root_cause'] = root_line.strip()
                if i + 2 < len(lines):
                    child_line = lines[i + 2]
                    if 'Child alarm IDs:' in child_line:
                        log_info['child_alarms'] = child_line.strip()
                break
    return log_info

def restore_edge_to_graph(connection_details):
    """
    Restores an edge to the persistent_physical_graph based on connection details
    """
    try:
        # Extract connection name from the details
        for detail in connection_details:
            if "Connection Name:" in detail:
                connection_name = detail.split("Connection Name:")[1].strip()
                aend, bend = connection_name.split('-')
                
                # Add the edge back to the graph
                if not persistent_physical_graph.has_edge(aend, bend):
                    persistent_physical_graph.add_edge(aend, bend)
                    print(f"\nRestored edge {aend}-{bend} to the physical graph")
                    return True
                else:
                    print(f"\nEdge {aend}-{bend} already exists in the graph")
                    return False
        
        print("\nCouldn't find connection name in the details")
        return False
        
    except Exception as e:
        print(f"\nError while restoring edge: {str(e)}")
        return False

if __name__ == "__main__":
    log_file = 'final.log'
    
    # Get timestamp from user input
    timestamp = input("Please enter the timestamp to search for: ")
    
    results = fetch_log_by_timestamp(log_file, timestamp)
    if results['timestamp_line']:
        print("\nFound the following information:")
        
        if results['connection_details']:
            print("\nConnection Details:")
            for detail in results['connection_details']:
                print(detail)
            
            # Try to restore the edge
            restored = restore_edge_to_graph(results['connection_details'])
            if restored:
                print("Successfully restored the connection in the physical graph")
            else:
                print("Failed to restore the connection")
                
        print("\nTimestamp entry:")
        print(results['timestamp_line'])
        
        if results['root_cause']:
            print("\nRoot Cause Alarms:")
            print(results['root_cause'])
            
        if results['child_alarms']:
            print("\nChild Alarms:")
            print(results['child_alarms'])
    else:
        print("No log entries found for the timestamp:", timestamp)
