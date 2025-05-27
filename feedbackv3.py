import sys
import re
import logging
import numpy as np
import configparser
import mysql.connector
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QTableWidget, QTableWidgetItem, QHeaderView, 
                            QSplitter, QMessageBox, QStatusBar, QTabWidget,
                            QTextEdit, QListWidget, QListWidgetItem, QCheckBox)
from PyQt5.QtCore import Qt
import torch
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import MessagePassing

def setup_logging():
    """Configure logging to file and console"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:\n%(message)s\n',
        handlers=[
            logging.FileHandler("ring_visualizer.log", mode="a"),
            logging.StreamHandler()
        ]
    )

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
    return mydb, cursor

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

def identify_block_routers(cursor, topology_graph):
    """Identify block routers and mark them in the graph"""
    query_block_router = """
        SELECT ip 
        FROM network_element 
        WHERE devicetype = 'BLOCK_ROUTER'
    """
    cursor.execute(query_block_router)
    block_routers = cursor.fetchall()
    
    # Create a set of block router IPs for faster lookups
    block_router_ips = {router['ip'] for router in block_routers}
    for ip in block_router_ips:
        if ip in topology_graph.nodes:
            topology_graph.nodes[ip]['is_block'] = True
    
    logging.info(f"Found and marked {len(block_router_ips)} BLOCK_ROUTERs")
    return block_router_ips

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
    """
    # Temporary structure to hold nodes for each ring
    ring_data = {}

    # Iterate through all nodes in the graph
    for node, attrs in graph.nodes(data=True):
        # Extract ring names and block IP
        pr_name = attrs.get('pr_name')
        lr_name_original = attrs.get('lr_name')
        block_ip = attrs.get('block_ip')

        # Normalize lr_name to extract the number
        lr_name_normalized = None
        if lr_name_original:
            # Try to extract just the number from input
            number_match = re.search(r'(?:no\.?|[-\s]|RING-|RING\s*)(\d+)', lr_name_original, re.IGNORECASE)
            if not number_match:  # Try finding number at the end if first regex failed
                number_match = re.search(r'(\d+)$', lr_name_original)

            if number_match:
                lr_name_normalized = str(int(number_match.group(1)))

        # Skip nodes without required ring information
        if not pr_name or not lr_name_normalized:
            continue

        # Create tuple key with normalized ring names
        ring_key = (pr_name, lr_name_normalized)

        # Initialize entry for this ring if it doesn't exist
        if ring_key not in ring_data:
            ring_data[ring_key] = {'nodes': set(), 'block_ip': None}

        # Add the current node to the set for this ring
        ring_data[ring_key]['nodes'].add(node)

        # Store the block_ip associated with this ring
        if block_ip and ring_data[ring_key]['block_ip'] is None:
            ring_data[ring_key]['block_ip'] = block_ip

    # Now, create the final subgraphs with block routers
    ring_graphs = {}
    for ring_key, data in ring_data.items():
        nodes_for_subgraph = data['nodes'].copy()
        block_ip_for_ring = data['block_ip']

        # Add the block router node if it exists and isn't already included
        if block_ip_for_ring and graph.has_node(block_ip_for_ring):
            nodes_for_subgraph.add(block_ip_for_ring)

        # Create a subgraph containing these nodes
        if nodes_for_subgraph:
            ring_graphs[ring_key] = graph.subgraph(nodes_for_subgraph).copy()

    return ring_graphs

def add_ups_devices(cursor, topology_graph):
    """Add UPS devices as nodes connected to their routers"""
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
            continue
            
        if not topology_graph.has_node(ups_ip):
            router_name = topology_graph.nodes[router_ip].get('name', f"Router-{router_ip}")
            ups_name = f"UPS-{router_name}"
            topology_graph.add_node(ups_ip, name=ups_name, devicetype="UPS", is_ups=True)
            ups_count += 1
        
        topology_graph.add_edge(router_ip, ups_ip, edge_type="router_ups")
    
    logging.info(f"Added {ups_count} UPS devices as nodes to the topology graph")
    return topology_graph

class RingVisualizerApp(QMainWindow):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.initUI()
    def initUI(self):
        # Main window setup
        self.setWindowTitle('Ring Visualization Tool')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Input area at the top
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Physical Ring Name:"))
        self.pr_entry = QLineEdit()
        input_layout.addWidget(self.pr_entry)
        
        input_layout.addWidget(QLabel("Logical Ring Number:"))
        self.lr_entry = QLineEdit()
        input_layout.addWidget(self.lr_entry)
        
        self.search_button = QPushButton("Search & Visualize")
        self.search_button.clicked.connect(self.find_and_display_ring)
        input_layout.addWidget(self.search_button)
        
        main_layout.addLayout(input_layout)
        
        # Splitter for graph and attributes
        splitter = QSplitter(Qt.Horizontal)
        
        # Graph visualization area
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Create toolbar for matplotlib
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Create a widget to hold the canvas and toolbar
        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)
        graph_layout.addWidget(self.toolbar)
        graph_layout.addWidget(self.canvas)
        
        # Add to splitter
        splitter.addWidget(graph_widget)
        
        # Create a tab widget for table and details
        self.data_tabs = QTabWidget()
        
        # Attributes table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["ID", "IP", "Name", "Block Router", "Failed", "Degree"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.selectionModel().selectionChanged.connect(self.on_table_selection_changed)
        
        # Details widget
        self.details_widget = QWidget()
        details_layout = QVBoxLayout(self.details_widget)
        
        self.node_selector_label = QLabel("Select a node to view details:")
        self.node_selector = QListWidget()
        self.node_selector.itemClicked.connect(self.on_node_selected_from_list)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        
        # Add toggle checkboxes
        self.toggle_failed_checkbox = QCheckBox("Mark as Failed")
        self.toggle_failed_checkbox.stateChanged.connect(self.toggle_failed_status)
        
        # Add isolated toggle checkbox
        self.toggle_isolated_checkbox = QCheckBox("Mark as Isolated")
        self.toggle_isolated_checkbox.stateChanged.connect(self.toggle_isolated_status)
        
        details_layout.addWidget(self.node_selector_label)
        details_layout.addWidget(self.node_selector)
        details_layout.addWidget(self.details_text)
        details_layout.addWidget(self.toggle_failed_checkbox)
        details_layout.addWidget(self.toggle_isolated_checkbox)
        
        # Add tabs to tab widget
        self.data_tabs.addTab(self.table, "Table View")
        self.data_tabs.addTab(self.details_widget, "Details View")
        
        # Add tab widget to splitter
        splitter.addWidget(self.data_tabs)
        
        # Set initial splitter sizes
        splitter.setSizes([600, 400])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Enter Physical Ring name and Logical Ring number to visualize")
        
        # Add annotation for node hover
        self.annotation = self.ax.annotate("", xy=(0, 0), xytext=(20, 20),
                                         textcoords="offset points",
                                         bbox=dict(boxstyle="round", fc="w"),
                                         arrowprops=dict(arrowstyle="->"))
        self.annotation.set_visible(False)

        # Connect events for interactivity
        self.canvas.mpl_connect('pick_event', self.on_node_pick)
        self.canvas.mpl_connect('motion_notify_event', self.on_node_hover)
        
        # Add this to the initUI method
        self.create_block_data_button = QPushButton("Create Data from Block Components")
        self.create_block_data_button.clicked.connect(self.on_create_block_data_clicked)
        main_layout.addWidget(self.create_block_data_button)
        
        # Create a panel for GNN training functionality
        gnn_train_layout = QHBoxLayout()
        self.model_path_label = QLabel("Model Path:")
        self.model_path_entry = QLineEdit()
        self.model_path_entry.setText("modelv1.pt")  # Default path from rcav8.py
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.on_load_model_clicked)

        self.train_button = QPushButton("Train with Current Data")
        self.train_button.clicked.connect(self.on_train_model_clicked)
        self.train_button.setEnabled(False)  # Disable until model is loaded

        gnn_train_layout.addWidget(self.model_path_label)
        gnn_train_layout.addWidget(self.model_path_entry)
        gnn_train_layout.addWidget(self.load_model_button)
        gnn_train_layout.addWidget(self.train_button)

        main_layout.addLayout(gnn_train_layout)
    
    def find_and_display_ring(self):
        """Find the ring and update the visualization"""
        pr_name = self.pr_entry.text().strip().upper()
        lr_name_input = self.lr_entry.text().strip()
        
        # Try to normalize LR name to extract the number
        lr_number = None
        if lr_name_input:
            # Try to extract just the number from input
            number_match = re.search(r'\d+', lr_name_input)
            if number_match:
                lr_number = str(int(number_match.group(0)))  # Remove leading zeros
            else:
                lr_number = lr_name_input
        
        if not pr_name or not lr_number:
            QMessageBox.critical(self, "Input Error", "Please enter both Physical Ring name and Logical Ring number")
            return
        
        # Use the existing function to get ring subgraphs
        ring_subgraphs = get_ring_node_mapping(self.graph)
        
        # Look for the requested ring
        ring_key = (pr_name, lr_number)
        if ring_key not in ring_subgraphs:
            self.statusBar.showMessage(f"Ring not found: PR '{pr_name}', LR '{lr_number}'")
            # Clear displays
            self.ax.clear()
            self.table.setRowCount(0)
            self.canvas.draw()
            return
        
        # Get the subgraph
        ring_subgraph = ring_subgraphs[ring_key]
        self.statusBar.showMessage(f"Found ring: PR '{pr_name}', LR '{lr_number}' with {ring_subgraph.number_of_nodes()} nodes and {ring_subgraph.number_of_edges()} edges")
        
        # Find block router in this ring
        block_node = None
        for node, attrs in ring_subgraph.nodes(data=True):
            if attrs.get('is_block', False):
                block_node = node
                break
        
        # Clear previous visualization
        self.ax.clear()
        self.table.setRowCount(0)
        
        # Identify main ring nodes and spur nodes
        node_degrees = dict(ring_subgraph.degree())
        
        # --- Identify Main Ring Nodes and Spur Nodes ---
        main_ring_nodes_set = set()
        spur_connections = {}  # spur_node -> connector_node
        
        for node, degree in node_degrees.items():
            if degree >= 2:
                main_ring_nodes_set.add(node)
            elif degree == 1:
                # Find the single neighbor (connector)
                connector = list(ring_subgraph.neighbors(node))[0]
                spur_connections[node] = connector
        
        # Validate spurs are connected to main ring
        valid_spur_connections = {}
        for spur, connector in spur_connections.items():
            if connector in main_ring_nodes_set:
                valid_spur_connections[spur] = connector
            else:
                # Treat the connector as part of main ring for positioning
                main_ring_nodes_set.add(connector)
                valid_spur_connections[spur] = connector
        
        spur_nodes_set = set(valid_spur_connections.keys())
        main_ring_nodes_list = list(main_ring_nodes_set)
        main_ring_count = len(main_ring_nodes_list)
        
        # --- Order Main Ring Nodes Sequentially ---
        ordered_main_ring_nodes = []
        
        # If block node is part of main ring, use it as start
        # Otherwise use first main ring node
        if block_node in main_ring_nodes_set:
            start_node = block_node
        else:
            start_node = main_ring_nodes_list[0] if main_ring_nodes_list else None
        
        if start_node:
            try:
                ordered_main_ring_nodes = [start_node]
                current_node = start_node
                previous_node = None
                visited_main_ring = {start_node}
                
                while len(ordered_main_ring_nodes) < main_ring_count:
                    found_next = False
                    # Consider only neighbors that are also in the main ring set
                    main_ring_neighbors = [n for n in ring_subgraph.neighbors(current_node) 
                                          if n in main_ring_nodes_set]
                    main_ring_neighbors.sort()  # Deterministic order
                    
                    for neighbor in main_ring_neighbors:
                        if neighbor != previous_node:
                            if neighbor not in visited_main_ring:
                                ordered_main_ring_nodes.append(neighbor)
                                visited_main_ring.add(neighbor)
                                previous_node = current_node
                                current_node = neighbor
                                found_next = True
                                break
                    
                    if not found_next and len(ordered_main_ring_nodes) < main_ring_count:
                        # Add remaining unvisited main ring nodes
                        remaining_main = [n for n in main_ring_nodes_list if n not in visited_main_ring]
                        ordered_main_ring_nodes.extend(remaining_main)
                        break
            except Exception as e:
                ordered_main_ring_nodes = main_ring_nodes_list  # Fallback
        else:
            ordered_main_ring_nodes = main_ring_nodes_list  # Fallback
        
        # IMPORTANT: Ensure block node is at index 0 by reordering if needed
        if block_node in ordered_main_ring_nodes and ordered_main_ring_nodes[0] != block_node:
            # Remove block node from current position
            ordered_main_ring_nodes.remove(block_node)
            # Insert at beginning (north position)
            ordered_main_ring_nodes.insert(0, block_node)
        
        # Create complete ordered node list for ID assignment
        ordered_nodes = ordered_main_ring_nodes.copy()
        # Add spur nodes, sorted for consistency
        ordered_nodes.extend(sorted(list(spur_nodes_set)))
        
        # If block node is a spur, ensure it's first in the list
        if block_node not in ordered_nodes:
            if block_node in spur_nodes_set:
                ordered_nodes.remove(block_node)
                ordered_nodes.insert(0, block_node)
        
        # Position nodes in a circular layout
        pos = {}
        
        # Create a perfect circular layout for main ring nodes
        if ordered_main_ring_nodes:
            center = np.array([0.5, 0.5])
            radius = 0.4
            num_nodes = len(ordered_main_ring_nodes)
            
            # Position main ring nodes in a circle following their sequential order
            for i, node in enumerate(ordered_main_ring_nodes):
                # Calculate angle - start from top (north) and go clockwise
                # First node (block node if it exists) will be at the top
                angle = 2 * np.pi * i / num_nodes - np.pi/2  # Start from top (-pi/2)
                
                # Calculate position
                x = center[0] + radius * np.cos(angle)
                y = -(center[1] + radius * np.sin(angle))
                pos[node] = np.array([x, y])
            
            # Position spur nodes radially outward from their connection points
            for spur_node, connector_node in valid_spur_connections.items():
                if connector_node in pos:
                    connector_pos = pos[connector_node]
                    # Calculate direction from center to connector
                    direction = connector_pos - center
                    # Normalize and extend slightly
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction) * 0.15
                        pos[spur_node] = connector_pos + direction
                    else:
                        # Fallback if direction is zero
                        pos[spur_node] = connector_pos + np.array([0.15, 0])
        else:
            # Fallback to spring layout if we have no main ring nodes
            pos = nx.spring_layout(ring_subgraph, seed=42)
            # If block node exists, force it to the top
            if block_node in pos:
                pos[block_node][1] = 1.0  # Set y to maximum
        
        # Draw the network
        # Node colors
        node_colors = []
        node_sizes = []
        node_order = list(ring_subgraph.nodes())  # Default order for drawing
        
        for node in node_order:
            if node == block_node:
                node_colors.append('red')  # Block router
                node_sizes.append(450)  # Larger size for block router
            elif ring_subgraph.nodes[node].get('is_ups', False):
                node_colors.append('yellow')  # UPS
                node_sizes.append(300)
            elif ring_subgraph.nodes[node].get('failed', False):
                node_colors.append('gray')  # Failed node
                node_sizes.append(300)
            else:
                node_colors.append('skyblue')  # Normal node
                node_sizes.append(300)
        
        # Edge colors
        edge_colors = []
        for u, v in ring_subgraph.edges():
            if ring_subgraph.edges[u, v].get('failed', False):
                edge_colors.append('red')  # Failed edge
            else:
                edge_colors.append('black')  # Normal edge
        
        # Draw the graph
        node_collection = nx.draw_networkx(
            ring_subgraph, 
            pos=pos,
            ax=self.ax,
            with_labels=False,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color=edge_colors,
            font_size=10
        )
        
        # Set the picker property on the node collection after drawing
        if node_collection:
            node_collection.set_picker(5)  # 5 points tolerance for picking
        
        # Assign IDs based on our ordered node list
        node_ids = {}
        for i, node in enumerate(ordered_nodes):
            node_ids[node] = i
        
        # Add node labels with proper IDs
        labels = {}
        for node in ring_subgraph.nodes():
            labels[node] = str(node_ids.get(node, '?'))
        
        nx.draw_networkx_labels(ring_subgraph, pos, labels, font_size=9, font_color='white')
        
        # Populate the table using our ordered node list
        self.table.setRowCount(len(ordered_nodes))
        
        for i, node in enumerate(ordered_nodes):
            # Add node to the attribute table
            attrs = ring_subgraph.nodes[node]
            name = attrs.get('name', 'Unknown')
            degree = node_degrees.get(node, 0)
            
            # Populate table row
            self.table.setItem(i, 0, QTableWidgetItem(str(node_ids[node])))
            self.table.setItem(i, 1, QTableWidgetItem(str(node)))
            self.table.setItem(i, 2, QTableWidgetItem(str(name)))
            self.table.setItem(i, 3, QTableWidgetItem("Yes" if attrs.get('is_block', False) else "No"))
            self.table.setItem(i, 4, QTableWidgetItem("Yes" if attrs.get('failed', False) else "No"))
            self.table.setItem(i, 5, QTableWidgetItem(str(degree)))
        
        # Store node positions for interactivity
        self.node_positions = pos
        self.node_ids = node_ids
        self.current_ring_subgraph = ring_subgraph
        
        # Set plot title and draw
        self.ax.set_title(f"Ring: PR {pr_name}, LR {lr_number}")
        self.ax.axis('off')
        self.canvas.draw()

    def on_table_selection_changed(self, selected, deselected):
        """Handle selection of a node in the table"""
        indexes = selected.indexes()
        if indexes:
            row = indexes[0].row()
            node_ip = self.table.item(row, 1).text()
            self.select_node(node_ip)

    def on_node_selected_from_list(self, item):
        """Handle selection of a node from the dropdown list"""
        node = item.data(Qt.UserRole)
        self.select_node(node)

    def on_node_pick(self, event):
        """Handle node selection in the graph"""
        if not hasattr(self, 'current_ring_subgraph') or not self.current_ring_subgraph:
            return
            
        # Get the index of the picked node
        ind = event.ind
        if len(ind) == 0:
            return
            
        # Get the picked node index
        node_index = ind[0]
        
        # Convert to actual node - fix the variable name from "nodes" to "node_list"
        node_list = list(self.current_ring_subgraph.nodes())
        if node_index < len(node_list):
            selected_node = node_list[node_index]
            self.select_node(selected_node)

    def on_node_hover(self, event):
        """Show tooltip on node hover"""
        if not hasattr(self, 'node_positions') or not self.current_ring_subgraph:
            return
        
        # Check if mouse is over a node
        if event.xdata is None or event.ydata is None:
            if hasattr(self, 'annotation'):
                self.annotation.set_visible(False)
                self.canvas.draw_idle()
            return
        
        closest_node = None
        min_distance = float('inf')
        
        # Find the closest node to the cursor
        if hasattr(self, 'node_positions'):
            for node, pos in self.node_positions.items():
                distance = np.sqrt((pos[0] - event.xdata)**2 + (pos[1] - event.ydata)**2)
                if distance < min_distance and distance < 0.05:  # Threshold for hover
                    min_distance = distance
                    closest_node = node
        
        if closest_node and hasattr(self, 'annotation'):
            # Show annotation with node info
            attrs = self.current_ring_subgraph.nodes[closest_node]
            node_id = getattr(self, 'node_ids', {}).get(closest_node, '?')
            name = attrs.get('name', 'Unknown')
            is_block = "Yes" if attrs.get('is_block', False) else "No"
            is_failed = "Yes" if attrs.get('failed', False) else "No"
            is_isolated = "Yes" if attrs.get('isolated', False) else "No"
            
            annotation_text = f"ID: {node_id}\nIP: {closest_node}\nName: {name}\n"
            annotation_text += f"Block Router: {is_block}\nFailed: {is_failed}\nIsolated: {is_isolated}"
            
            self.annotation.xy = (self.node_positions[closest_node][0], self.node_positions[closest_node][1])
            self.annotation.set_text(annotation_text)
            self.annotation.set_visible(True)
        elif hasattr(self, 'annotation'):
            self.annotation.set_visible(False)
        
        self.canvas.draw_idle()

    def select_node(self, node):
        """Select a node and show its details"""
        if not hasattr(self, 'current_ring_subgraph') or not self.current_ring_subgraph or node not in self.current_ring_subgraph:
            return
        
        self.selected_node = node
        attrs = self.current_ring_subgraph.nodes[node]
        
        # Format details display
        details_text = f"<h3>Node Details</h3>"
        details_text += f"<p><b>ID:</b> {getattr(self, 'node_ids', {}).get(node, '?')}</p>"
        details_text += f"<p><b>IP Address:</b> {node}</p>"
        details_text += f"<p><b>Name:</b> {attrs.get('name', 'Unknown')}</p>"
        
        # Display attributes
        details_text += "<h4>Attributes</h4><ul>"
        
        # Block router status
        is_block = attrs.get('is_block', False)
        details_text += f"<li><b>Block Router:</b> {'Yes' if is_block else 'No'}</li>"
        
        # Failed status
        is_failed = attrs.get('failed', False)
        details_text += f"<li><b>Failed:</b> {'Yes' if is_failed else 'No'}</li>"
        
        # Isolated status
        is_isolated = attrs.get('isolated', False)
        details_text += f"<li><b>Isolated:</b> {'Yes' if is_isolated else 'No'}</li>"
        
        details_text += "</ul>"
        
        # Show details
        if hasattr(self, 'details_text'):
            self.details_text.setHtml(details_text)
            
            # Update the toggle failed checkbox to match current state
            if hasattr(self, 'toggle_failed_checkbox'):
                self.toggle_failed_checkbox.setChecked(is_failed)
                
            # Update the toggle isolated checkbox to match current state
            if hasattr(self, 'toggle_isolated_checkbox'):
                self.toggle_isolated_checkbox.setChecked(is_isolated)
        
        # Switch to details tab if it exists
        if hasattr(self, 'data_tabs') and hasattr(self, 'details_widget'):
            self.data_tabs.setCurrentWidget(self.details_widget)
        
        # Update status bar if it exists
        if hasattr(self, 'statusBar'):
            self.statusBar.showMessage(f"Selected Node: {node} (ID: {getattr(self, 'node_ids', {}).get(node, '?')})")
        
        # Highlight the corresponding row in the table
        if hasattr(self, 'table'):
            for row in range(self.table.rowCount()):
                if self.table.item(row, 1) and self.table.item(row, 1).text() == str(node):
                    self.table.selectRow(row)
                    break

    def toggle_failed_status(self, state):
        """Toggle the 'failed' attribute for the selected node"""
        if not hasattr(self, 'selected_node') or not self.selected_node:
            return
        
        if not hasattr(self, 'current_ring_subgraph') or not self.current_ring_subgraph:
            return
        
        if self.selected_node not in self.current_ring_subgraph:
            return
        
        # Get the current status
        current_status = self.current_ring_subgraph.nodes[self.selected_node].get('failed', False)
        
        # Toggle the failed status
        new_status = bool(state)  # Convert Qt.Checked to bool
        
        # Update the node attribute in both the ring subgraph and the main graph
        self.current_ring_subgraph.nodes[self.selected_node]['failed'] = new_status
        
        # Also update in the original graph if the node exists there
        if self.selected_node in self.graph:
            self.graph.nodes[self.selected_node]['failed'] = new_status
        
        # Update the visualization
        self.redraw_visualization()
        
        # Update the details panel
        self.select_node(self.selected_node)
        
        # Update status message
        action = "marked as failed" if new_status else "marked as operational"
        self.statusBar.showMessage(f"Node {self.selected_node} {action}")

    def toggle_isolated_status(self, state):
        """Toggle the 'isolated' attribute for the selected node"""
        if not hasattr(self, 'selected_node') or not self.selected_node:
            return
        
        if not hasattr(self, 'current_ring_subgraph') or not self.current_ring_subgraph:
            return
        
        if self.selected_node not in self.current_ring_subgraph:
            return
        
        # Toggle the isolated status
        new_status = bool(state)  # Convert Qt.Checked to bool
        
        # Update the node attribute in both the ring subgraph and the main graph
        self.current_ring_subgraph.nodes[self.selected_node]['isolated'] = new_status
        
        # Also update in the original graph if the node exists there
        if self.selected_node in self.graph:
            self.graph.nodes[self.selected_node]['isolated'] = new_status
        
        # Update the visualization
        self.redraw_visualization()
        
        # Update the details panel
        self.select_node(self.selected_node)
        
        # Update status message
        action = "marked as isolated" if new_status else "marked as connected"
        self.statusBar.showMessage(f"Node {self.selected_node} {action}")

    def redraw_visualization(self):
        """Redraw the current visualization to reflect changes"""
        if hasattr(self, 'current_ring_subgraph') and self.current_ring_subgraph:
            # Save current view limits if needed
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # Clear the axes
            self.ax.clear()
            
            # Draw the network
            # Node colors
            node_colors = []
            node_sizes = []
            
            for node in self.current_ring_subgraph.nodes():
                attrs = self.current_ring_subgraph.nodes[node]
                is_block = attrs.get('is_block', False)
                is_failed = attrs.get('failed', False)
                is_ups = attrs.get('is_ups', False)
                is_isolated = attrs.get('isolated', False)
                
                if is_block:
                    node_colors.append('red')  # Block router
                    node_sizes.append(450)
                elif is_isolated:
                    node_colors.append('purple')  # Isolated node
                    node_sizes.append(300)
                elif is_ups:
                    node_colors.append('yellow')  # UPS
                    node_sizes.append(300)
                elif is_failed:
                    node_colors.append('gray')  # Failed node
                    node_sizes.append(300)
                else:
                    node_colors.append('skyblue')  # Normal node
                    node_sizes.append(300)
            
            # Edge colors
            edge_colors = []
            for u, v in self.current_ring_subgraph.edges():
                if self.current_ring_subgraph.edges[u, v].get('failed', False):
                    edge_colors.append('red')  # Failed edge
                else:
                    edge_colors.append('black')  # Normal edge
            
            # Draw the graph
            node_collection = nx.draw_networkx(
                self.current_ring_subgraph, 
                pos=self.node_positions,
                ax=self.ax,
                with_labels=False,
                node_color=node_colors,
                node_size=node_sizes,
                edge_color=edge_colors,
                font_size=10
            )

            # Set the picker property on the node collection after drawing
            if node_collection:
                node_collection.set_picker(5)  # 5 points tolerance for picking
            
            # Add node labels
            labels = {}
            for node in self.current_ring_subgraph.nodes():
                labels[node] = str(self.node_ids.get(node, '?'))
            
            nx.draw_networkx_labels(self.current_ring_subgraph, self.node_positions, labels, font_size=9, font_color='white')
            
            # Update the table rows
            for row in range(self.table.rowCount()):
                node_ip = self.table.item(row, 1).text()
                if node_ip in self.current_ring_subgraph:
                    attrs = self.current_ring_subgraph.nodes[node_ip]
                    is_failed = attrs.get('failed', False)
                    self.table.setItem(row, 4, QTableWidgetItem("Yes" if is_failed else "No"))
            
            # Restore view limits
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.ax.axis('off')
            
            # Redraw canvas
            self.canvas.draw()

    def create_data_object_from_block_components(self):
        """
        Create a PyG Data object from:
        1. All connected components from the block router in the main topology graph
        2. Using the failed and isolated attributes from user modifications
        3. With isolated nodes as the target labels
        """
        if not hasattr(self, 'graph') or not self.graph:
            self.statusBar.showMessage("No topology graph available")
            return None
        
        # Get PR and LR from the interface
        pr_name = self.pr_entry.text().strip().upper()
        lr_name_input = self.lr_entry.text().strip()
        
        # Extract LR number
        lr_number = None
        if lr_name_input:
            number_match = re.search(r'\d+', lr_name_input)
            if number_match:
                lr_number = str(int(number_match.group(0)))
            else:
                lr_number = lr_name_input
        
        if not pr_name or not lr_number:
            self.statusBar.showMessage("Please enter both PR and LR values")
            return None
        
        # Find the block router in the current visualization
        block_node = None
        for node, attrs in self.current_ring_subgraph.nodes(data=True):
            if attrs.get('is_block', False):
                block_node = node
                break
        
        if not block_node:
            self.statusBar.showMessage("No block router found")
            return None
        
        # Get all nodes connected to the block router in the original topology graph
        # This includes nodes from other rings that are connected to this block router
        connected_subgraph = nx.node_connected_component(self.graph, block_node)
        connected_subgraph = self.graph.subgraph(connected_subgraph).copy()
        
        # Identify which nodes belong to our current PR/LR
        # We'll keep track of all nodes, but only modify attributes for nodes in our current ring
        current_ring_nodes = set(self.current_ring_subgraph.nodes())
        
        # Collect failed and isolated nodes from user modifications
        failed_nodes = []
        isolated_nodes = []
        
        for node in connected_subgraph.nodes():
            # If this node is in our current ring visualization
            if node in current_ring_nodes:
                # Apply user modifications from the visualization
                attrs = self.current_ring_subgraph.nodes[node]
                
                if attrs.get('failed', False):
                    failed_nodes.append(node)
                    connected_subgraph.nodes[node]['failed'] = True
                else:
                    connected_subgraph.nodes[node]['failed'] = False
                    
                if attrs.get('isolated', False):
                    isolated_nodes.append(node)
                    connected_subgraph.nodes[node]['isolated'] = True
                else:
                    connected_subgraph.nodes[node]['isolated'] = False
        
        # Create the PyG Data object
        # Node features: [is_source, is_failed, is_ups]
        nodes = list(connected_subgraph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        x = torch.zeros((len(nodes), 3), dtype=torch.float)
        
        # Set features
        for i, node in enumerate(nodes):
            attrs = connected_subgraph.nodes[node]
            # Is source (block router)
            x[i, 0] = 1.0 if node == block_node else 0.0
            # Is failed
            x[i, 1] = 1.0 if node in failed_nodes else 0.0
            # Is UPS
            x[i, 2] = 1.0 if attrs.get('is_ups', False) else 0.0
        
        # Edge index and attributes
        edge_index = []
        edge_attr = []
        
        for u, v in connected_subgraph.edges():
            # Add both directions for undirected graph
            edge_index.append([node_to_idx[u], node_to_idx[v]])
            edge_index.append([node_to_idx[v], node_to_idx[u]])
            
            # Edge features
            edge_attr.append([0.0])  # is_failed=0
            edge_attr.append([0.0])
        
        # Convert to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t() if edge_index else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.zeros((0, 1), dtype=torch.float)
        
        # Target: isolated nodes
        y = torch.zeros(len(nodes), dtype=torch.float)
        for node in isolated_nodes:
            if node in node_to_idx:  # Safety check
                y[node_to_idx[node]] = 1.0
        
        # Create data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=len(nodes)
        )
        
        # Store original node IDs and PR/LR info
        data.node_ids = nodes
        data.pr_name = pr_name
        data.lr_number = lr_number
        
        self.statusBar.showMessage(
            f"Created Data object with {len(nodes)} nodes ({len(connected_subgraph.edges())} edges), " 
            f"{len(failed_nodes)} failed, {len(isolated_nodes)} isolated"
        )
        return data

    def on_create_block_data_clicked(self):
        """Handle click on create data from block components button"""
        data = self.create_data_object_from_block_components()
        if data:
            # Show summary
            summary = (f"Data Object Created:\n"
                      f"- PR: {data.pr_name}, LR: {data.lr_number}\n"
                      f"- Nodes: {data.num_nodes}\n"
                      f"- Edges: {data.edge_index.shape[1] // 2}\n"  # Divide by 2 for undirected
                      f"- Failed Nodes: {int(data.x[:, 1].sum().item())}\n"
                      f"- Isolated Nodes: {int(data.y.sum().item())}")
            
            QMessageBox.information(self, "Block Components Data", summary)
            
            # Save to file
            filename = f"block_data_{data.pr_name}_{data.lr_number}.pt"
            torch.save(data, filename)
            self.statusBar.showMessage(f"Data saved to {filename}")
    
    def load_gnn_model(self, model_path):
        """Load a pre-trained GNN model from a file"""
        try:
            # Create model with same architecture as in rcav8.py
            model = IsolationGNN(num_node_features=3, num_edge_features=1, hidden_channels=32, num_layers=18)
            
            # Load pre-trained weights
            model_state = torch.load(model_path)
            model.load_state_dict(model_state)
            
            self.statusBar.showMessage(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            self.statusBar.showMessage(f"Error loading model: {str(e)}")
            QMessageBox.critical(self, "Model Loading Error", f"Could not load model: {str(e)}")
            return None

    def train_model(self, model, data, epochs=50, learning_rate=0.001):
        """Train the model with the new data from visualization"""
        if not model or not data:
            self.statusBar.showMessage("Model or data not available for training")
            return
        
        # Progress dialog
        progress = QMessageBox(QMessageBox.Information, "Training Progress", "Training in progress...", QMessageBox.Cancel)
        progress.setModal(False)
        progress.show()
        
        # Remember initial evaluation mode and switch to training
        was_training = model.training
        model.train()
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Loss history for reporting
        losses = []
        
        try:
            # Training loop
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                # Forward pass
                output = model(data)
                
                # Calculate loss - binary cross entropy for binary classification
                # Our target is y which indicates whether nodes are isolated
                loss = F.binary_cross_entropy(output, data.y)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Save loss for reporting
                current_loss = loss.item()
                losses.append(current_loss)
                
                # Update status bar
                self.statusBar.showMessage(f"Epoch {epoch+1}/{epochs}, Loss: {current_loss:.4f}")
                QApplication.processEvents()  # Keep UI responsive
                
                # Check if user cancelled
                if not progress.isVisible():
                    break
                    
            # Final status message
            self.statusBar.showMessage(f"Training completed. Final loss: {losses[-1]:.4f}")
            
            # Close progress dialog
            progress.accept()
            
            # Create a simple training curve visualization
            self.display_training_curve(losses)
            
            # Return to previous evaluation mode
            if not was_training:
                model.eval()
                
            return model
                
        except Exception as e:
            progress.accept()
            self.statusBar.showMessage(f"Error during training: {str(e)}")
            QMessageBox.critical(self, "Training Error", f"Error during model training: {str(e)}")
            
            if not was_training:
                model.eval()
            
            return None

    def display_training_curve(self, losses):
        """Display a training curve in a new window"""
        # Create a new figure for the training curve
        training_fig = plt.figure(figsize=(8, 4))
        ax = training_fig.add_subplot(111)
        ax.plot(losses)
        ax.set_title('Training Loss Curve')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True)
        
        # Create a new window for the training curve
        training_window = QMainWindow(self)
        training_window.setWindowTitle("Training Progress")
        training_widget = QWidget()
        layout = QVBoxLayout(training_widget)
        
        # Create a canvas for the matplotlib figure
        canvas = FigureCanvas(training_fig)
        nav_toolbar = NavigationToolbar(canvas, training_window)
        
        layout.addWidget(nav_toolbar)
        layout.addWidget(canvas)
        
        # Add save model button with updated text
        save_button = QPushButton("Save and Overwrite Original Model")
        save_button.clicked.connect(lambda: self.save_model())
        layout.addWidget(save_button)
        
        training_window.setCentralWidget(training_widget)
        training_window.resize(800, 500)
        training_window.show()

    def save_model(self):
        """Save the trained model, overwriting the original file"""
        if not hasattr(self, 'trained_model') or not self.trained_model:
            QMessageBox.warning(self, "Save Error", "No trained model available to save")
            return
        
        if not hasattr(self, 'original_model_path'):
            QMessageBox.warning(self, "Save Error", "Original model path not found")
            return
        
        try:
            # Confirm before overwriting
            reply = QMessageBox.question(
                self, 
                "Confirm Overwrite", 
                f"This will overwrite the original model file at {self.original_model_path}. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
            
            # Save model state dictionary, overwriting the original file
            torch.save(self.trained_model.state_dict(), self.original_model_path)
            
            self.statusBar.showMessage(f"Model saved to {self.original_model_path} (original file overwritten)")
            QMessageBox.information(self, "Model Saved", f"Model successfully saved to {self.original_model_path}")
        except Exception as e:
            self.statusBar.showMessage(f"Error saving model: {str(e)}")
            QMessageBox.critical(self, "Save Error", f"Could not save model: {str(e)}")

    def on_load_model_clicked(self):
        """Handle click on load model button"""
        model_path = self.model_path_entry.text()
        if not model_path:
            QMessageBox.warning(self, "Input Error", "Please enter a model path")
            return
        
        # Load the model
        self.trained_model = self.load_gnn_model(model_path)
        if self.trained_model:
            self.train_button.setEnabled(True)
            # Store the original model path for overwriting
            self.original_model_path = model_path
            self.statusBar.showMessage(f"Model loaded from {model_path} and ready for training")

    def on_train_model_clicked(self):
        """Handle click on train model button"""
        # First, create the data object from block components
        data = self.create_data_object_from_block_components()
        if not data:
            QMessageBox.warning(self, "Data Error", "Could not create data object from visualization")
            return
        
        # Check that we have a loaded model
        if not hasattr(self, 'trained_model') or not self.trained_model:
            QMessageBox.warning(self, "Model Error", "Please load a model first")
            return
        
        # Start training process
        self.statusBar.showMessage("Starting model training...")
        
        # Show data summary before training
        isolated_count = int(data.y.sum().item())
        total_nodes = data.num_nodes
        
        summary = (f"Training with:\n"
                  f"- {total_nodes} nodes\n"
                  f"- {isolated_count} isolated nodes ({isolated_count/total_nodes*100:.1f}%)\n"
                  f"- {data.edge_index.shape[1]//2} edges")
        QMessageBox.information(self, "Training Data Summary", summary)
        
        # Train the model - default 50 epochs
        self.trained_model = self.train_model(self.trained_model, data)

# First, let's define the EdgeAwareConv layer used in rcav8.py
class IsolationGNN(torch.nn.Module):
    def __init__(self, num_node_features=3, num_edge_features=1, hidden_channels=32, num_layers=6):
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

def main():
    """Main function to load data and launch the GUI"""
    try:
        setup_logging()
        logging.info("Starting Ring Visualization Tool")
        
        # Connect to database
        mydb, cursor = setup_database()
        logging.info("Connected to database")
        
        # Load network topology
        topology_graph = load_topology(cursor)
        logging.info(f"Loaded topology with {topology_graph.number_of_nodes()} nodes and {topology_graph.number_of_edges()} edges")
        
        # Identify block routers
        block_router_ips = identify_block_routers(cursor, topology_graph)
        
        # Assign nodes to their blocks
        topology_graph = assign_nodes_to_blocks(topology_graph, block_router_ips)
        
        # Add UPS devices
        topology_graph = add_ups_devices(cursor, topology_graph)
        
        # Launch GUI
        logging.info("Launching GUI")
        
        app = QApplication(sys.argv)
        main_window = RingVisualizerApp(topology_graph)
        main_window.show()
        sys.exit(app.exec_())
        
    except Exception as e:
        logging.error(f"Error in main function: {e}", exc_info=True)
        QMessageBox.critical(None, "Error", f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()