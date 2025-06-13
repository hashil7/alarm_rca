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
                            QTextEdit, QListWidget, QListWidgetItem, QCheckBox, QComboBox)
from PyQt5.QtCore import Qt
import torch
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import MessagePassing, GCNConv
from datetime import datetime
import os

class UPSLowBatteryAlarmDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

class IsolationGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        
        # Node and edge encoders
        self.node_encoder = torch.nn.Linear(num_node_features, hidden_channels)
        self.edge_encoder = torch.nn.Linear(num_edge_features, hidden_channels)
        
        # Convolutional layers with proper naming
        self.conv_layers = torch.nn.ModuleList()
        for _ in range(6):  # Original model has 6 conv layers
            layer = torch.nn.ModuleDict({
                'lin_node': torch.nn.Linear(hidden_channels, hidden_channels),
                'lin_update': torch.nn.Linear(hidden_channels * 2, hidden_channels),
                'lin_edge': torch.nn.Linear(hidden_channels, hidden_channels)
            })
            self.conv_layers.append(layer)
        
        # Classifier layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(hidden_channels, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Initial encodings
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # Message passing layers
        for conv in self.conv_layers:
            # Node feature transformation
            node_features = conv['lin_node'](x)
            
            # Edge feature transformation
            edge_features = conv['lin_edge'](edge_attr)
            
            # Reshape edge_features to match node features
            edge_features = edge_features.mean(dim=0).expand(node_features.size(0), -1)
            
            # Concatenate node features with transformed edge features
            combined_features = torch.cat([node_features, edge_features], dim=1)
            
            # Update node features
            x = conv['lin_update'](combined_features)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Final classification
        x = self.classifier(x)
        return torch.sigmoid(x).squeeze()

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
        WHERE n1.devicetype IN ('GP_ROUTER','BLOCK_ROUTER') 
        AND n2.devicetype IN ('GP_UPS','BLOCK_UPS')
    """
    cursor.execute(query_ups)
    router_ups_mappings = cursor.fetchall()
    
    ups_count = 0
    for mapping in router_ups_mappings:
        router_ip = mapping["router_ip"]
        
        # Mark the router as having UPS
        if topology_graph.has_node(router_ip):
            topology_graph.nodes[router_ip]['is_ups'] = True
            ups_count += 1
    
    logging.info(f"Added UPS attributes to {ups_count} nodes in the topology graph")
    return topology_graph

class RingVisualizerApp(QMainWindow):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        # Add state history tracking
        self.node_history = {}  # Store historical states
        self.edge_history = {}  # Add this to track edge failures
        self.selected_edge = None  # Add this to track selected edge
        self.current_ring_subgraph = None  # Initialize as None
        self.initUI()
    def initUI(self):
        # Main window setup - optimized for 1366x768
        self.setWindowTitle('Ring Visualization Tool')
        
        # Force smaller initial size for 1366x768
        self.setGeometry(50, 50, 1300, 650)  # Smaller height
        self.showMaximized()  # Always start maximized
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(3)  # Reduce spacing
        main_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        
        # Compact input area
        input_layout = QHBoxLayout()
        input_layout.setSpacing(5)
        
        input_layout.addWidget(QLabel("PR:"))  # Shorter labels
        self.pr_entry = QLineEdit()
        self.pr_entry.setMaximumWidth(120)  # Smaller width
        input_layout.addWidget(self.pr_entry)
        
        input_layout.addWidget(QLabel("LR:"))
        self.lr_entry = QLineEdit()
        self.lr_entry.setMaximumWidth(80)
        input_layout.addWidget(self.lr_entry)
        
        self.search_button = QPushButton("Search")
        self.search_button.setMaximumWidth(80)
        self.search_button.clicked.connect(self.find_and_display_ring)
        input_layout.addWidget(self.search_button)
        
        input_layout.addStretch()
        main_layout.addLayout(input_layout)
        
        # Horizontal splitter with smaller proportions
        splitter = QSplitter(Qt.Horizontal)
        
        # SMALLER graph visualization area
        self.figure = plt.figure(figsize=(5, 3.5))  # Much smaller figure
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Compact toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)
        graph_layout.setSpacing(2)
        graph_layout.addWidget(self.toolbar)
        graph_layout.addWidget(self.canvas)
        
        splitter.addWidget(graph_widget)
        
        # COMPACT tab widget for controls
        self.data_tabs = QTabWidget()
        self.data_tabs.setMaximumWidth(320)  # Limit width
        
        # Compact table
        self.table = QTableWidget()
        self.table.setColumnCount(4)  # Fewer columns
        self.table.setHorizontalHeaderLabels(["ID", "IP", "Name", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.selectionModel().selectionChanged.connect(self.on_table_selection_changed)
        
        # COMPACT details widget
        self.details_widget = QWidget()
        details_layout = QVBoxLayout(self.details_widget)
        details_layout.setSpacing(3)
        
        # Smaller components
        self.node_selector_label = QLabel("Node:")
        self.node_selector = QListWidget()
        self.node_selector.setMaximumHeight(80)  # Limit height
        self.node_selector.itemClicked.connect(self.on_node_selected_from_list)
        
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(100)  # Limit height
        self.details_text.setReadOnly(True)
        
        # COMPACT alarm configuration
        self.alarm_group = QWidget()
        alarm_layout = QVBoxLayout(self.alarm_group)
        alarm_layout.setSpacing(2)
        
        alarm_layout.addWidget(QLabel("Alarms:"))
        
        # Single row for alarm types
        alarm_type_layout = QHBoxLayout()
        self.ups_alarm_checkbox = QCheckBox("UPS")
        self.node_alarm_checkbox = QCheckBox("Node")
        self.link_alarm_checkbox = QCheckBox("Link")
        alarm_type_layout.addWidget(self.ups_alarm_checkbox)
        alarm_type_layout.addWidget(self.node_alarm_checkbox)
        alarm_type_layout.addWidget(self.link_alarm_checkbox)
        alarm_layout.addLayout(alarm_type_layout)
        
        # Single row for alarm status
        alarm_status_layout = QHBoxLayout()
        self.current_alarm_checkbox = QCheckBox("Current")
        self.previous_alarm_checkbox = QCheckBox("Previous")
        self.isolated_alarm_checkbox = QCheckBox("Isolated")
        alarm_status_layout.addWidget(self.current_alarm_checkbox)
        alarm_status_layout.addWidget(self.previous_alarm_checkbox)
        alarm_status_layout.addWidget(self.isolated_alarm_checkbox)
        alarm_layout.addLayout(alarm_status_layout)
        
        # Connect signals
        self.ups_alarm_checkbox.stateChanged.connect(self.on_alarm_changed)
        self.node_alarm_checkbox.stateChanged.connect(self.on_alarm_changed)
        self.link_alarm_checkbox.stateChanged.connect(self.on_alarm_changed)
        self.current_alarm_checkbox.stateChanged.connect(self.on_alarm_status_changed)
        self.previous_alarm_checkbox.stateChanged.connect(self.on_alarm_status_changed)
        self.isolated_alarm_checkbox.stateChanged.connect(self.on_alarm_status_changed)
        
        # COMPACT edge controls
        self.edge_selection_group = QWidget()
        edge_selection_layout = QVBoxLayout(self.edge_selection_group)
        edge_selection_layout.setSpacing(2)
        
        edge_selection_layout.addWidget(QLabel("Edge Control:"))
        self.edge_node1_combo = QComboBox()
        self.edge_node2_combo = QComboBox()
        edge_selection_layout.addWidget(QLabel("Node 1:"))
        edge_selection_layout.addWidget(self.edge_node1_combo)
        edge_selection_layout.addWidget(QLabel("Node 2:"))
        edge_selection_layout.addWidget(self.edge_node2_combo)
        
        self.mark_edge_button = QPushButton("Mark Failed")
        self.mark_edge_button.setMaximumHeight(25)
        self.mark_edge_button.clicked.connect(self.toggle_selected_edge)
        edge_selection_layout.addWidget(self.mark_edge_button)
        
        self.undo_edge_button = QPushButton("Undo")
        self.undo_edge_button.setMaximumHeight(25)
        self.undo_edge_button.clicked.connect(self.undo_edge_failures)
        edge_selection_layout.addWidget(self.undo_edge_button)
        
        self.edge_previous_failed_checkbox = QCheckBox("Previous Failed")
        self.edge_current_failed_checkbox = QCheckBox("Current Failed")

        edge_selection_layout.addWidget(self.edge_previous_failed_checkbox)
        edge_selection_layout.addWidget(self.edge_current_failed_checkbox)

        # Connect signals
        self.edge_previous_failed_checkbox.stateChanged.connect(self.on_edge_status_changed)
        self.edge_current_failed_checkbox.stateChanged.connect(self.on_edge_status_changed)
        
        # Add to details layout
        details_layout.addWidget(self.node_selector_label)
        details_layout.addWidget(self.node_selector)
        details_layout.addWidget(self.details_text)
        details_layout.addWidget(self.alarm_group)
        details_layout.addWidget(self.edge_selection_group)
        
        # Add tabs
        self.data_tabs.addTab(self.table, "Table")
        self.data_tabs.addTab(self.details_widget, "Controls")
        
        splitter.addWidget(self.data_tabs)
        
        # Set splitter sizes - more space for graph
        splitter.setSizes([800, 320])
        main_layout.addWidget(splitter)
        
        # COMPACT bottom controls in single row
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(5)
        
        self.create_block_data_button = QPushButton("Create Data")
        self.create_block_data_button.setMaximumHeight(25)
        self.create_block_data_button.clicked.connect(self.on_create_block_data_clicked)
        bottom_layout.addWidget(self.create_block_data_button)
        
        # Model controls in same row - ENHANCED with dropdown
        bottom_layout.addWidget(QLabel("Model:"))

        # Create a combo box for model selection
        self.model_combo = QComboBox()
        self.model_combo.setMaximumWidth(150)
        self.populate_model_combo()
        bottom_layout.addWidget(self.model_combo)

        # Connect signal to update text field
        self.model_combo.currentTextChanged.connect(self.on_model_combo_changed)

        # Keep the text entry for custom paths
        self.model_path_entry = QLineEdit()
        self.model_path_entry.setText("modelv3.pt")
        self.model_path_entry.setMaximumWidth(100)
        bottom_layout.addWidget(self.model_path_entry)

        # Add refresh button to reload models list
        self.refresh_models_button = QPushButton("↻")
        self.refresh_models_button.setMaximumHeight(25)
        self.refresh_models_button.setMaximumWidth(30)
        self.refresh_models_button.setToolTip("Refresh models list")
        self.refresh_models_button.clicked.connect(self.populate_model_combo)
        bottom_layout.addWidget(self.refresh_models_button)
        
        self.load_model_button = QPushButton("Load")
        self.load_model_button.setMaximumHeight(25)
        self.load_model_button.clicked.connect(self.on_load_model_clicked)  # Add this line
        bottom_layout.addWidget(self.load_model_button)  # Add this line

        self.train_button = QPushButton("Train")
        self.train_button.setMaximumHeight(25)
        self.train_button.clicked.connect(self.on_train_model_clicked)
        self.train_button.setEnabled(False)
        bottom_layout.addWidget(self.train_button)

        # Add after the train button
        self.save_model_button = QPushButton("Save")
        self.save_model_button.setMaximumHeight(25)
        self.save_model_button.clicked.connect(self.save_trained_model)
        self.save_model_button.setEnabled(False)  # Disable until model is trained
        bottom_layout.addWidget(self.save_model_button)

        # Create status bar - ADD THIS
        

        # Add annotation for node hover - MOVE THIS AFTER STATUS BAR
        self.annotation = self.ax.annotate("", xy=(0, 0), xytext=(20, 20),
                                         textcoords="offset points",
                                         bbox=dict(boxstyle="round", fc="w"),
                                         arrowprops=dict(arrowstyle="->"))
        self.annotation.set_visible(False)

        # Connect events for interactivity
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_node_hover)

        # Connect combo box signals
        self.edge_node1_combo.currentIndexChanged.connect(self.on_edge_combo_changed)
        self.edge_node2_combo.currentIndexChanged.connect(self.on_edge_combo_changed)

        # Add the bottom layout to main layout
        main_layout.addLayout(bottom_layout)
        
        # Set central widget
        self.setCentralWidget(central_widget)

    def auto_select_edge_in_combos(self, edge):
        """Automatically select the picked edge in combo boxes"""
        u, v = edge
        
        # Find and select u in combo1
        for i in range(self.edge_node1_combo.count()):
            if self.edge_node1_combo.itemData(i) == u:
                self.edge_node1_combo.setCurrentIndex(i)
                break
        
        # Find and select v in combo2
        for i in range(self.edge_node2_combo.count()):
            if self.edge_node2_combo.itemData(i) == v:
                self.edge_node2_combo.setCurrentIndex(i)
                break
        
        # Update edge status checkboxes
        self.update_edge_checkboxes_from_selection()

    def update_edge_checkboxes_from_selection(self):
        """Update edge checkboxes based on current combo selection"""
        node1 = self.edge_node1_combo.currentData()
        node2 = self.edge_node2_combo.currentData()
        
        if not node1 or not node2 or node1 == node2:
            return
            
        edge_key = tuple(sorted([node1, node2]))
        history = self.edge_history.get(edge_key, {
            'previous_failed': False,
            'current_failed': False
        })
        
        # Update checkboxes to reflect current state
        self.edge_previous_failed_checkbox.setChecked(history['previous_failed'])
        self.edge_current_failed_checkbox.setChecked(history['current_failed'])

    def highlight_selected_edge(self, edge):
        """Highlight the selected edge in the visualization"""
        if not hasattr(self, 'current_ring_subgraph'):
            return
            
        # Store the selected edge for highlighting in redraw
        self.highlighted_edge = edge
        self.redraw_visualization()

    def show_edge_details(self, edge):
        """Show details for the selected edge"""
        if not edge or not hasattr(self, 'details_text'):
            return
            
        u, v = edge
        
        # Format edge details
        details_text = f"<h3>Edge Details</h3>"
        details_text += f"<p><b>Node A:</b> {u}</p>"
        details_text += f"<p><b>Node B:</b> {v}</p>"
        
        # Get edge history
        edge_key = tuple(sorted([u, v]))
        history = self.edge_history.get(edge_key, {
            'previous_failed': False,
            'current_failed': False,
            'failure_timestamp': None
        })
        
        details_text += "<h4>Edge Status</h4><ul>"
        details_text += f"<li><b>Currently Failed:</b> {'Yes' if history['current_failed'] else 'No'}</li>"
        details_text += f"<li><b>Previously Failed:</b> {'Yes' if history['previous_failed'] else 'No'}</li>"
        if history['failure_timestamp']:
            details_text += f"<li><b>Last Status Change:</b> {history['failure_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</li>"
        details_text += "</ul>"
        
        details_text += "<p><i>Use checkboxes below to modify edge status</i></p>"
        
        self.details_text.setHtml(details_text)
        
        # Update status bar
        self.statusBar().showMessage(f"Selected Edge: {u} <-> {v}")

    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
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
            self.statusBar().showMessage(f"Ring not found: PR '{pr_name}', LR '{lr_number}'")
            # Clear displays
            self.ax.clear()
            self.table.setRowCount(0)
            self.canvas.draw()
            return
        
        # Initialize the subgraph before using it
        self.current_ring_subgraph = ring_subgraphs[ring_key]
        if not self.current_ring_subgraph:
            self.statusBar().showMessage("Failed to create ring subgraph")
            return
            
        self.statusBar().showMessage(f"Found ring: PR '{pr_name}', LR '{lr_number}' with {self.current_ring_subgraph.number_of_nodes()} nodes and {self.current_ring_subgraph.number_of_edges()} edges")
        
        # Find block router in this ring
        block_node = None
        for node, attrs in self.current_ring_subgraph.nodes(data=True):
            if attrs.get('is_block', False):
                block_node = node
                break
        
        # Clear previous visualization
        self.ax.clear()
        self.table.setRowCount(0)
        
        # Identify main ring nodes and spur nodes
        node_degrees = dict(self.current_ring_subgraph.degree())
        
        # --- Identify Main Ring Nodes and Spur Nodes ---
        main_ring_nodes_set = set()
        spur_connections = {}  # spur_node -> connector_node
        
        for node, degree in node_degrees.items():
            if degree >= 2:
                main_ring_nodes_set.add(node)
            elif degree == 1:
                # Find the single neighbor (connector)
                connector = list(self.current_ring_subgraph.neighbors(node))[0]
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
                    main_ring_neighbors = [n for n in self.current_ring_subgraph.neighbors(current_node) 
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
            pos = nx.spring_layout(self.current_ring_subgraph, seed=42)
            # If block node exists, force it to the top
            if block_node in pos:
                pos[block_node][1] = 1.0  # Set y to maximum
        
        # Draw the network
        # Node colors
        node_colors = []
        node_sizes = []
        node_order = list(self.current_ring_subgraph.nodes())  # Default order for drawing
        
        for node in node_order:
            # Get node history
            history = self.node_history.get(node, {
                'previous_failed': False,
                'current_failed': False
            })
            
            if node == block_node:
                node_colors.append('red')  # Block router
                node_sizes.append(450)
            elif self.current_ring_subgraph.nodes[node].get('isolated', False):
                node_colors.append('purple')  # Isolated node
                node_sizes.append(300)
            elif self.current_ring_subgraph.nodes[node].get('ups_low_battery', False):
                if self.current_alarm_checkbox.isChecked():
                    node_colors.append('orange')  # Current UPS alarm
                elif self.previous_alarm_checkbox.isChecked():
                    node_colors.append('yellow')  # Previous UPS alarm
                node_sizes.append(300)
            else:
                node_colors.append('skyblue')  # Default color
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
        for node in self.current_ring_subgraph.nodes():
            labels[node] = str(node_ids.get(node, '?'))
        
        nx.draw_networkx_labels(self.current_ring_subgraph, pos, labels, font_size=9, font_color='white')
        
        # Populate the table using our ordered node list
        self.table.setRowCount(len(ordered_nodes))
        
        for i, node in enumerate(ordered_nodes):
            # Add node to the attribute table
            attrs = self.current_ring_subgraph.nodes[node]
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
        self.current_ring_subgraph = self.current_ring_subgraph
        
        # Set plot title and draw
        self.ax.set_title(f"Ring: PR {pr_name}, LR {lr_number}")
        self.ax.axis('off')
        self.canvas.draw()

        # Update edge selection comboboxes
        self.update_edge_selection_combos()

        # Store for model metadata
        self.current_pr_name = pr_name
        self.current_lr_number = lr_number

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


    def on_node_hover(self, event):
        """Show tooltip on node hover with click instructions"""
        if not hasattr(self, 'node_positions') or not self.current_ring_subgraph:
            return
        
        # Check if mouse is over a node or edge
        if event.xdata is None or event.ydata is None:
            if hasattr(self, 'annotation'):
                self.annotation.set_visible(False)
                self.canvas.setCursor(Qt.ArrowCursor)
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
            # Change cursor to pointing hand
            self.canvas.setCursor(Qt.PointingHandCursor)
            
            # Show annotation with node info and instructions
            attrs = self.current_ring_subgraph.nodes[closest_node]
            node_id = getattr(self, 'node_ids', {}).get(closest_node, '?')
            name = attrs.get('name', 'Unknown')
            is_block = "Yes" if attrs.get('is_block', False) else "No"
            
            # Get status from history
            history = self.node_history.get(closest_node, {})
            status_parts = []
            if attrs.get('isolated', False):
                status_parts.append("Isolated")
            if history.get('current_failed', False):
                status_parts.append("Current Failed")
            if history.get('previous_failed', False):
                status_parts.append("Previous Failed")
            if attrs.get('ups_low_battery', False):
                status_parts.append("UPS Low Battery")
            if not status_parts:
                status_parts.append("Normal")
            
            annotation_text = f"ID: {node_id}\nIP: {closest_node}\nName: {name}\n"
            annotation_text += f"Block Router: {is_block}\nStatus: {', '.join(status_parts)}\n"
            annotation_text += f"\nRight-click to select node\nLeft-click to select edge"
            
            self.annotation.xy = (self.node_positions[closest_node][0], self.node_positions[closest_node][1])
            self.annotation.set_text(annotation_text)
            self.annotation.set_visible(True)
        else:
            if hasattr(self, 'annotation'):
                self.annotation.set_visible(False)
            self.canvas.setCursor(Qt.ArrowCursor)
        
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
        
        # Get node history
        history = self.node_history.get(node, {
            'previous_failed': False,
            'current_failed': False,
            'link_down': False
        })
        
        details_text += f"<li><b>Previously Failed:</b> {'Yes' if history['previous_failed'] else 'No'}</li>"
        details_text += f"<li><b>Current Failed:</b> {'Yes' if history['current_failed'] else 'No'}</li>"
        details_text += f"<li><b>Link Down:</b> {'Yes' if history.get('link_down', False) else 'No'}</li>"
        details_text += "</ul>"
        
        # Add UPS status details
        details_text += "<h4>UPS Status</h4><ul>"
        if attrs.get('is_ups', False):
            details_text += f"<li><b>Has UPS:</b> Yes</li>"
            details_text += f"<li><b>UPS Alarm:</b> {'Yes' if attrs.get('ups_alarm', False) else 'No'}</li>"
            details_text += f"<li><b>Low Battery:</b> {'Yes' if attrs.get('ups_low_battery', False) else 'No'}</li>"
            if node in self.node_history and 'alarm_timestamp' in self.node_history[node]:
                details_text += f"<li><b>Last Alarm:</b> {self.node_history[node]['alarm_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</li>"
        else:
            details_text += "<li><b>Has UPS:</b> No</li>"
        details_text += "</ul>"
        
        # Show details
        if hasattr(self, 'details_text'):
            self.details_text.setHtml(details_text)
            
            # Update checkboxes
            if hasattr(self, 'toggle_pfailed_checkbox'):
                self.toggle_pfailed_checkbox.setChecked(history['previous_failed'])
            if hasattr(self, 'toggle_cfailed_checkbox'):
                self.toggle_cfailed_checkbox.setChecked(history['current_failed'])
    
        # Update visualization to reflect link_down status
        self.redraw_visualization()

    def on_mouse_click(self, event):
        """Unified mouse click handler for both nodes and edges"""
        if not hasattr(self, 'current_ring_subgraph') or not event.xdata or not event.ydata:
            return
        
        # Get click coordinates
        x, y = event.xdata, event.ydata
        
        # Check if it's a right-click (for node selection)
        if event.button == 3:  # Right mouse button
            self.handle_node_click(x, y)
            return
        
        # Left-click - try edge first, then node (for backwards compatibility)
        elif event.button == 1:  # Left mouse button
            # Try edge selection first
            if self.handle_edge_click(x, y):
                return
            
            # If no edge was clicked, try node selection
            self.handle_node_click_left(x, y)
            return

    def handle_node_click_left(self, x, y):
        """Handle left-click for node selection (backwards compatibility)"""
        if not hasattr(self, 'node_positions') or not self.current_ring_subgraph:
            return False
        
        closest_node = None
        min_distance = float('inf')
        click_threshold = 0.05  # Threshold for node selection
        
        # Find the closest node to the click
        for node, pos in self.node_positions.items():
            distance = np.sqrt((pos[0] - x)**2 + (pos[1] - y)**2)
            if distance < min_distance and distance < click_threshold:
                min_distance = distance
                closest_node = node
        
        if closest_node:
            # Check if clicking on already selected node to deselect
            if (hasattr(self, 'selected_node') and 
                self.selected_node == closest_node):
                self.deselect_node()
                return True
            
            # Store the selected node
            self.selected_node = closest_node
            
            # Update the controls to reflect this selection
            self.select_node_in_controls(closest_node)
            
            # Show node details
            self.select_node(closest_node)
            
            # Highlight the selected node
            self.highlight_selected_node(closest_node)
            
            # Update status bar
            node_id = self.node_ids.get(closest_node, '?')
            self.statusBar().showMessage(f"Selected node {node_id}: {closest_node} (left-click)")
            return True
        else:
            # Clicked on empty space - deselect current node
            if hasattr(self, 'selected_node') and self.selected_node:
                self.deselect_node()
                return True
        
        return False

    def handle_node_click(self, x, y):
        """Handle right-click for node selection/deselection"""
        if not hasattr(self, 'node_positions') or not self.current_ring_subgraph:
            return
        
        closest_node = None
        min_distance = float('inf')
        click_threshold = 0.05  # Threshold for node selection
        
        # Find the closest node to the click
        for node, pos in self.node_positions.items():
            distance = np.sqrt((pos[0] - x)**2 + (pos[1] - y)**2)
            if distance < min_distance and distance < click_threshold:
                min_distance = distance
                closest_node = node
        
        if closest_node:
            # Check if clicking on already selected node to deselect
            if (hasattr(self, 'selected_node') and 
                self.selected_node == closest_node):
                self.deselect_node()
                return True
            
            # Store the selected node
            self.selected_node = closest_node
            
            # Update the controls to reflect this selection
            self.select_node_in_controls(closest_node)
            
            # Show node details
            self.select_node(closest_node)
            
            # Highlight the selected node
            self.highlight_selected_node(closest_node)
            
            # Update status bar
            node_id = self.node_ids.get(closest_node, '?')
            self.statusBar().showMessage(f"Selected node {node_id}: {closest_node} (right-click again to deselect)")
            return True
        else:
            # Clicked on empty space - deselect current node
            if hasattr(self, 'selected_node') and self.selected_node:
                self.deselect_node()
                return True
        
        return False

    def handle_edge_click(self, x, y):
        """Handle left-click for edge selection/deselection"""
        if not hasattr(self, 'current_ring_subgraph'):
            return False
        
        # Find closest edge with threshold
        min_dist = float('inf')
        closest_edge = None
        click_threshold = 0.03  # Smaller threshold for edge precision
        
        for (u, v) in self.current_ring_subgraph.edges():
            if u in self.node_positions and v in self.node_positions:
                pos_u = self.node_positions[u]
                pos_v = self.node_positions[v]
                
                # Calculate distance from click to edge
                dist = self.point_to_line_dist(
                    (x, y),
                    (pos_u[0], pos_u[1]),
                    (pos_v[0], pos_v[1])
                )
                
                if dist < min_dist and dist < click_threshold:
                    min_dist = dist
                    closest_edge = (u, v)
        
        if closest_edge:
            # Check if clicking on already selected edge to deselect
            if (hasattr(self, 'selected_edge') and 
                self.selected_edge and
                (self.selected_edge == closest_edge or 
                 self.selected_edge == (closest_edge[1], closest_edge[0]))):
                self.deselect_edge()
                return True
            
            self.selected_edge = closest_edge
            self.show_edge_details(closest_edge)
            
            # Auto-select the edge in the combo boxes
            self.auto_select_edge_in_combos(closest_edge)
            
            # Highlight the selected edge
            self.highlight_selected_edge(closest_edge)
            
            # Update status bar
            u, v = closest_edge
            self.statusBar().showMessage(f"Selected edge: {u} <-> {v} (left-click again to deselect)")
            return True
        else:
            # Clicked on empty space - deselect current edge
            if hasattr(self, 'selected_edge') and self.selected_edge:
                self.deselect_edge()
                return True
        
        return False

    def point_to_line_dist(self, point, line_start, line_end):
        """Calculate distance from point to line segment"""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Vector from line start to end
        line_vec = (x2-x1, y2-y1)
        # Vector from line start to point
        point_vec = (x-x1, y-y1)
        # Length of line
        line_len = np.sqrt(line_vec[0]**2 + line_vec[1]**2)
        
        if line_len == 0:
            return np.sqrt((x-x1)**2 + (y-y1)**2)
            
        # Project point onto line
        t = max(0, min(1, (point_vec[0]*line_vec[0] + point_vec[1]*line_vec[1])/line_len**2))
        
        # Get closest point on line
        proj_x = x1 + t*(x2-x1)
        proj_y = y1 + t*(y2-y1)
        
        return np.sqrt((x-proj_x)**2 + (y-proj_y)**2)

    def show_edge_details(self, edge):
        """Show details for the selected edge"""
        if not edge or not hasattr(self, 'details_text'):
            return
            
        u, v = edge
        
        # Format edge details
        details_text = f"<h3>Link Details</h3>"
        details_text += f"<p><b>End A:</b> {u}</p>"
        details_text += f"<p><b>End B:</b> {v}</p>"
        
        # Get edge attributes
        attrs = self.current_ring_subgraph.edges[edge]
        is_failed = attrs.get('failed', False)
        
        # Get edge history
        edge_key = tuple(sorted([u, v]))
        history = self.edge_history.get(edge_key, {
            'previous_failed': False,
            'current_failed': False,
            'failure_timestamp': None
        })
        
        details_text += "<h4>Link Status</h4><ul>"
        details_text += f"<li><b>Currently Failed:</b> {'Yes' if history['current_failed'] else 'No'}</li>"
        details_text += f"<li><b>Previously Failed:</b> {'Yes' if history['previous_failed'] else 'No'}</li>"
        if history['failure_timestamp']:
            details_text += f"<li><b>Last Status Change:</b> {history['failure_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</li>"
        details_text += "</ul>"
        
        self.details_text.setHtml(details_text)
        
        # Update checkboxes
        if hasattr(self, 'toggle_link_failed_checkbox'):
            self.toggle_link_failed_checkbox.setChecked(is_failed)
        
        # Update status bar
        self.statusBar().showMessage(f"Selected Link: {u} <-> {v}")

    def toggle_failed_status(self, state):
        """Toggle the failed status for the selected node"""
        if not hasattr(self, 'selected_node') or not self.selected_node:
            return
        
        node = self.selected_node
        sender = self.sender()  # Get the checkbox that triggered this
        
        # Initialize history if not exists
        if node not in self.node_history:
            self.node_history[node] = {
                'previous_failed': False,
                'current_failed': False,
                'failure_timestamp': None
            }
        
        # Update appropriate status based on which checkbox was clicked
        if sender == self.toggle_pfailed_checkbox:
            self.node_history[node]['previous_failed'] = bool(state)
            status_type = "previous"
        else:  # Current failed checkbox
            self.node_history[node]['current_failed'] = bool(state)
            status_type = "current"
        
        self.node_history[node]['failure_timestamp'] = datetime.now()
        
        # Update graph attributes
        self.current_ring_subgraph.nodes[node]['failed'] = bool(state)
        if node in self.graph:
            self.graph.nodes[node]['failed'] = bool(state)
        
        # Redraw and update
        self.redraw_visualization()
        self.select_node(node)
        
        # Update status message
        action = "marked as failed" if bool(state) else "marked as operational"
        self.statusBar().showMessage(f"Node {node} {action} ({status_type} state)")

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
        self.statusBar().showMessage(f"Node {self.selected_node} {action}")

    def redraw_visualization(self):
        """Redraw the current visualization to reflect changes"""
        if not hasattr(self, 'current_ring_subgraph') or self.current_ring_subgraph is None:
            return
            
        # Save current view limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Clear the axes
        self.ax.clear()
        
        # Draw edges with different colors for failure status
        if self.current_ring_subgraph.number_of_edges() > 0:
            normal_edges = []
            current_failed_edges = []
            previous_failed_edges = []
            highlighted_edges = []
            
            for u, v in self.current_ring_subgraph.edges():
                edge_key = tuple(sorted([u, v]))
                history = self.edge_history.get(edge_key, {
                    'previous_failed': False,
                    'current_failed': False
                })
                
                # Check if this edge is highlighted (selected)
                is_highlighted = (hasattr(self, 'highlighted_edge') and 
                                self.highlighted_edge and
                                (self.highlighted_edge == (u, v) or self.highlighted_edge == (v, u)))
                
                if is_highlighted:
                    highlighted_edges.append((u, v))
                elif history['current_failed'] or self.current_ring_subgraph.edges[u, v].get('failed', False):
                    current_failed_edges.append((u, v))
                elif history['previous_failed']:
                    previous_failed_edges.append((u, v))
                else:
                    normal_edges.append((u, v))
            
            # Draw edges in layers
            if normal_edges:
                nx.draw_networkx_edges(self.current_ring_subgraph, self.node_positions,
                                     edgelist=normal_edges, edge_color='black', width=1.0, ax=self.ax)
            if previous_failed_edges:
                nx.draw_networkx_edges(self.current_ring_subgraph, self.node_positions,
                                     edgelist=previous_failed_edges, edge_color='orange', 
                                     width=2.0, style='dashed', ax=self.ax)
            if current_failed_edges:
                nx.draw_networkx_edges(self.current_ring_subgraph, self.node_positions,
                                     edgelist=current_failed_edges, edge_color='red', 
                                     width=2.5, style='solid', ax=self.ax)
            if highlighted_edges:
                nx.draw_networkx_edges(self.current_ring_subgraph, self.node_positions,
                                     edgelist=highlighted_edges, edge_color='lime', 
                                     width=4.0, style='solid', ax=self.ax)
        
        # Draw nodes with different colors and highlighting
        if self.current_ring_subgraph.number_of_nodes() > 0:
            node_colors = []
            node_sizes = []
            edge_colors = []  # For node borders
            node_order = list(self.current_ring_subgraph.nodes())
            
            # Find block router node
            block_node = None
            for n, attrs in self.current_ring_subgraph.nodes(data=True):
                if attrs.get('is_block', False):
                    block_node = n
                    break
            
            for node in node_order:
                # Check if this node is highlighted (selected)
                is_highlighted = (hasattr(self, 'highlighted_node') and 
                                self.highlighted_node == node)
                
                # Determine node color
                if node == block_node:
                    node_colors.append('red')  # Block router
                    node_sizes.append(450)
                elif self.current_ring_subgraph.nodes[node].get('isolated', False):
                    node_colors.append('purple')  # Isolated node
                    node_sizes.append(300)
                elif self.current_ring_subgraph.nodes[node].get('node_not_reachable', False):
                    node_colors.append('gray')  # Node not reachable
                    node_sizes.append(300)
                elif self.current_ring_subgraph.nodes[node].get('ups_low_battery', False):
                    node_colors.append('orange')  # UPS alarm
                    node_sizes.append(300)
                else:
                    node_colors.append('skyblue')  # Default color
                    node_sizes.append(300)
                
                # Add thick border for highlighted nodes
                if is_highlighted:
                    edge_colors.append('yellow')
                    node_sizes[-1] = node_sizes[-1] + 100  # Make highlighted nodes bigger
                else:
                    edge_colors.append('black')
            
            # Draw the nodes
            node_collection = nx.draw_networkx_nodes(
                self.current_ring_subgraph, 
                pos=self.node_positions,
                ax=self.ax,
                node_color=node_colors,
                node_size=node_sizes,
                edgecolors=edge_colors,
                linewidths=3
            )
            
            # Set picker property for the old pick_event (keep as backup)
            if node_collection:
                node_collection.set_picker(5)
            
            # Add node labels
            if hasattr(self, 'node_ids'):
                labels = {node: str(self.node_ids.get(node, '?')) for node in self.current_ring_subgraph.nodes()}
                nx.draw_networkx_labels(self.current_ring_subgraph, self.node_positions, 
                                      labels, font_size=9, font_color='white', ax=self.ax)
        
        # Add legends
        legend_elements = [
            plt.Line2D([0], [0], color='black', lw=1, label='Normal Edge'),
            plt.Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Previous Failed Edge'),
            plt.Line2D([0], [0], color='red', lw=2.5, label='Current Failed Edge'),
            plt.Line2D([0], [0], color='lime', lw=4, label='Selected Edge (left-click)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', 
                      markersize=10, label='Normal Node'),
            plt.Line2D([0], [0], marker='o', color='yellow', markerfacecolor='skyblue', 
                      markersize=12, linewidth=3, label='Selected Node (right-click)')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=7)
        
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
            self.statusBar().showMessage("No topology graph available")
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
            self.statusBar().showMessage("Please enter both PR and LR values")
            return None
        
        # Find the block router in the current visualization
        block_node = None
        for node, attrs in self.current_ring_subgraph.nodes(data=True):
            if attrs.get('is_block', False):
                block_node = node
                break
        
        if not block_node:
            self.statusBar().showMessage("No block router found")
            return None
        
        # Get all nodes connected to the block router in the original topology graph
        connected_subgraph = nx.node_connected_component(self.graph, block_node)
        connected_subgraph = self.graph.subgraph(connected_subgraph).copy()
        
        # Identify which nodes belong to our current PR/LR
        current_ring_nodes = set(self.current_ring_subgraph.nodes())
        
        # Collect failed and isolated nodes from user modifications
        failed_nodes = []
        isolated_nodes = []
        
        # FIXED: Check both node history and current ring subgraph attributes
        for node in connected_subgraph.nodes():
            # Get history for this node
            history = self.node_history.get(node, {
                'previous_failed': False,
                'current_failed': False
            })
            
            # Check if node is failed (from history OR current ring attributes)
            is_failed = False
            if node in current_ring_nodes:
                # Check current ring subgraph attributes
                ring_attrs = self.current_ring_subgraph.nodes[node]
                is_failed = (ring_attrs.get('failed', False) or 
                            history.get('current_failed', False) or
                            ring_attrs.get('ups_low_battery', False) or
                            ring_attrs.get('node_not_reachable', False) or
                            ring_attrs.get('link_down', False))
            else:
                # For nodes not in current ring, just check history
                is_failed = history.get('current_failed', False)
            
            if is_failed:
                failed_nodes.append(node)
                connected_subgraph.nodes[node]['current_failed'] = True
            else:
                connected_subgraph.nodes[node]['current_failed'] = False
            
            # Set previous failed status
            connected_subgraph.nodes[node]['previous_failed'] = history.get('previous_failed', False)
            
            # Check if node is isolated
            is_isolated = False
            if node in current_ring_nodes:
                is_isolated = self.current_ring_subgraph.nodes[node].get('isolated', False)
            
            if is_isolated:
                isolated_nodes.append(node)
                connected_subgraph.nodes[node]['isolated'] = True
            else:
                connected_subgraph.nodes[node]['isolated'] = False
        
        # Create the PyG Data object
        nodes = list(connected_subgraph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Node features: [is_source, current_failed, previous_failed, is_ups, ups_low_battery_alarm_received]
        x = torch.zeros((len(nodes), 5), dtype=torch.float)
        
        for node in nodes:
            idx = node_to_idx[node]
            history = self.node_history.get(node, {
                'previous_failed': False,
                'current_failed': False
            })
            
            is_source = 1 if node == block_node else 0
            is_current_failed = 1 if connected_subgraph.nodes[node].get('current_failed', False) else 0
            is_previous_failed = 1 if connected_subgraph.nodes[node].get('previous_failed', False) else 0
            is_ups = 1 if connected_subgraph.nodes[node].get('is_ups', False) else 0
            ups_low_battery_alarm_received = 1 if connected_subgraph.nodes[node].get('ups_low_battery_alarm_received', False) else 0

            x[idx] = torch.tensor([
                is_source,
                is_current_failed,
                is_previous_failed,
                is_ups,
                ups_low_battery_alarm_received
            ], dtype=torch.float)
        
        # Edge index and attributes
        edge_index = []
        edge_attr = []
        
        for u, v in connected_subgraph.edges():
            # Add both directions for undirected graph
            edge_index.append([node_to_idx[u], node_to_idx[v]])
            edge_index.append([node_to_idx[v], node_to_idx[u]])
            
            # Edge features: [current_failed, previous_failed] - VERIFY THESE MATCH TRAINING
            edge_key = tuple(sorted([u, v]))
            edge_history = self.edge_history.get(edge_key, {
                'current_failed': False,
                'previous_failed': False
            })
            
            current_failed = 1.0 if edge_history.get('current_failed', False) else 0.0
            previous_failed = 1.0 if edge_history.get('previous_failed', False) else 0.0
            
            edge_attr.append([current_failed, previous_failed])
            edge_attr.append([current_failed, previous_failed])  # Same for reverse direction

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Target: isolated nodes
        y = torch.zeros(len(nodes), dtype=torch.float)
        for node in isolated_nodes:
            if node in node_to_idx:
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
        
        self.statusBar().showMessage(
            f"Created Data object with {len(nodes)} nodes ({len(connected_subgraph.edges())} edges), " 
            f"{len(failed_nodes)} failed, {len(isolated_nodes)} isolated"
        )
        
        current_failed_edges = int((edge_attr[:, 0].sum().item())) // 2  # Divide by 2 for undirected
        previous_failed_edges = int((edge_attr[:, 1].sum().item())) // 2
    
        self.statusBar().showMessage(
            f"Created Data object: {len(nodes)} nodes, {len(connected_subgraph.edges())} edges | "
            f"Failed: {len(failed_nodes)} nodes, {current_failed_edges} edges | "
            f"Isolated: {len(isolated_nodes)} nodes"
        )
        return data

    def on_create_block_data_clicked(self):
        """Handle click on create data from block components button"""
        data = self.create_data_object_from_block_components()
        if data:
            # Calculate edge statistics
            total_edges = data.edge_index.shape[1] // 2  # Divide by 2 for undirected
            failed_nodes = int(data.x[:, 1].sum().item())  # Current failed nodes
            previous_failed_nodes = int(data.x[:, 2].sum().item())  # Previous failed nodes
            isolated_nodes = int(data.y.sum().item())
            
            # Calculate failed edges
            current_failed_edges = int(data.edge_attr[:, 0].sum().item()) // 2  # Divide by 2 for undirected
            previous_failed_edges = int(data.edge_attr[:, 1].sum().item()) // 2  # Divide by 2 for undirected
            
            # Enhanced summary with all data
            summary = (f"Data Object Created:\n"
                      f"- PR: {data.pr_name}, LR: {data.lr_number}\n"
                      f"- Total Nodes: {data.num_nodes}\n"
                      f"- Current Failed Nodes: {failed_nodes}\n"
                      f"- Previous Failed Nodes: {previous_failed_nodes}\n"
                      f"- Isolated Nodes: {isolated_nodes}\n"
                      f"- Total Edges: {total_edges}\n"
                      f"- Current Failed Edges: {current_failed_edges}\n"
                      f"- Previous Failed Edges: {previous_failed_edges}\n"
                      f"- Normal Edges: {total_edges - current_failed_edges}")
            
            QMessageBox.information(self, "Block Components Data", summary)
            
            # Save to file
            filename = f"block_data_{data.pr_name}_{data.lr_number}.pt"
            torch.save(data, filename)
            self.statusBar().showMessage(f"Data saved to {filename}")
    
    def create_initial_model(self):
        """Create and save an initial model if none exists"""
        try:
            # Initialize new model
            model = IsolationGNN(
                num_node_features=5,
                num_edge_features=2,
                hidden_channels=32,
                num_layers=6
            )
            
            # Save state dict with proper extension
            model_path = self.model_path_entry.text()
            if not model_path.endswith('.pt'):
                model_path = model_path + '.pt'
                self.model_path_entry.setText(model_path)
        
            # Save only the state dict, not the full model
            torch.save(model.state_dict(), model_path)
            
            self.statusBar().showMessage(f"Created new model at {model_path}")
            return model_path
            
        except Exception as e:
            QMessageBox.critical(self, "Model Creation Error", 
                           f"Could not create model: {str(e)}")
            return None

    def load_gnn_model(self, model_path):
        """Load GNN model with backward compatibility"""
        try:
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "Model Load Error", f"Model file not found: {model_path}")
                return None
            
            # Create model with correct architecture
            model = IsolationGNN(
                num_node_features=5,
                num_edge_features=2,
                hidden_channels=32,
                num_layers=6
            )
            
            # Load the saved data
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            
            # Handle both checkpoint format and direct state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # New checkpoint format (from older versions of feedbackv4.py)
                model.load_state_dict(checkpoint['model_state_dict'])
                self.statusBar().showMessage(f"Loaded checkpoint model: {os.path.basename(model_path)}")
            else:
                # Direct state dict format (compatible with rcav8.py)
                model.load_state_dict(checkpoint)
                self.statusBar().showMessage(f"Loaded state dict model: {os.path.basename(model_path)}")
            
            model.eval()
            return model
            
        except Exception as e:
            QMessageBox.critical(self, "Model Load Error", 
                        f"Could not load model from {model_path}: {str(e)}")
            return None

    def save_model(self):
        """Save the trained model"""
        if not hasattr(self, 'trained_model') or not self.trained_model:
            QMessageBox.warning(self, "Save Error", "No trained model available to save")
            return
        
        try:
            # Ensure proper file extension
            model_path = self.original_model_path
            if not model_path.endswith('.pt'):
                model_path = model_path + '.pt'
                
            # Save complete checkpoint
            torch.save({
                'model_state_dict': self.trained_model.state_dict(),
                'num_node_features': 5,
                'num_edge_features': 2,
                'hidden_channels': 32,
                'num_layers': 18
            }, model_path)
            
            self.statusBar().showMessage(f"Model saved to {model_path}")
            QMessageBox.information(self, "Model Saved", 
                              f"Model successfully saved to {model_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save model: {str(e)}")

    def on_load_model_clicked(self):
        """Handle click on load model button"""
        model_path = self.model_path_entry.text()
        if not model_path:
            QMessageBox.warning(self, "Input Error", "Please select a model")
            return
        
        # Check if model file exists
        if not os.path.exists(model_path):
            reply = QMessageBox.question(self, "Model Not Found", 
                f"Model file not found: {model_path}\nCreate new model?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                
            if reply == QMessageBox.Yes:
                model_path = self.create_initial_model()
            else:
                return
        
        # Show loading message
        self.statusBar().showMessage(f"Loading model from {os.path.basename(model_path)}...")
        
        self.trained_model = self.load_gnn_model(model_path)
        if self.trained_model:
            self.train_button.setEnabled(True)
            self.original_model_path = model_path
            
            # Show success message with model info
            info = self.get_model_info(model_path)
            self.statusBar().showMessage(f"Model loaded: {os.path.basename(model_path)}")
            
            # Optionally show model info dialog
            QMessageBox.information(self, "Model Loaded Successfully", 
                                f"Model: {os.path.basename(model_path)}\n\n{info}")

    def on_train_model_clicked(self):
        """Handle click on train model button"""
        # First, create the data object from block components
        data = self.create_data_object_from_block_components()
        if not data:
            QMessageBox.warning(self, "Data Error", "Could not create data object from visualization")
            return
        
        # Check that we have a loaded model
        if not hasattr(self, 'trained_model') or not self.trained_model:
            # Create model with correct architecture
            self.trained_model = IsolationGNN(
                num_node_features=5,
                num_edge_features=2,
                hidden_channels=32,
                num_layers=6  # Match existing models
            )
        
        # Start training process
        self.statusBar().showMessage("Starting model training...")
        
        # Show data summary before training
        isolated_count = int(data.y.sum().item())
        total_nodes = data.num_nodes
        
        summary = (f"Training with:\n"
                  f"- {total_nodes} nodes\n"
                  f"- {isolated_count} isolated nodes ({isolated_count/total_nodes*100:.1f}%)\n"
                  f"- {data.edge_index.shape[1]//2} edges")
        QMessageBox.information(self, "Training Data Summary", summary)
        
        # Train the model
        self.trained_model = self.train_model(self.trained_model, data)
        
        # 🔧 ADD AUTO-SAVE AFTER TRAINING
        if self.trained_model:
            # Ask user if they want to save
            reply = QMessageBox.question(
                self, 
                "Save Trained Model", 
                "Training completed successfully!\n\nDo you want to save the trained model?",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.save_trained_model()

    def train_model(self, model, data, num_epochs=50, learning_rate=0.001):
        """Train the model on the provided data"""
        try:
            # Set model to training mode
            model.train()
            
            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # Binary cross entropy loss
            criterion = torch.nn.BCELoss()
            
            # Training loop
            for epoch in range(num_epochs):
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                out = model(data)
                
                # Calculate loss
                loss = criterion(out, data.y)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update status every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self.statusBar().showMessage(
                        f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}"
                    )
                    QApplication.processEvents()  # Keep UI responsive
            
            # Set model back to evaluation mode
            model.eval()
            
            # Enable save button
            if hasattr(self, 'save_model_button'):
                self.save_model_button.setEnabled(True)
            
            # Show completion message
            QMessageBox.information(
                self, 
                "Training Complete", 
                f"Model trained for {num_epochs} epochs\nFinal loss: {loss.item():.4f}\n\n"
                f"Click 'Save' to save the trained model."
            )
            
            return model
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Training Error", 
                f"Error during training: {str(e)}"
            )
            return None

    def update_edge_selection_combos(self):
        """Update the node lists in the edge selection combos"""
        if not hasattr(self, 'current_ring_subgraph') or not self.current_ring_subgraph:
            return
            
        # Clear existing items
        self.edge_node1_combo.clear()
        self.edge_node2_combo.clear()
        
        # Get sorted list of nodes
        nodes = sorted(list(self.current_ring_subgraph.nodes()))
        
        # Add nodes to combos
        for node in nodes:
            node_text = f"Node {self.node_ids.get(node, '?')} ({node})"
            self.edge_node1_combo.addItem(node_text, node)
            self.edge_node2_combo.addItem(node_text, node)

    def toggle_selected_edge(self):
        """Toggle failed status for the edge selected in combos"""
        if not hasattr(self, 'current_ring_subgraph') or not self.current_ring_subgraph:
            return
            
        # Get selected nodes
        node1 = self.edge_node1_combo.currentData()
        node2 = self.edge_node2_combo.currentData()
        
        if node1 == node2:
            QMessageBox.warning(self, "Invalid Selection", "Please select two different nodes")
            return
            
        if not self.current_ring_subgraph.has_edge(node1, node2):
            QMessageBox.warning(self, "Invalid Edge", "No direct connection exists between selected nodes")
            return
        
        # Get edge key
        edge_key = tuple(sorted([node1, node2]))
        
        # Get current edge history or initialize
        edge_history = self.edge_history.get(edge_key, {
            'previous_failed': False,
            'current_failed': False,
            'failure_timestamp': None
        })
        
        # Toggle current status
        new_current_status = not edge_history['current_failed']
        
        # FIXED: Preserve previous_failed, only update current_failed
        self.edge_history[edge_key] = {
            'previous_failed': edge_history['previous_failed'],  # Keep existing
            'current_failed': new_current_status,               # Update current
            'failure_timestamp': datetime.now()
        }
        
        # Update graph attributes
        self.current_ring_subgraph.edges[node1, node2]['failed'] = new_current_status
        
        # Update visualization
        self.redraw_visualization()
        
        # Update status bar
        action = "marked as failed" if new_current_status else "marked as operational"
        self.statusBar().showMessage(f"Edge {node1} <-> {node2} {action}")

    def undo_edge_failures(self):
        """Clear all edge failures in the current visualization"""
        if not hasattr(self, 'current_ring_subgraph') or not self.current_ring_subgraph:
            return
        
        # Clear all edge failures
        for u, v in self.current_ring_subgraph.edges():
            edge_key = tuple(sorted([u, v]))
            
            # Update edge attributes
            self.current_ring_subgraph.edges[u, v]['failed'] = False
            self.current_ring_subgraph.edges[v, u]['failed'] = False
            
            # Update edge history
            if edge_key in self.edge_history:
                self.edge_history[edge_key]['current_failed'] = False
                self.edge_history[edge_key]['failure_timestamp'] = datetime.now()
    
        # Update visualization
        self.redraw_visualization()
        
        # Update status bar
        self.statusBar().showMessage("All edge failures cleared")
        
        # Update edge details if an edge is currently selected
        if hasattr(self, 'selected_edge') and self.selected_edge:
            self.show_edge_details(self.selected_edge)
    
    def on_alarm_changed(self, state):
        """Handle alarm type checkbox changes"""
        if not hasattr(self, 'selected_node') or not self.selected_node:
            return
            
        sender = self.sender()
        
        # Uncheck other alarm type checkboxes
        if state:
            if sender != self.ups_alarm_checkbox:
                self.ups_alarm_checkbox.setChecked(False)
            if sender != self.node_alarm_checkbox:
                self.node_alarm_checkbox.setChecked(False)
            if sender != self.link_alarm_checkbox:
                self.link_alarm_checkbox.setChecked(False)
                
        self.apply_alarm_changes()

    def on_alarm_status_changed(self, state):
        """Handle alarm status checkbox changes"""
        if not hasattr(self, 'selected_node') or not self.selected_node:
            return
            
        sender = self.sender()
        
        # Uncheck other status checkboxes
        if state:
            if sender != self.current_alarm_checkbox:
                self.current_alarm_checkbox.setChecked(False)
            if sender != self.previous_alarm_checkbox:
                self.previous_alarm_checkbox.setChecked(False)
            if sender != self.isolated_alarm_checkbox:
                self.isolated_alarm_checkbox.setChecked(False)
                
        self.apply_alarm_changes()

    def apply_alarm_changes(self):
        """Apply the alarm changes based on checkbox states"""
        if not hasattr(self, 'selected_node') or not self.selected_node:
            return
            
        node = self.selected_node
        
        # Clear previous alarms
        self.current_ring_subgraph.nodes[node].update({
            'ups_low_battery': False,
            'node_not_reachable': False,
            'link_down': False,
            'isolated': False
        })
        
        # Initialize node history if needed
        if node not in self.node_history:
            self.node_history[node] = {
                'previous_failed': False,
                'current_failed': False,
                'link_down': False,
                'ups_low_battery': False,
                'node_not_reachable': False
            }
        
        # Apply new alarm state
        if self.ups_alarm_checkbox.isChecked():
            alarm_attr = 'ups_low_battery'
        elif self.node_alarm_checkbox.isChecked():
            alarm_attr = 'node_not_reachable'
        elif self.link_alarm_checkbox.isChecked():
            alarm_attr = 'link_down'
        else:
            return
            
        # Apply status
        if self.current_alarm_checkbox.isChecked():
            self.node_history[node]['current_failed'] = True
            self.current_ring_subgraph.nodes[node][alarm_attr] = True
        elif self.previous_alarm_checkbox.isChecked():
            self.node_history[node]['previous_failed'] = True
            self.current_ring_subgraph.nodes[node][alarm_attr] = True
        elif self.isolated_alarm_checkbox.isChecked():
            self.current_ring_subgraph.nodes[node]['isolated'] = True
        
        # Record timestamp
        self.node_history[node]['alarm_timestamp'] = datetime.now()
        
        # Update visualization
        self.redraw_visualization()
        
        # Update details panel
        self.select_node(node)
        
    # Add this method to the RingVisualizerApp class
    def simulate_ups_alarm(self, node):
        """Simulate a UPS alarm for the given node"""
        if not node in self.current_ring_subgraph:
            return False
            
        # Check if node has UPS
        if not self.current_ring_subgraph.nodes[node].get('is_ups', False):
            QMessageBox.warning(self, "UPS Alarm Error", "Selected node does not have a UPS")
            return False
        
        # Set UPS alarm attributes
        self.current_ring_subgraph.nodes[node].update({
            'ups_low_battery': True,
            'ups_alarm': True
        })
        
        # Update node history
        if node not in self.node_history:
            self.node_history[node] = {}
        
        self.node_history[node].update({
            'ups_low_battery': True,
            'alarm_timestamp': datetime.now()
        })
        
        # Auto-check appropriate checkboxes
        self.ups_alarm_checkbox.setChecked(True)
        self.current_alarm_checkbox.setChecked(True)
        
        # Update visualization
        self.redraw_visualization()
        self.select_node(node)
        
        # Update status bar
        self.statusBar().showMessage(f"UPS alarm simulated for node {node}")
        return True

    def create_initial_model(self):
        """Create and save an initial model if none exists"""
        try:
            # Initialize new model
            model = IsolationGNN(
                num_node_features=5,
                num_edge_features=2,
                hidden_channels=32,
                num_layers=6
            )
            
            # Save state dict with proper extension
            model_path = self.model_path_entry.text()
            if not model_path.endswith('.pt'):
                model_path = model_path + '.pt'
                self.model_path_entry.setText(model_path)
        
            # Save only the state dict, not the full model
            torch.save(model.state_dict(), model_path)
            
            self.statusBar().showMessage(f"Created new model at {model_path}")
            return model_path
            
        except Exception as e:
            QMessageBox.critical(self, "Model Creation Error", 
                           f"Could not create model: {str(e)}")
            return None

    def on_edge_status_changed(self, state):
        """Handle edge status checkbox changes"""
        node1 = self.edge_node1_combo.currentData()
        node2 = self.edge_node2_combo.currentData()
        
        if not node1 or not node2 or node1 == node2:
            return
        
        # Check if we have a valid graph and edge exists
        if not hasattr(self, 'current_ring_subgraph') or not self.current_ring_subgraph:
            return
            
        if not self.current_ring_subgraph.has_edge(node1, node2):
            QMessageBox.warning(self, "Invalid Edge", 
                              f"No direct connection exists between {node1} and {node2}")
            return
            
        edge_key = tuple(sorted([node1, node2]))
        
        # Initialize if needed
        if edge_key not in self.edge_history:
            self.edge_history[edge_key] = {
                'previous_failed': False,
                'current_failed': False,
                'failure_timestamp': None
            }
        
        # Update based on checkboxes
        self.edge_history[edge_key]['previous_failed'] = self.edge_previous_failed_checkbox.isChecked()
        self.edge_history[edge_key]['current_failed'] = self.edge_current_failed_checkbox.isChecked()
        self.edge_history[edge_key]['failure_timestamp'] = datetime.now()
        
        # Update graph attributes for compatibility - now safe to access
        failed_status = self.edge_history[edge_key]['current_failed']
        self.current_ring_subgraph.edges[node1, node2]['failed'] = failed_status
        
        # Update visualization
        self.redraw_visualization()
        
        # Update status bar
        status = []
        if self.edge_history[edge_key]['current_failed']:
            status.append("current failed")
        if self.edge_history[edge_key]['previous_failed']:
            status.append("previous failed")
        if not status:
            status = ["normal"]
        
        self.statusBar().showMessage(f"Edge {node1} <-> {node2}: {', '.join(status)}")

    def highlight_selected_node(self, node):
        """Highlight the selected node in the visualization"""
        if not hasattr(self, 'current_ring_subgraph'):
            return
            
        # Store the selected node for highlighting in redraw
        self.highlighted_node = node
        self.redraw_visualization()

    def select_node_in_controls(self, node):
        """Update all control elements to reflect the selected node"""
        if not hasattr(self, 'current_ring_subgraph') or node not in self.current_ring_subgraph:
            return
        
        # Update node selector list
        if hasattr(self, 'node_selector'):
            # Clear current selection
            self.node_selector.clearSelection()
            
            # Find and select the node in the list
            for i in range(self.node_selector.count()):
                item = self.node_selector.item(i)
                if item and item.data(Qt.UserRole) == node:
                    self.node_selector.setCurrentItem(item)
                    break
        
        # Update table selection
        if hasattr(self, 'table'):
            # Find the node in the table and select its row
            for row in range(self.table.rowCount()):
                ip_item = self.table.item(row, 1)  # IP column
                if ip_item and ip_item.text() == node:
                    self.table.selectRow(row)
                    break
        
        # Update alarm checkboxes based on node state
        self.update_alarm_checkboxes_from_node(node)

    def update_alarm_checkboxes_from_node(self, node):
        """Update alarm checkboxes based on selected node's current state"""
        if not hasattr(self, 'current_ring_subgraph') or node not in self.current_ring_subgraph:
            return
        
        # Get node attributes and history
        attrs = self.current_ring_subgraph.nodes[node]
        history = self.node_history.get(node, {
            'previous_failed': False,
            'current_failed': False
        })
        
        # Clear all checkboxes first
        self.ups_alarm_checkbox.setChecked(False)
        self.node_alarm_checkbox.setChecked(False)
        self.link_alarm_checkbox.setChecked(False)
        self.current_alarm_checkbox.setChecked(False)
        self.previous_alarm_checkbox.setChecked(False)
        self.isolated_alarm_checkbox.setChecked(False)
        
        # Set appropriate alarm type
        if attrs.get('ups_low_battery', False):
            self.ups_alarm_checkbox.setChecked(True)
        elif attrs.get('node_not_reachable', False):
            self.node_alarm_checkbox.setChecked(True)
        elif attrs.get('link_down', False):
            self.link_alarm_checkbox.setChecked(True)
        
        # Set appropriate status
        if attrs.get('isolated', False):
            self.isolated_alarm_checkbox.setChecked(True)
        elif history.get('current_failed', False):
            self.current_alarm_checkbox.setChecked(True)
        elif history.get('previous_failed', False):
            self.previous_alarm_checkbox.setChecked(True)

    def on_edge_combo_changed(self):
        """Handle changes in edge selection combo boxes"""
        # Update the checkboxes when combo selection changes
        self.update_edge_checkboxes_from_selection()
        
        # Show details for the selected edge
        node1 = self.edge_node1_combo.currentData()
        node2 = self.edge_node2_combo.currentData()
        
        if node1 and node2 and node1 != node2:
            if hasattr(self, 'current_ring_subgraph') and self.current_ring_subgraph.has_edge(node1, node2):
                self.show_edge_details((node1, node2))

    def deselect_node(self):
        """Deselect the currently selected node"""
        # Clear all node-related selections
        if hasattr(self, 'selected_node'):
            self.selected_node = None
        
        if hasattr(self, 'highlighted_node'):
            self.highlighted_node = None
        
        # Clear table selection (disconnect signal temporarily to avoid recursion)
        if hasattr(self, 'table'):
            self.table.selectionModel().selectionChanged.disconnect()
            self.table.clearSelection()
            self.table.selectionModel().selectionChanged.connect(self.on_table_selection_changed)
        
        # Clear node list selection
        if hasattr(self, 'node_selector'):
            self.node_selector.clearSelection()
        
        # Clear alarm checkboxes
        self.clear_alarm_checkboxes()
        
        # Clear details text
        if hasattr(self, 'details_text'):
            self.details_text.clear()
        
        # Update visualization
        self.redraw_visualization()
        
        # Update status bar
        self.statusBar().showMessage("Node deselected")

    def deselect_edge(self):
        """Deselect the currently selected edge"""
        # Clear all edge-related selections
        if hasattr(self, 'selected_edge'):
            self.selected_edge = None
        
        if hasattr(self, 'highlighted_edge'):
            self.highlighted_edge = None
        
        # Clear combo box selections (disconnect signals temporarily)
        if hasattr(self, 'edge_node1_combo'):
            self.edge_node1_combo.currentIndexChanged.disconnect()
            self.edge_node1_combo.setCurrentIndex(-1)
            self.edge_node1_combo.currentIndexChanged.connect(self.on_edge_combo_changed)
            
        if hasattr(self, 'edge_node2_combo'):
            self.edge_node2_combo.currentIndexChanged.disconnect()
            self.edge_node2_combo.setCurrentIndex(-1)
            self.edge_node2_combo.currentIndexChanged.connect(self.on_edge_combo_changed)
        
        # Clear edge status checkboxes (disconnect signals temporarily)
        if hasattr(self, 'edge_previous_failed_checkbox'):
            self.edge_previous_failed_checkbox.stateChanged.disconnect()
            self.edge_previous_failed_checkbox.setChecked(False)
            self.edge_previous_failed_checkbox.stateChanged.connect(self.on_edge_status_changed)
            
        if hasattr(self, 'edge_current_failed_checkbox'):
            self.edge_current_failed_checkbox.stateChanged.disconnect()
            self.edge_current_failed_checkbox.setChecked(False)
            self.edge_current_failed_checkbox.stateChanged.connect(self.on_edge_status_changed)
        
        # Clear details text
        if hasattr(self, 'details_text'):
            self.details_text.clear()
        
        # Update visualization
        self.redraw_visualization()
        
        # Update status bar
        self.statusBar().showMessage("Edge deselected")

    def clear_alarm_checkboxes(self):
        """Clear all alarm checkboxes"""
        self.ups_alarm_checkbox.setChecked(False)
        self.node_alarm_checkbox.setChecked(False)
        self.link_alarm_checkbox.setChecked(False)
        self.current_alarm_checkbox.setChecked(False)
        self.previous_alarm_checkbox.setChecked(False)
        self.isolated_alarm_checkbox.setChecked(False)

    def deselect_all(self):
        """Deselect both nodes and edges"""
        self.deselect_node()
        self.deselect_edge()
        self.statusBar().showMessage("All selections cleared")

    def populate_model_combo(self):
        """Populate the model combo box with available models"""
        # Clear existing items
        self.model_combo.clear()
        
        # Define models directory
        models_dir = "models"
        
        # Add default option
        self.model_combo.addItem("Select Model...", "")
        
        # Check if models directory exists
        if os.path.exists(models_dir) and os.path.isdir(models_dir):
            try:
                # Get all .pt files in models directory
                model_files = []
                for file in os.listdir(models_dir):
                    if file.endswith('.pt') or file.endswith('.pth'):
                        model_files.append(file)
                
                # Sort the files
                model_files.sort()
                
                # Add models to combo box
                for model_file in model_files:
                    model_path = os.path.join(models_dir, model_file)
                    # Display just filename, store full path
                    self.model_combo.addItem(model_file, model_path)
                
                if model_files:
                    self.statusBar().showMessage(f"Found {len(model_files)} model(s) in {models_dir}/")
                else:
                    self.statusBar().showMessage(f"No models found in {models_dir}/")
                    
            except Exception as e:
                self.statusBar().showMessage(f"Error reading models directory: {str(e)}")
                
        else:
            # Create models directory if it doesn't exist
            try:
                os.makedirs(models_dir, exist_ok=True)
                self.statusBar().showMessage(f"Created {models_dir}/ directory")
            except Exception as e:
                self.statusBar().showMessage(f"Could not create {models_dir}/ directory: {str(e)}")
        
        # Add option for custom path
        self.model_combo.addItem("Custom Path...", "custom")

    def on_model_combo_changed(self, text):
        """Handle model combo box selection change"""
        current_data = self.model_combo.currentData()
        
        if current_data == "custom":
            # Enable text entry for custom path
            self.model_path_entry.setEnabled(True)
            self.model_path_entry.setFocus()
            self.statusBar().showMessage("Enter custom model path")
            
        elif current_data and current_data != "":
            # Update text entry with selected model path
            self.model_path_entry.setText(current_data)
            self.model_path_entry.setEnabled(False)
            
            # Show model info
            try:
                file_size = os.path.getsize(current_data)
                file_size_mb = file_size / (1024 * 1024)
                self.statusBar().showMessage(f"Selected: {text} ({file_size_mb:.1f} MB)")
            except:
                self.statusBar().showMessage(f"Selected: {text}")
        else:
            # No selection
            self.model_path_entry.setEnabled(True)
            self.statusBar().showMessage("No model selected")

    def get_model_info(self, model_path):
        """Get information about a model file"""
        try:
            if not os.path.exists(model_path):
                return "Model file not found"
            
            # Get file size
            file_size = os.path.getsize(model_path)
            file_size_mb = file_size / (1024 * 1024)
            
            # Try to load model to get architecture info
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        # Old checkpoint format
                        info = f"Size: {file_size_mb:.1f} MB\n"
                        info += f"Format: Checkpoint (old format)\n"
                        info += f"Node features: {checkpoint.get('num_node_features', 'Unknown')}\n"
                        info += f"Edge features: {checkpoint.get('num_edge_features', 'Unknown')}\n"
                        info += f"Hidden channels: {checkpoint.get('hidden_channels', 'Unknown')}\n"
                        info += f"Layers: {checkpoint.get('num_layers', 'Unknown')}"
                        return info
                    else:
                        # New state dict format (compatible with rcav8.py)
                        info = f"Size: {file_size_mb:.1f} MB\n"
                        info += f"Format: State dict (rcav8.py compatible)\n"
                        info += f"Parameters: {len(checkpoint)} tensors"
                        return info
                else:
                    return f"Size: {file_size_mb:.1f} MB\nFormat: Full model object"
                    
            except Exception as e:
                return f"Size: {file_size_mb:.1f} MB\nCould not read model details: {str(e)}"
                
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def save_trained_model(self):
        """Save the trained model in the format expected by rcav8.py"""
        if not hasattr(self, 'trained_model') or not self.trained_model:
            QMessageBox.warning(self, "Save Error", "No trained model available to save")
            return
        
        try:
            # Get original model path or create new one
            if hasattr(self, 'original_model_path') and self.original_model_path:
                base_path = self.original_model_path
            else:
                base_path = self.model_path_entry.text()
            
            # Create timestamped filename for backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(base_path)[0]
            backup_path = f"{base_name}_backup_{timestamp}.pt"
            
            # Ask user for save options
            reply = QMessageBox.question(
                self, 
                "Save Options", 
                f"Choose save option:\n\n"
                f"Yes = Overwrite original ({os.path.basename(base_path)})\n"
                f"No = Save as backup ({os.path.basename(backup_path)})\n"
                f"Cancel = Don't save",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.No  # Default to backup
            )
            
            if reply == QMessageBox.Cancel:
                return
            elif reply == QMessageBox.Yes:
                save_path = base_path
                save_type = "original"
            else:  # No = backup
                save_path = backup_path
                save_type = "backup"
            
            # Ensure proper file extension
            if not save_path.endswith('.pt'):
                save_path = save_path + '.pt'
            
            # 🔧 CHANGED: Save only state dict (not full checkpoint) to match rcav8.py expectations
            torch.save(self.trained_model.state_dict(), save_path)
            
            # Update the model path entry if we saved as original
            if reply == QMessageBox.Yes:
                self.model_path_entry.setText(save_path)
                self.original_model_path = save_path
            
            # Show success message
            self.statusBar().showMessage(f"Model saved as {save_type}: {os.path.basename(save_path)}")
            QMessageBox.information(
                self, 
                "Model Saved Successfully", 
                f"Model saved as {save_type}:\n{save_path}\n\n"
                f"File size: {os.path.getsize(save_path) / (1024*1024):.1f} MB\n"
                f"Format: State dict (compatible with rcav8.py)"
            )
            
            # Refresh model combo if saved to models directory
            if save_path.startswith('models/'):
                self.populate_model_combo()
                
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save model: {str(e)}")
            logging.error(f"Model save error: {str(e)}")

def main():
    """Main application entry point"""
    # Set up logging
    setup_logging()
    
    # Set up database and get topology
    try:
        mydb, cursor = setup_database()
        topology_graph = load_topology(cursor)
        block_router_ips = identify_block_routers(cursor, topology_graph)
        topology_graph = assign_nodes_to_blocks(topology_graph, block_router_ips)
        topology_graph = add_ups_devices(cursor, topology_graph)
        
        # Create and show the application
        app = QApplication(sys.argv)
        main_window = RingVisualizerApp(topology_graph)
        main_window.show()
        sys.exit(app.exec_())
        
    except Exception as e:
        logging.error(f"Application startup error: {str(e)}")
        sys.exit(1)
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'mydb' in locals():
            mydb.close()

if __name__ == '__main__':
    main()
