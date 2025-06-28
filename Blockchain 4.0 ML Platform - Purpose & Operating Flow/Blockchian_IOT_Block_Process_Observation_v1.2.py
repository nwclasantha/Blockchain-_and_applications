import sys
import hashlib
import time
import threading
import random
import json
import sqlite3
import traceback
from datetime import datetime, timedelta
from queue import Queue, PriorityQueue
from collections import deque

# Core PyQt5 imports with error handling
try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    print("PyQt5 core loaded successfully")
except ImportError as e:
    print(f"Error: PyQt5 not installed. Please run: pip install PyQt5")
    sys.exit(1)

# Optional imports with fallbacks
try:
    from PyQt5.QtChart import *
    HAS_CHARTS = True
    print("PyQt5 Charts loaded")
except ImportError:
    HAS_CHARTS = False
    print("Warning: PyQt5 Charts not available. Install with: pip install PyQtChart")

try:
    import numpy as np
    HAS_NUMPY = True
    print("NumPy loaded")
except ImportError:
    HAS_NUMPY = False
    print("Warning: NumPy not available. Some features will be limited.")

try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
    print("Scikit-learn loaded")
except ImportError:
    HAS_SKLEARN = False
    print("Warning: Scikit-learn not available. AI features will be limited.")

# Enable high DPI support
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

# ===================== Blockchain Core (No External Dependencies) =====================
class Transaction:
    def __init__(self, sensor_id, value, timestamp, transaction_type="sensor_data"):
        self.sensor_id = sensor_id
        self.value = value
        self.timestamp = timestamp
        self.type = transaction_type
        self.signature = self.generate_signature()
        
    def generate_signature(self):
        data = f"{self.sensor_id}{self.value}{self.timestamp}{self.type}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def to_dict(self):
        return {
            'sensor_id': self.sensor_id,
            'value': self.value,
            'timestamp': self.timestamp,
            'type': self.type,
            'signature': self.signature
        }

class Block:
    def __init__(self, index, timestamp, transactions, previous_hash, miner_id="system", nonce=0):
        self.index = index
        self.timestamp = timestamp
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.miner_id = miner_id
        self.nonce = nonce
        self.hash = self.compute_hash()
        
    def compute_hash(self):
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() if hasattr(tx, 'to_dict') else tx for tx in self.transactions],
            'previous_hash': self.previous_hash,
            'miner_id': self.miner_id,
            'nonce': self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty=2):
        target = '0' * difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.compute_hash()
        return self.hash
    
    def to_dict(self):
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() if hasattr(tx, 'to_dict') else tx for tx in self.transactions],
            'previous_hash': self.previous_hash,
            'miner_id': self.miner_id,
            'nonce': self.nonce,
            'hash': self.hash
        }

class Blockchain:
    def __init__(self, difficulty=2):
        self.chain = []
        self.difficulty = difficulty
        self.pending_transactions = []
        self.mining_reward = 100
        self.create_genesis_block()
        
    def create_genesis_block(self):
        genesis = Block(0, time.time(), [], '0', 'genesis')
        genesis.hash = genesis.mine_block(self.difficulty)
        self.chain.append(genesis)
        
    def add_transaction(self, transaction):
        self.pending_transactions.append(transaction)
        
    def mine_pending_transactions(self, miner_id):
        if not self.pending_transactions:
            return None
            
        block = Block(
            len(self.chain),
            time.time(),
            self.pending_transactions[:10],  # Limit transactions per block
            self.chain[-1].hash,
            miner_id
        )
        
        block.hash = block.mine_block(self.difficulty)
        self.chain.append(block)
        
        # Remove mined transactions
        self.pending_transactions = self.pending_transactions[10:]
        
        return block
    
    def validate_chain(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            if current.previous_hash != previous.hash:
                return False
                
            if current.hash != current.compute_hash():
                return False
                
        return True

# ===================== Simple Anomaly Detector =====================
class SimpleAnomalyDetector:
    def __init__(self):
        self.history = deque(maxlen=50)
        self.threshold_multiplier = 2.5
        
    def detect_anomaly(self, value):
        if len(self.history) < 10:
            self.history.append(value)
            return False, 0.0
            
        # Simple statistical anomaly detection
        mean = sum(self.history) / len(self.history)
        std = (sum((x - mean) ** 2 for x in self.history) / len(self.history)) ** 0.5
        
        # Calculate z-score
        z_score = abs(value - mean) / (std + 0.0001)  # Avoid division by zero
        is_anomaly = z_score > self.threshold_multiplier
        
        # Normalize score to 0-1 range
        anomaly_score = min(z_score / (self.threshold_multiplier * 2), 1.0)
        
        self.history.append(value)
        return is_anomaly, anomaly_score

# ===================== Custom Widgets =====================
class MetricCard(QFrame):
    def __init__(self, title, value="0", subtitle="", color="#00ff88"):
        super().__init__()
        self.title = title
        self.color = color
        self.init_ui()
        self.set_value(value, subtitle)
        
    def init_ui(self):
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #1e1e2e;
                border: 2px solid {self.color};
                border-radius: 10px;
                padding: 10px;
            }}
            QLabel {{
                color: #ffffff;
            }}
        """)
        
        layout = QVBoxLayout()
        
        title_label = QLabel(self.title)
        title_label.setStyleSheet("font-size: 12px; color: #888888;")
        layout.addWidget(title_label)
        
        self.value_label = QLabel("0")
        self.value_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {self.color};")
        layout.addWidget(self.value_label)
        
        self.subtitle_label = QLabel("")
        self.subtitle_label.setStyleSheet("font-size: 10px; color: #aaaaaa;")
        layout.addWidget(self.subtitle_label)
        
        self.setLayout(layout)
        
    def set_value(self, value, subtitle=None):
        self.value_label.setText(str(value))
        if subtitle is not None:
            self.subtitle_label.setText(subtitle)

class BlockWidget(QFrame):
    def __init__(self, block_data, parent=None):
        super().__init__(parent)
        self.block_data = block_data
        self.init_ui()
        
    def init_ui(self):
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet("""
            QFrame {
                background-color: #1e1e2e;
                border: 2px solid #00ff88;
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
            }
            QLabel {
                color: #ffffff;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Block header
        header_layout = QHBoxLayout()
        block_num = QLabel(f"Block #{self.block_data['index']}")
        block_num.setStyleSheet("font-size: 16px; font-weight: bold; color: #00ff88;")
        header_layout.addWidget(block_num)
        header_layout.addStretch()
        
        # Check for anomalies in transactions
        has_anomaly = False
        if 'transactions' in self.block_data:
            for tx in self.block_data['transactions']:
                if isinstance(tx, dict) and tx.get('is_anomaly', False):
                    has_anomaly = True
                    break
        
        if has_anomaly:
            anomaly_label = QLabel("⚠️ ANOMALY")
            anomaly_label.setStyleSheet("color: #ff4444; font-weight: bold;")
            header_layout.addWidget(anomaly_label)
        
        layout.addLayout(header_layout)
        
        # Block details
        details = QLabel(f"""
Timestamp: {datetime.fromtimestamp(self.block_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}
Hash: {self.block_data['hash'][:16]}...
Previous: {self.block_data['previous_hash'][:16]}...
Miner: {self.block_data['miner_id']}
Transactions: {len(self.block_data.get('transactions', []))}
        """)
        details.setWordWrap(True)
        details.setStyleSheet("font-family: monospace; font-size: 10px;")
        layout.addWidget(details)
        
        self.setLayout(layout)

# ===================== Simple Chart Widget =====================
class SimpleChart(QWidget):
    def __init__(self, title="Chart"):
        super().__init__()
        self.title = title
        self.data = []
        self.max_points = 100
        self.init_ui()
        
    def init_ui(self):
        self.setMinimumHeight(200)
        self.setStyleSheet("background-color: #1e1e2e; border: 1px solid #00ff88; border-radius: 5px;")
        
    def add_data(self, value):
        self.data.append(value)
        if len(self.data) > self.max_points:
            self.data.pop(0)
        self.update()
        
    def paintEvent(self, event):
        if not self.data:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(30, 30, 46))
        
        # Draw title
        painter.setPen(QPen(QColor(0, 255, 136), 2))
        painter.drawText(10, 20, self.title)
        
        # Draw grid
        painter.setPen(QPen(QColor(100, 100, 100), 1, Qt.DotLine))
        for i in range(0, self.height(), 40):
            painter.drawLine(0, i, self.width(), i)
            
        # Draw data
        if len(self.data) > 1:
            painter.setPen(QPen(QColor(0, 255, 255), 2))
            
            # Calculate points
            width = self.width() - 20
            height = self.height() - 40
            max_val = max(self.data) if self.data else 1
            min_val = min(self.data) if self.data else 0
            range_val = max_val - min_val if max_val != min_val else 1
            
            points = []
            for i, value in enumerate(self.data):
                x = 10 + (i * width / (len(self.data) - 1))
                y = 30 + height - ((value - min_val) / range_val * height)
                points.append(QPointF(x, y))
            
            # Draw lines
            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i + 1])
                
            # Draw points
            painter.setPen(QPen(QColor(0, 255, 136), 3))
            for point in points[-5:]:  # Only draw last 5 points
                painter.drawEllipse(point, 3, 3)

# ===================== Data Thread =====================
class DataThread(QThread):
    new_data = pyqtSignal(dict)
    new_block = pyqtSignal(dict)
    
    def __init__(self, blockchain, detector):
        super().__init__()
        self.blockchain = blockchain
        self.detector = detector
        self.running = True
        self.sensors = [f"sensor_{i}" for i in range(5)]
        self.transaction_count = 0
        
    def run(self):
        while self.running:
            try:
                # Generate data for each sensor
                for sensor_id in self.sensors:
                    # Generate sensor value
                    base_value = 25 + random.gauss(0, 3)
                    if random.random() < 0.1:  # 10% chance of anomaly
                        value = base_value + random.uniform(10, 20)
                    else:
                        value = base_value
                    
                    # Detect anomaly
                    is_anomaly, anomaly_score = self.detector.detect_anomaly(value)
                    
                    # Create transaction
                    tx = Transaction(sensor_id, value, time.time())
                    self.blockchain.add_transaction(tx)
                    
                    # Emit signal
                    self.new_data.emit({
                        'sensor_id': sensor_id,
                        'value': value,
                        'is_anomaly': is_anomaly,
                        'anomaly_score': anomaly_score,
                        'timestamp': time.time(),
                        'transaction': tx.to_dict()
                    })
                    
                    self.transaction_count += 1
                    
                    # Mine block every 10 transactions
                    if self.transaction_count >= 10:
                        block = self.blockchain.mine_pending_transactions("auto_miner")
                        if block:
                            self.new_block.emit(block.to_dict())
                        self.transaction_count = 0
                    
                    self.msleep(100)  # Small delay between sensors
                    
            except Exception as e:
                print(f"Error in data thread: {e}")
                
            self.msleep(500)  # Main loop delay
    
    def stop(self):
        self.running = False

# ===================== Main GUI Application =====================
class BlockchainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.blockchain = Blockchain()
        self.detector = SimpleAnomalyDetector()
        self.init_ui()
        self.setup_data_thread()
        
    def init_ui(self):
        self.setWindowTitle("Blockchain 4.0 - IoT & AI Integration Platform")
        self.setGeometry(100, 100, 1400, 800)
        
        # Set style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a0a0a;
            }
            QWidget {
                background-color: #1a1a2e;
                color: #ffffff;
                font-family: Arial, sans-serif;
            }
            QPushButton {
                background-color: #2e2e3e;
                border: 2px solid #00ff88;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
                color: #00ff88;
            }
            QPushButton:hover {
                background-color: #3e3e4e;
            }
            QPushButton:pressed {
                background-color: #00ff88;
                color: #000000;
            }
            QTabWidget::pane {
                border: 2px solid #00ff88;
                background-color: #1a1a2e;
            }
            QTabBar::tab {
                background-color: #2e2e3e;
                color: #ffffff;
                padding: 8px 15px;
                margin: 2px;
            }
            QTabBar::tab:selected {
                background-color: #00ff88;
                color: #000000;
            }
            QTableWidget {
                background-color: #1e1e2e;
                alternate-background-color: #2a2a3a;
                gridline-color: #444444;
            }
            QHeaderView::section {
                background-color: #00ff88;
                color: #000000;
                padding: 5px;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #1e1e2e;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #00ff88;
                border-radius: 6px;
                min-height: 20px;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Create tabs
        self.tabs = QTabWidget()
        
        # Add tabs
        self.tabs.addTab(self.create_dashboard_tab(), "📊 Dashboard")
        self.tabs.addTab(self.create_blockchain_tab(), "⛓️ Blockchain")
        self.tabs.addTab(self.create_transactions_tab(), "💱 Transactions")
        self.tabs.addTab(self.create_analytics_tab(), "📈 Analytics")
        
        main_layout.addWidget(self.tabs)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("background-color: #1e1e2e; color: #00ff88;")
        self.setStatusBar(self.status_bar)
        self.update_status("System initialized")
        
        # Timer for GUI updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_gui)
        self.update_timer.start(1000)
        
    def create_header(self):
        header = QWidget()
        header.setMaximumHeight(80)
        header.setStyleSheet("background-color: #1e1e2e; border-bottom: 2px solid #00ff88;")
        
        layout = QHBoxLayout(header)
        
        # Title
        title = QLabel("⛓️ BLOCKCHAIN 4.0")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ff88;")
        layout.addWidget(title)
        
        subtitle = QLabel("IoT & AI Integration Platform")
        subtitle.setStyleSheet("font-size: 14px; color: #cccccc; margin-left: 10px;")
        layout.addWidget(subtitle)
        
        layout.addStretch()
        
        # Control buttons
        self.start_btn = QPushButton("▶️ Start")
        self.start_btn.clicked.connect(self.toggle_monitoring)
        layout.addWidget(self.start_btn)
        
        self.export_btn = QPushButton("💾 Export")
        self.export_btn.clicked.connect(self.export_blockchain)
        layout.addWidget(self.export_btn)
        
        return header
        
    def create_dashboard_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Metrics
        metrics_layout = QHBoxLayout()
        
        self.metric_cards = {
            'blocks': MetricCard("Total Blocks", "1", "in chain", "#00ff88"),
            'transactions': MetricCard("Transactions", "0", "processed", "#00ffff"),
            'anomalies': MetricCard("Anomalies", "0", "detected", "#ff4444"),
            'sensors': MetricCard("Active Sensors", "5", "online", "#ffaa00"),
        }
        
        for card in self.metric_cards.values():
            metrics_layout.addWidget(card)
            
        layout.addLayout(metrics_layout)
        
        # Charts
        charts_layout = QHBoxLayout()
        
        # Sensor data chart
        sensor_group = QGroupBox("Sensor Data")
        sensor_layout = QVBoxLayout(sensor_group)
        self.sensor_chart = SimpleChart("Temperature Readings (°C)")
        sensor_layout.addWidget(self.sensor_chart)
        charts_layout.addWidget(sensor_group)
        
        # Anomaly chart
        anomaly_group = QGroupBox("Anomaly Detection")
        anomaly_layout = QVBoxLayout(anomaly_group)
        self.anomaly_chart = SimpleChart("Anomaly Score")
        anomaly_layout.addWidget(self.anomaly_chart)
        charts_layout.addWidget(anomaly_group)
        
        layout.addLayout(charts_layout)
        
        # Activity table
        activity_group = QGroupBox("Recent Activity")
        activity_layout = QVBoxLayout(activity_group)
        
        self.activity_table = QTableWidget(0, 5)
        self.activity_table.setHorizontalHeaderLabels(["Time", "Type", "Sensor", "Value", "Status"])
        self.activity_table.horizontalHeader().setStretchLastSection(True)
        self.activity_table.setAlternatingRowColors(True)
        
        activity_layout.addWidget(self.activity_table)
        layout.addWidget(activity_group)
        
        return widget
        
    def create_blockchain_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Blockchain controls
        controls_layout = QHBoxLayout()
        
        mine_btn = QPushButton("⛏️ Mine Block")
        mine_btn.clicked.connect(self.force_mine_block)
        controls_layout.addWidget(mine_btn)
        
        verify_btn = QPushButton("✓ Verify Chain")
        verify_btn.clicked.connect(self.verify_blockchain)
        controls_layout.addWidget(verify_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Blockchain visualization
        viz_group = QGroupBox("Blockchain Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        # Scroll area for blocks
        self.blocks_scroll = QScrollArea()
        self.blocks_scroll.setWidgetResizable(True)
        self.blocks_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        self.blocks_container = QWidget()
        self.blocks_layout = QHBoxLayout(self.blocks_container)
        self.blocks_layout.addStretch()
        
        self.blocks_scroll.setWidget(self.blocks_container)
        viz_layout.addWidget(self.blocks_scroll)
        
        layout.addWidget(viz_group)
        
        # Block details
        details_group = QGroupBox("Block Details")
        details_layout = QVBoxLayout(details_group)
        
        self.block_table = QTableWidget(0, 6)
        self.block_table.setHorizontalHeaderLabels(["Index", "Timestamp", "Hash", "Miner", "Nonce", "Txns"])
        self.block_table.horizontalHeader().setStretchLastSection(True)
        
        details_layout.addWidget(self.block_table)
        layout.addWidget(details_group)
        
        return widget
        
    def create_transactions_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Transaction pool
        pool_group = QGroupBox("Transaction Pool")
        pool_layout = QVBoxLayout(pool_group)
        
        self.tx_pool_table = QTableWidget(0, 5)
        self.tx_pool_table.setHorizontalHeaderLabels(["Timestamp", "Sensor", "Value", "Type", "Signature"])
        self.tx_pool_table.horizontalHeader().setStretchLastSection(True)
        
        pool_layout.addWidget(self.tx_pool_table)
        layout.addWidget(pool_group)
        
        # Confirmed transactions
        confirmed_group = QGroupBox("Confirmed Transactions")
        confirmed_layout = QVBoxLayout(confirmed_group)
        
        self.confirmed_tx_table = QTableWidget(0, 6)
        self.confirmed_tx_table.setHorizontalHeaderLabels(["Block", "Timestamp", "Sensor", "Value", "Type", "Signature"])
        self.confirmed_tx_table.horizontalHeader().setStretchLastSection(True)
        
        confirmed_layout.addWidget(self.confirmed_tx_table)
        layout.addWidget(confirmed_group)
        
        return widget
        
    def create_analytics_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Statistics
        stats_group = QGroupBox("Blockchain Statistics")
        stats_layout = QFormLayout(stats_group)
        
        self.stats_labels = {
            'chain_size': QLabel("0 blocks"),
            'total_transactions': QLabel("0"),
            'avg_block_time': QLabel("0.0 seconds"),
            'anomaly_rate': QLabel("0.0%"),
            'chain_validity': QLabel("✓ Valid")
        }
        
        stats_layout.addRow("Chain Size:", self.stats_labels['chain_size'])
        stats_layout.addRow("Total Transactions:", self.stats_labels['total_transactions'])
        stats_layout.addRow("Avg Block Time:", self.stats_labels['avg_block_time'])
        stats_layout.addRow("Anomaly Rate:", self.stats_labels['anomaly_rate'])
        stats_layout.addRow("Chain Validity:", self.stats_labels['chain_validity'])
        
        layout.addWidget(stats_group)
        
        # Performance metrics
        perf_group = QGroupBox("Performance Metrics")
        perf_layout = QVBoxLayout(perf_group)
        
        self.perf_text = QTextEdit()
        self.perf_text.setReadOnly(True)
        self.perf_text.setMaximumHeight(200)
        
        perf_layout.addWidget(self.perf_text)
        layout.addWidget(perf_group)
        
        layout.addStretch()
        
        return widget
        
    def setup_data_thread(self):
        self.data_thread = DataThread(self.blockchain, self.detector)
        self.data_thread.new_data.connect(self.handle_new_data)
        self.data_thread.new_block.connect(self.handle_new_block)
        
    def toggle_monitoring(self):
        if self.start_btn.text() == "▶️ Start":
            self.data_thread.start()
            self.start_btn.setText("⏸️ Pause")
            self.update_status("Monitoring started")
        else:
            self.data_thread.stop()
            self.start_btn.setText("▶️ Start")
            self.update_status("Monitoring paused")
            
    def handle_new_data(self, data):
        # Update sensor chart
        self.sensor_chart.add_data(data['value'])
        
        # Update anomaly chart
        self.anomaly_chart.add_data(data['anomaly_score'])
        
        # Update activity table
        row = self.activity_table.rowCount()
        if row > 20:  # Limit rows
            self.activity_table.removeRow(0)
            row = 19
            
        self.activity_table.insertRow(row)
        self.activity_table.setItem(row, 0, QTableWidgetItem(datetime.now().strftime('%H:%M:%S')))
        self.activity_table.setItem(row, 1, QTableWidgetItem("Sensor Data"))
        self.activity_table.setItem(row, 2, QTableWidgetItem(data['sensor_id']))
        self.activity_table.setItem(row, 3, QTableWidgetItem(f"{data['value']:.2f}°C"))
        
        status_item = QTableWidgetItem("Anomaly" if data['is_anomaly'] else "Normal")
        if data['is_anomaly']:
            status_item.setForeground(QBrush(QColor(255, 68, 68)))
            # Update anomaly count
            current = int(self.metric_cards['anomalies'].value_label.text())
            self.metric_cards['anomalies'].set_value(current + 1)
        else:
            status_item.setForeground(QBrush(QColor(0, 255, 136)))
        self.activity_table.setItem(row, 4, status_item)
        
        # Update transaction pool
        self.update_transaction_pool()
        
    def handle_new_block(self, block_data):
        # Add block widget
        block_widget = BlockWidget(block_data)
        self.blocks_layout.insertWidget(self.blocks_layout.count() - 1, block_widget)
        
        # Scroll to latest block
        QTimer.singleShot(100, lambda: self.blocks_scroll.horizontalScrollBar().setValue(
            self.blocks_scroll.horizontalScrollBar().maximum()
        ))
        
        # Update block table
        row = self.block_table.rowCount()
        self.block_table.insertRow(row)
        
        self.block_table.setItem(row, 0, QTableWidgetItem(str(block_data['index'])))
        self.block_table.setItem(row, 1, QTableWidgetItem(
            datetime.fromtimestamp(block_data['timestamp']).strftime('%H:%M:%S')
        ))
        self.block_table.setItem(row, 2, QTableWidgetItem(block_data['hash'][:16] + "..."))
        self.block_table.setItem(row, 3, QTableWidgetItem(block_data['miner_id']))
        self.block_table.setItem(row, 4, QTableWidgetItem(str(block_data['nonce'])))
        self.block_table.setItem(row, 5, QTableWidgetItem(str(len(block_data['transactions']))))
        
        # Update metrics
        self.metric_cards['blocks'].set_value(len(self.blockchain.chain))
        total_tx = sum(len(block.transactions) for block in self.blockchain.chain)
        self.metric_cards['transactions'].set_value(total_tx)
        
        # Update confirmed transactions
        for tx in block_data['transactions']:
            row = self.confirmed_tx_table.rowCount()
            self.confirmed_tx_table.insertRow(row)
            
            self.confirmed_tx_table.setItem(row, 0, QTableWidgetItem(str(block_data['index'])))
            self.confirmed_tx_table.setItem(row, 1, QTableWidgetItem(
                datetime.fromtimestamp(tx['timestamp']).strftime('%H:%M:%S')
            ))
            self.confirmed_tx_table.setItem(row, 2, QTableWidgetItem(tx['sensor_id']))
            self.confirmed_tx_table.setItem(row, 3, QTableWidgetItem(f"{tx['value']:.2f}"))
            self.confirmed_tx_table.setItem(row, 4, QTableWidgetItem(tx['type']))
            self.confirmed_tx_table.setItem(row, 5, QTableWidgetItem(tx['signature'][:16] + "..."))
        
    def update_transaction_pool(self):
        # Clear and repopulate transaction pool table
        self.tx_pool_table.setRowCount(0)
        
        for tx in self.blockchain.pending_transactions[-10:]:  # Show last 10
            row = self.tx_pool_table.rowCount()
            self.tx_pool_table.insertRow(row)
            
            tx_dict = tx.to_dict()
            self.tx_pool_table.setItem(row, 0, QTableWidgetItem(
                datetime.fromtimestamp(tx_dict['timestamp']).strftime('%H:%M:%S')
            ))
            self.tx_pool_table.setItem(row, 1, QTableWidgetItem(tx_dict['sensor_id']))
            self.tx_pool_table.setItem(row, 2, QTableWidgetItem(f"{tx_dict['value']:.2f}"))
            self.tx_pool_table.setItem(row, 3, QTableWidgetItem(tx_dict['type']))
            self.tx_pool_table.setItem(row, 4, QTableWidgetItem(tx_dict['signature'][:16] + "..."))
            
    def force_mine_block(self):
        if self.blockchain.pending_transactions:
            block = self.blockchain.mine_pending_transactions("manual_miner")
            if block:
                self.handle_new_block(block.to_dict())
                QMessageBox.information(self, "Success", f"Block #{block.index} mined successfully!")
        else:
            QMessageBox.warning(self, "No Transactions", "No pending transactions to mine")
            
    def verify_blockchain(self):
        is_valid = self.blockchain.validate_chain()
        if is_valid:
            self.stats_labels['chain_validity'].setText("✓ Valid")
            self.stats_labels['chain_validity'].setStyleSheet("color: #00ff88;")
            QMessageBox.information(self, "Blockchain Valid", "The blockchain is valid and intact!")
        else:
            self.stats_labels['chain_validity'].setText("✗ Invalid")
            self.stats_labels['chain_validity'].setStyleSheet("color: #ff4444;")
            QMessageBox.critical(self, "Blockchain Invalid", "The blockchain validation failed!")
            
    def export_blockchain(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Blockchain", "blockchain_export.json", "JSON Files (*.json)"
        )
        if filename:
            try:
                export_data = {
                    'chain': [block.to_dict() for block in self.blockchain.chain],
                    'statistics': {
                        'total_blocks': len(self.blockchain.chain),
                        'total_transactions': sum(len(block.transactions) for block in self.blockchain.chain),
                        'export_time': time.time()
                    }
                }
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                self.update_status(f"Blockchain exported to {filename}")
                QMessageBox.information(self, "Export Complete", "Blockchain exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
                
    def update_gui(self):
        # Update statistics
        if len(self.blockchain.chain) > 0:
            self.stats_labels['chain_size'].setText(f"{len(self.blockchain.chain)} blocks")
            
            total_tx = sum(len(block.transactions) for block in self.blockchain.chain)
            self.stats_labels['total_transactions'].setText(str(total_tx))
            
            # Calculate average block time
            if len(self.blockchain.chain) > 1:
                time_diffs = []
                for i in range(1, len(self.blockchain.chain)):
                    time_diff = self.blockchain.chain[i].timestamp - self.blockchain.chain[i-1].timestamp
                    time_diffs.append(time_diff)
                avg_time = sum(time_diffs) / len(time_diffs)
                self.stats_labels['avg_block_time'].setText(f"{avg_time:.1f} seconds")
                
            # Update performance text
            self.perf_text.clear()
            self.perf_text.append(f"Blockchain Performance Report")
            self.perf_text.append(f"=" * 40)
            self.perf_text.append(f"Chain Length: {len(self.blockchain.chain)} blocks")
            self.perf_text.append(f"Pending Transactions: {len(self.blockchain.pending_transactions)}")
            self.perf_text.append(f"Mining Difficulty: {self.blockchain.difficulty}")
            self.perf_text.append(f"Block Reward: {self.blockchain.mining_reward}")
            
    def update_status(self, message):
        self.status_bar.showMessage(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        
    def closeEvent(self, event):
        if hasattr(self, 'data_thread'):
            self.data_thread.stop()
            self.data_thread.wait()
        event.accept()

# ===================== Main Entry Point =====================
def main():
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        # Set application icon if available
        app.setWindowIcon(QIcon())
        
        # Create and show main window
        window = BlockchainGUI()
        window.show()
        
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Error starting application: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == '__main__':
    main()