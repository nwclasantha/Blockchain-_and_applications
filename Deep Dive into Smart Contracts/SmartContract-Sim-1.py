#!/usr/bin/env python3
"""
Educational Blockchain Simulation Application - Complete Working Version
A comprehensive GUI-based blockchain simulator for educational purposes
Featuring smart contract deployment, transaction management, and mining simulation

Requirements:
pip install customtkinter ecdsa
"""

import customtkinter as ctk
import hashlib
import json
import time
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
import tkinter as tk
from tkinter import messagebox
import ecdsa
import random
import string

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class Transaction:
    """Represents a blockchain transaction"""
    def __init__(self, sender: str, receiver: str, amount: float, fee: float = 0.001):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.fee = fee
        self.timestamp = datetime.now().isoformat()
        self.tx_id = self.generate_tx_id()
        self.signature = None
        
    def generate_tx_id(self) -> str:
        """Generate unique transaction ID"""
        data = f"{self.sender}{self.receiver}{self.amount}{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tx_id': self.tx_id,
            'sender': self.sender,
            'receiver': self.receiver,
            'amount': self.amount,
            'fee': self.fee,
            'timestamp': self.timestamp,
            'signature': self.signature
        }

class SmartContract:
    """Represents a smart contract"""
    def __init__(self, name: str, code: str, creator: str):
        self.name = name
        self.code = code
        self.creator = creator
        self.deployed_at = datetime.now().isoformat()
        self.address = self.generate_address()
        self.state = {}
        
    def generate_address(self) -> str:
        """Generate contract address"""
        data = f"{self.name}{self.creator}{self.deployed_at}"
        return "0x" + hashlib.sha256(data.encode()).hexdigest()[:20]
    
    def execute(self, function_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate contract execution"""
        result = {
            'success': True,
            'result': f"Executed {function_name} with params: {params}",
            'gas_used': random.randint(21000, 100000)
        }
        return result

class Block:
    """Represents a blockchain block"""
    def __init__(self, index: int, transactions: List[Transaction], previous_hash: str, miner: str = "System"):
        self.index = index
        self.transactions = transactions
        self.timestamp = datetime.now().isoformat()
        self.previous_hash = previous_hash
        self.miner = miner
        self.nonce = 0
        self.difficulty = 2  # Educational difficulty
        self.merkle_root = self.calculate_merkle_root()
        self.hash = self.calculate_hash()
        
    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_data = {
            'index': self.index,
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'nonce': self.nonce,
            'miner': self.miner
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def calculate_merkle_root(self) -> str:
        """Calculate Merkle root of transactions"""
        if not self.transactions:
            return hashlib.sha256("".encode()).hexdigest()
        
        tx_hashes = [hashlib.sha256(json.dumps(tx.to_dict()).encode()).hexdigest() 
                    for tx in self.transactions]
        
        while len(tx_hashes) > 1:
            if len(tx_hashes) % 2 == 1:
                tx_hashes.append(tx_hashes[-1])
            
            new_hashes = []
            for i in range(0, len(tx_hashes), 2):
                combined = tx_hashes[i] + tx_hashes[i + 1]
                new_hashes.append(hashlib.sha256(combined.encode()).hexdigest())
            tx_hashes = new_hashes
        
        return tx_hashes[0]
    
    def mine_block(self, difficulty: int = 2) -> None:
        """Mine the block with proof of work"""
        target = "0" * difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.calculate_hash()
            if self.nonce > 100000:  # Educational safeguard
                break

class User:
    """Represents a blockchain user"""
    def __init__(self, name: str):
        self.name = name
        self.private_key = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
        self.public_key = self.private_key.get_verifying_key()
        self.address = self.generate_address()
        
    def generate_address(self) -> str:
        """Generate user address from public key"""
        pub_key_hex = self.public_key.to_string().hex()
        return "0x" + hashlib.sha256(pub_key_hex.encode()).hexdigest()[:20]
    
    def sign_transaction(self, transaction: Transaction) -> str:
        """Sign a transaction"""
        tx_data = json.dumps(transaction.to_dict(), sort_keys=True)
        signature = self.private_key.sign(tx_data.encode())
        return signature.hex()

class Blockchain:
    """Main blockchain class"""
    def __init__(self):
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.users: Dict[str, User] = {}
        self.smart_contracts: Dict[str, SmartContract] = {}
        self.mining_reward = 10.0
        self.initial_balances: Dict[str, float] = {}  # Track initial balances
        self.create_genesis_block()
        
    def create_genesis_block(self) -> None:
        """Create the first block in the blockchain"""
        genesis_block = Block(0, [], "0", "Genesis")
        genesis_block.hash = genesis_block.calculate_hash()
        self.chain.append(genesis_block)
    
    def get_latest_block(self) -> Block:
        """Get the latest block in the chain"""
        return self.chain[-1]
    
    def add_transaction(self, transaction: Transaction) -> bool:
        """Add a transaction to pending transactions"""
        if self.validate_transaction(transaction):
            self.pending_transactions.append(transaction)
            return True
        return False
    
    def validate_transaction(self, transaction: Transaction) -> bool:
        """Validate a transaction"""
        if transaction.sender == "System":  # Mining reward
            return True
        
        sender_balance = self.get_balance(transaction.sender)
        return sender_balance >= transaction.amount + transaction.fee
    
    def mine_pending_transactions(self, miner_address: str) -> Block:
        """Mine pending transactions into a new block"""
        # Add mining reward
        reward_tx = Transaction("System", miner_address, self.mining_reward)
        self.pending_transactions.append(reward_tx)
        
        # Create new block
        new_block = Block(
            len(self.chain),
            self.pending_transactions.copy(),
            self.get_latest_block().hash,
            miner_address
        )
        
        # Mine the block
        new_block.mine_block()
        
        # Add to chain and clear pending transactions
        self.chain.append(new_block)
        self.pending_transactions = []
        
        return new_block
    
    def get_balance(self, address: str) -> float:
        """Calculate balance for an address"""
        balance = self.initial_balances.get(address, 0.0)  # Start with initial balance
        
        for block in self.chain:
            for tx in block.transactions:
                if tx.sender == address:
                    balance -= (tx.amount + tx.fee)
                if tx.receiver == address:
                    balance += tx.amount
        
        return balance
    
    def deploy_contract(self, contract: SmartContract) -> bool:
        """Deploy a smart contract"""
        self.smart_contracts[contract.address] = contract
        return True
    
    def is_chain_valid(self) -> bool:
        """Validate the entire blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            if current_block.hash != current_block.calculate_hash():
                return False
            
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True

class BlockchainSimulatorApp:
    """Main GUI application"""
    def __init__(self):
        self.blockchain = Blockchain()
        self.current_user = None
        self.mining_active = False
        
        # Initialize GUI
        self.root = ctk.CTk()
        self.root.title("Educational Blockchain Simulator")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.create_sample_users()
        self.setup_gui()
        
    def create_sample_users(self):
        """Create sample users for demonstration"""
        users = ["Alice", "Bob", "Charlie", "Diana"]
        for name in users:
            user = User(name)
            self.blockchain.users[user.address] = user
            # Set initial balance of 100 coins for each user
            self.blockchain.initial_balances[user.address] = 100.0
    
    def setup_gui(self):
        """Setup the main GUI"""
        # Create main container
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create header
        self.create_header()
        
        # Create tabview
        self.tabview = ctk.CTkTabview(self.main_container)
        self.tabview.pack(fill="both", expand=True, pady=(10, 0))
        
        # Add tabs
        self.create_dashboard_tab()
        self.create_transactions_tab()
        self.create_smart_contracts_tab()
        self.create_mining_tab()
        self.create_blockchain_explorer_tab()
        self.create_education_tab()
        
        # Update displays
        self.update_all_displays()
        
        # Set initial user after all components are created
        user_names = [user.name for user in self.blockchain.users.values()]
        if user_names:
            self.on_user_change(user_names[0])
    
    def create_header(self):
        """Create application header"""
        header_frame = ctk.CTkFrame(self.main_container)
        header_frame.pack(fill="x", pady=(0, 10))
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame, 
            text="🔗 Educational Blockchain Simulator", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(side="left", padx=20, pady=15)
        
        # User selection
        user_frame = ctk.CTkFrame(header_frame)
        user_frame.pack(side="right", padx=20, pady=10)
        
        ctk.CTkLabel(user_frame, text="Current User:").pack(side="left", padx=(10, 5))
        
        self.user_var = ctk.StringVar()
        user_names = [user.name for user in self.blockchain.users.values()]
        self.user_dropdown = ctk.CTkComboBox(
            user_frame,
            values=user_names,
            variable=self.user_var,
            command=self.on_user_change
        )
        self.user_dropdown.pack(side="left", padx=(0, 10))
        
        # Refresh button
        refresh_btn = ctk.CTkButton(
            user_frame,
            text="🔄",
            width=30,
            command=self.refresh_application
        )
        refresh_btn.pack(side="right", padx=(5, 10))
        
        if user_names:
            self.user_dropdown.set(user_names[0])
    
    def create_dashboard_tab(self):
        """Create dashboard tab"""
        self.dashboard_tab = self.tabview.add("📊 Dashboard")
        
        # Top stats frame
        stats_frame = ctk.CTkFrame(self.dashboard_tab)
        stats_frame.pack(fill="x", padx=10, pady=10)
        
        # Stats labels
        self.stats_labels = {}
        stats = ["Blocks", "Transactions", "Users", "Smart Contracts"]
        for i, stat in enumerate(stats):
            stat_frame = ctk.CTkFrame(stats_frame)
            stat_frame.grid(row=0, column=i, padx=10, pady=10, sticky="ew")
            stats_frame.grid_columnconfigure(i, weight=1)
            
            value_label = ctk.CTkLabel(stat_frame, text="0", font=ctk.CTkFont(size=20, weight="bold"))
            value_label.pack(pady=(10, 0))
            
            name_label = ctk.CTkLabel(stat_frame, text=stat)
            name_label.pack(pady=(0, 10))
            
            self.stats_labels[stat] = value_label
        
        # Content frame
        content_frame = ctk.CTkFrame(self.dashboard_tab)
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel - User info
        left_panel = ctk.CTkFrame(content_frame)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        ctk.CTkLabel(left_panel, text="👤 User Information", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.user_info_frame = ctk.CTkFrame(left_panel)
        self.user_info_frame.pack(fill="x", padx=10, pady=10)
        
        # Right panel - Recent transactions
        right_panel = ctk.CTkFrame(content_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        ctk.CTkLabel(right_panel, text="📝 Recent Transactions", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.recent_tx_text = ctk.CTkTextbox(right_panel, height=300)
        self.recent_tx_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_transactions_tab(self):
        """Create transactions tab"""
        self.transactions_tab = self.tabview.add("💸 Transactions")
        
        # Transaction form
        form_frame = ctk.CTkFrame(self.transactions_tab)
        form_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(form_frame, text="💰 Create New Transaction", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        # Form fields
        fields_frame = ctk.CTkFrame(form_frame)
        fields_frame.pack(fill="x", padx=10, pady=10)
        
        # Receiver
        ctk.CTkLabel(fields_frame, text="To:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.tx_receiver_var = ctk.StringVar()
        receiver_combo = ctk.CTkComboBox(
            fields_frame,
            values=[user.name for user in self.blockchain.users.values()],
            variable=self.tx_receiver_var
        )
        receiver_combo.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        # Amount
        ctk.CTkLabel(fields_frame, text="Amount:").grid(row=0, column=2, padx=10, pady=5, sticky="w")
        self.tx_amount_entry = ctk.CTkEntry(fields_frame, placeholder_text="0.00")
        self.tx_amount_entry.grid(row=0, column=3, padx=10, pady=5, sticky="ew")
        
        # Fee
        ctk.CTkLabel(fields_frame, text="Fee:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.tx_fee_entry = ctk.CTkEntry(fields_frame, placeholder_text="0.001")
        self.tx_fee_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        # Send button
        send_btn = ctk.CTkButton(
            fields_frame,
            text="📤 Send Transaction",
            command=self.send_transaction
        )
        send_btn.grid(row=1, column=2, columnspan=2, padx=10, pady=5, sticky="ew")
        
        fields_frame.grid_columnconfigure(1, weight=1)
        fields_frame.grid_columnconfigure(3, weight=1)
        
        # Transaction history
        history_frame = ctk.CTkFrame(self.transactions_tab)
        history_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(history_frame, text="📋 Transaction History", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.tx_history_text = ctk.CTkTextbox(history_frame, height=400)
        self.tx_history_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_smart_contracts_tab(self):
        """Create smart contracts tab"""
        self.contracts_tab = self.tabview.add("📜 Smart Contracts")
        
        # Contract deployment frame
        deploy_frame = ctk.CTkFrame(self.contracts_tab)
        deploy_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(deploy_frame, text="🚀 Deploy Smart Contract", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        # Contract form
        contract_form = ctk.CTkFrame(deploy_frame)
        contract_form.pack(fill="x", padx=10, pady=10)
        
        # Contract name
        ctk.CTkLabel(contract_form, text="Contract Name:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.contract_name_entry = ctk.CTkEntry(contract_form, placeholder_text="MyContract")
        self.contract_name_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        # Contract type
        ctk.CTkLabel(contract_form, text="Contract Type:").grid(row=0, column=2, padx=10, pady=5, sticky="w")
        self.contract_type_var = ctk.StringVar(value="Token")
        contract_type_combo = ctk.CTkComboBox(
            contract_form,
            values=["Token", "Voting", "Escrow", "NFT", "Custom"],
            variable=self.contract_type_var,
            command=self.on_contract_type_change
        )
        contract_type_combo.grid(row=0, column=3, padx=10, pady=5, sticky="ew")
        
        contract_form.grid_columnconfigure(1, weight=1)
        contract_form.grid_columnconfigure(3, weight=1)
        
        # Contract code area
        code_frame = ctk.CTkFrame(contract_form)
        code_frame.grid(row=1, column=0, columnspan=4, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(code_frame, text="Contract Code:").pack(anchor="w", padx=10, pady=(10, 0))
        self.contract_code_text = ctk.CTkTextbox(code_frame, height=200)
        self.contract_code_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Deploy button
        deploy_btn = ctk.CTkButton(
            contract_form,
            text="🚀 Deploy Contract",
            command=self.deploy_contract
        )
        deploy_btn.grid(row=2, column=0, columnspan=4, padx=10, pady=10)
        
        # Deployed contracts list
        deployed_frame = ctk.CTkFrame(self.contracts_tab)
        deployed_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(deployed_frame, text="📋 Deployed Contracts", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.contracts_text = ctk.CTkTextbox(deployed_frame, height=300)
        self.contracts_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Set default contract code
        self.on_contract_type_change("Token")
    
    def create_mining_tab(self):
        """Create mining tab"""
        self.mining_tab = self.tabview.add("⛏️ Mining")
        
        # Mining controls
        controls_frame = ctk.CTkFrame(self.mining_tab)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(controls_frame, text="⛏️ Mining Controls", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        # Controls
        controls_inner = ctk.CTkFrame(controls_frame)
        controls_inner.pack(fill="x", padx=10, pady=10)
        
        # Difficulty slider
        ctk.CTkLabel(controls_inner, text="Difficulty:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.difficulty_var = ctk.IntVar(value=2)
        difficulty_slider = ctk.CTkSlider(controls_inner, from_=1, to=4, variable=self.difficulty_var, number_of_steps=3)
        difficulty_slider.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.difficulty_label = ctk.CTkLabel(controls_inner, text="2")
        self.difficulty_label.grid(row=0, column=2, padx=10, pady=5)
        
        # Mining buttons
        self.start_mining_btn = ctk.CTkButton(
            controls_inner,
            text="⛏️ Start Mining",
            command=self.start_mining
        )
        self.start_mining_btn.grid(row=1, column=0, padx=10, pady=10)
        
        self.stop_mining_btn = ctk.CTkButton(
            controls_inner,
            text="⏹️ Stop Mining",
            command=self.stop_mining,
            state="disabled"
        )
        self.stop_mining_btn.grid(row=1, column=1, padx=10, pady=10)
        
        # Auto mine button
        self.auto_mine_btn = ctk.CTkButton(
            controls_inner,
            text="🔄 Auto Mine",
            command=self.auto_mine
        )
        self.auto_mine_btn.grid(row=1, column=2, padx=10, pady=10)
        
        controls_inner.grid_columnconfigure(1, weight=1)
        
        # Update difficulty label
        difficulty_slider.configure(command=lambda value: self.difficulty_label.configure(text=str(int(value))))
        
        # Mining stats
        stats_frame = ctk.CTkFrame(self.mining_tab)
        stats_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(stats_frame, text="📊 Mining Statistics", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.mining_stats_frame = ctk.CTkFrame(stats_frame)
        self.mining_stats_frame.pack(fill="x", padx=10, pady=10)
        
        # Mining log
        log_frame = ctk.CTkFrame(self.mining_tab)
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(log_frame, text="📝 Mining Log", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.mining_log_text = ctk.CTkTextbox(log_frame, height=300)
        self.mining_log_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_blockchain_explorer_tab(self):
        """Create blockchain explorer tab"""
        self.explorer_tab = self.tabview.add("🔍 Explorer")
        
        # Search frame
        search_frame = ctk.CTkFrame(self.explorer_tab)
        search_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(search_frame, text="🔍 Blockchain Explorer", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        search_inner = ctk.CTkFrame(search_frame)
        search_inner.pack(fill="x", padx=10, pady=10)
        
        self.search_entry = ctk.CTkEntry(search_inner, placeholder_text="Enter block index, transaction ID, or address...")
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        search_btn = ctk.CTkButton(search_inner, text="🔍 Search", command=self.search_blockchain)
        search_btn.pack(side="right")
        
        # Results frame
        results_frame = ctk.CTkFrame(self.explorer_tab)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(results_frame, text="📋 Blockchain Data", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.explorer_text = ctk.CTkTextbox(results_frame, height=500)
        self.explorer_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Navigation buttons
        nav_frame = ctk.CTkFrame(results_frame)
        nav_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(nav_frame, text="📊 View All Blocks", command=self.view_all_blocks).pack(side="left", padx=5)
        ctk.CTkButton(nav_frame, text="💸 View All Transactions", command=self.view_all_transactions).pack(side="left", padx=5)
        ctk.CTkButton(nav_frame, text="👥 View All Users", command=self.view_all_users).pack(side="left", padx=5)
        ctk.CTkButton(nav_frame, text="✅ Validate Chain", command=self.validate_blockchain).pack(side="left", padx=5)
    
    def create_education_tab(self):
        """Create education tab"""
        self.education_tab = self.tabview.add("🎓 Education")
        
        # Topics frame
        topics_frame = ctk.CTkFrame(self.education_tab)
        topics_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(topics_frame, text="🎓 Blockchain Education", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        # Topic selection
        topics_inner = ctk.CTkFrame(topics_frame)
        topics_inner.pack(fill="x", padx=10, pady=10)
        
        self.education_topic_var = ctk.StringVar(value="Hashing")
        topics = ["Hashing", "Digital Signatures", "Blocks", "Consensus", "Smart Contracts", "Mining", "Transactions"]
        
        topic_combo = ctk.CTkComboBox(
            topics_inner,
            values=topics,
            variable=self.education_topic_var,
            command=self.load_education_content
        )
        topic_combo.pack(side="left", padx=(0, 10))
        
        demo_btn = ctk.CTkButton(topics_inner, text="🚀 Run Demo", command=self.run_education_demo)
        demo_btn.pack(side="left")
        
        # Content frame
        content_frame = ctk.CTkFrame(self.education_tab)
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.education_content = ctk.CTkTextbox(content_frame, height=500)
        self.education_content.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Load initial content
        self.load_education_content("Hashing")
    
    def on_user_change(self, username):
        """Handle user change"""
        for user in self.blockchain.users.values():
            if user.name == username:
                self.current_user = user
                break
        self.update_user_info()
        self.update_deployed_contracts()  # Update contracts display when user changes
    
    def refresh_application(self):
        """Refresh all application data and displays"""
        self.update_all_displays()
        if self.current_user:
            # Force update user info to show correct balance
            self.update_user_info()
        messagebox.showinfo("Refresh", "Application refreshed! Balances updated.")
    
    def update_user_info(self):
        """Update user information display"""
        if not self.current_user or not hasattr(self, 'user_info_frame'):
            return
        
        # Clear existing info
        for widget in self.user_info_frame.winfo_children():
            widget.destroy()
        
        # User details
        info_text = f"""👤 Name: {self.current_user.name}
🏠 Address: {self.current_user.address}
💰 Balance: {self.blockchain.get_balance(self.current_user.address):.3f} coins
🔐 Public Key: {self.current_user.public_key.to_string().hex()[:20]}..."""
        
        info_label = ctk.CTkLabel(self.user_info_frame, text=info_text, justify="left")
        info_label.pack(padx=10, pady=10)
    
    def update_all_displays(self):
        """Update all display components"""
        self.update_stats()
        self.update_user_info()
        self.update_recent_transactions()
        self.update_mining_stats()
        self.update_deployed_contracts()
        # Only update explorer if it exists
        if hasattr(self, 'explorer_text'):
            self.view_all_blocks()
    
    def update_stats(self):
        """Update dashboard statistics"""
        if not hasattr(self, 'stats_labels'):
            return
            
        stats = {
            "Blocks": len(self.blockchain.chain),
            "Transactions": sum(len(block.transactions) for block in self.blockchain.chain),
            "Users": len(self.blockchain.users),
            "Smart Contracts": len(self.blockchain.smart_contracts)
        }
        
        for stat_name, value in stats.items():
            if stat_name in self.stats_labels:
                self.stats_labels[stat_name].configure(text=str(value))
    
    def update_recent_transactions(self):
        """Update recent transactions display"""
        if not hasattr(self, 'recent_tx_text'):
            return
            
        self.recent_tx_text.delete("1.0", "end")
        
        recent_transactions = []
        for block in reversed(self.blockchain.chain[-5:]):  # Last 5 blocks
            for tx in reversed(block.transactions):
                recent_transactions.append((block.index, tx))
                if len(recent_transactions) >= 10:
                    break
            if len(recent_transactions) >= 10:
                break
        
        for block_idx, tx in recent_transactions:
            tx_text = f"Block {block_idx}: {tx.sender[:10]}... → {tx.receiver[:10]}... | {tx.amount:.3f} coins\n"
            self.recent_tx_text.insert("end", tx_text)
    
    def send_transaction(self):
        """Send a new transaction"""
        if not self.current_user:
            messagebox.showerror("Error", "Please select a user first")
            return
        
        try:
            receiver_name = self.tx_receiver_var.get()
            amount = float(self.tx_amount_entry.get() or "0")
            fee = float(self.tx_fee_entry.get() or "0.001")
            
            if not receiver_name or amount <= 0:
                messagebox.showerror("Error", "Please fill in all fields correctly")
                return
            
            # Find receiver
            receiver_user = None
            for user in self.blockchain.users.values():
                if user.name == receiver_name:
                    receiver_user = user
                    break
            
            if not receiver_user:
                messagebox.showerror("Error", "Receiver not found")
                return
            
            # Create and add transaction
            transaction = Transaction(
                self.current_user.address,
                receiver_user.address,
                amount,
                fee
            )
            
            # Sign transaction
            transaction.signature = self.current_user.sign_transaction(transaction)
            
            if self.blockchain.add_transaction(transaction):
                messagebox.showinfo("Success", f"Transaction {transaction.tx_id} added to pending transactions")
                self.tx_amount_entry.delete(0, "end")
                self.update_all_displays()
            else:
                messagebox.showerror("Error", "Insufficient balance")
                
        except ValueError:
            messagebox.showerror("Error", "Invalid amount or fee")
    
    def on_contract_type_change(self, contract_type):
        """Handle contract type change"""
        contract_templates = {
            "Token": '''// Simple Token Contract
contract Token {
    mapping(address => uint256) public balances;
    uint256 public totalSupply;
    string public name;
    
    constructor(string memory _name, uint256 _totalSupply) {
        name = _name;
        totalSupply = _totalSupply;
        balances[msg.sender] = _totalSupply;
    }
    
    function transfer(address to, uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }
}''',
            "Voting": '''// Simple Voting Contract
contract Voting {
    mapping(address => bool) public hasVoted;
    mapping(string => uint256) public votes;
    string[] public candidates;
    
    constructor(string[] memory _candidates) {
        candidates = _candidates;
    }
    
    function vote(string memory candidate) public {
        require(!hasVoted[msg.sender], "Already voted");
        hasVoted[msg.sender] = true;
        votes[candidate]++;
    }
}''',
            "Escrow": '''// Simple Escrow Contract
contract Escrow {
    address public buyer;
    address public seller;
    uint256 public amount;
    bool public released;
    
    constructor(address _seller) payable {
        buyer = msg.sender;
        seller = _seller;
        amount = msg.value;
    }
    
    function release() public {
        require(msg.sender == buyer, "Only buyer can release");
        require(!released, "Already released");
        released = true;
        payable(seller).transfer(amount);
    }
}''',
            "NFT": '''// Simple NFT Contract
contract NFT {
    mapping(uint256 => address) public owners;
    mapping(address => uint256) public balances;
    uint256 public nextTokenId;
    
    function mint(address to) public returns (uint256) {
        uint256 tokenId = nextTokenId++;
        owners[tokenId] = to;
        balances[to]++;
        return tokenId;
    }
    
    function transfer(address to, uint256 tokenId) public {
        require(owners[tokenId] == msg.sender, "Not owner");
        owners[tokenId] = to;
        balances[msg.sender]--;
        balances[to]++;
    }
}''',
            "Custom": '''// Custom Contract
contract MyContract {
    // Add your custom contract code here
    
    constructor() {
        // Constructor logic
    }
    
    function myFunction() public {
        // Function logic
    }
}'''
        }
        
        if hasattr(self, 'contract_code_text'):
            self.contract_code_text.delete("1.0", "end")
            self.contract_code_text.insert("1.0", contract_templates.get(contract_type, ""))
    
    def deploy_contract(self):
        """Deploy a smart contract"""
        if not self.current_user:
            messagebox.showerror("Error", "Please select a user first")
            return
        
        name = self.contract_name_entry.get().strip()
        code = self.contract_code_text.get("1.0", "end-1c").strip()
        
        if not name:
            messagebox.showerror("Error", "Please provide a contract name")
            return
        
        if not code:
            messagebox.showerror("Error", "Contract code cannot be empty")
            return
        
        try:
            # Create smart contract with proper error handling
            contract = SmartContract(name, code, self.current_user.address)
            
            # Verify contract was created properly
            if not hasattr(contract, 'deployed_at'):
                raise AttributeError("Contract deployment failed - missing deployed_at attribute")
            
            if self.blockchain.deploy_contract(contract):
                messagebox.showinfo("Success", f"Contract '{name}' deployed successfully!\nAddress: {contract.address}")
                self.contract_name_entry.delete(0, "end")
                # Update displays safely
                try:
                    self.update_deployed_contracts()
                    self.update_all_displays()
                except Exception as display_error:
                    print(f"Display update error: {display_error}")
            else:
                messagebox.showerror("Error", "Failed to deploy contract")
        
        except Exception as e:
            print(f"Contract deployment error details: {e}")
            messagebox.showerror("Error", f"Contract deployment failed: {str(e)}")
            # Try to refresh the display anyway
            try:
                self.update_deployed_contracts()
            except:
                pass
    
    def update_deployed_contracts(self):
        """Update deployed contracts display"""
        if not hasattr(self, 'contracts_text'):
            return
            
        self.contracts_text.delete("1.0", "end")
        
        if not self.blockchain.smart_contracts:
            self.contracts_text.insert("end", "No contracts deployed yet.\n\nDeploy your first contract above!")
            return
        
        try:
            for contract in self.blockchain.smart_contracts.values():
                # Safely access contract attributes with defaults
                name = getattr(contract, 'name', 'Unknown')
                address = getattr(contract, 'address', 'Unknown')
                creator = getattr(contract, 'creator', 'Unknown')
                deployed_at = getattr(contract, 'deployed_at', 'Unknown')
                
                contract_info = f"""📜 Contract: {name}
🏠 Address: {address}
👤 Creator: {creator}
⏰ Deployed: {deployed_at}
{'='*60}

"""
                self.contracts_text.insert("end", contract_info)
        except Exception as e:
            error_msg = f"Error displaying contracts: {str(e)}\n\nContracts deployed: {len(self.blockchain.smart_contracts)}"
            self.contracts_text.insert("end", error_msg)
    
    def start_mining(self):
        """Start mining process"""
        if not self.current_user:
            messagebox.showerror("Error", "Please select a user first")
            return
        
        if len(self.blockchain.pending_transactions) == 0:
            messagebox.showinfo("Info", "No pending transactions to mine")
            return
        
        self.mining_active = True
        self.start_mining_btn.configure(state="disabled")
        self.stop_mining_btn.configure(state="normal")
        
        # Start mining in separate thread
        mining_thread = threading.Thread(target=self.mine_block_thread)
        mining_thread.daemon = True
        mining_thread.start()
    
    def mine_block_thread(self):
        """Mining thread function"""
        try:
            start_time = time.time()
            
            # Mine block with current difficulty
            difficulty = self.difficulty_var.get()
            
            self.log_mining(f"🔨 Starting mining with difficulty {difficulty}...")
            self.log_mining(f"📋 Mining {len(self.blockchain.pending_transactions)} pending transactions")
            
            # Create temporary block for mining
            temp_block = Block(
                len(self.blockchain.chain),
                self.blockchain.pending_transactions.copy() + [Transaction("System", self.current_user.address, self.blockchain.mining_reward)],
                self.blockchain.get_latest_block().hash,
                self.current_user.address
            )
            
            # Mine the block
            target = "0" * difficulty
            attempts = 0
            
            while self.mining_active and not temp_block.hash.startswith(target):
                temp_block.nonce += 1
                temp_block.hash = temp_block.calculate_hash()
                attempts += 1
                
                if attempts % 1000 == 0:
                    self.log_mining(f"⛏️ Attempts: {attempts:,}")
                
                if attempts > 100000:  # Safety limit
                    break
            
            if self.mining_active:
                # Add the mined block to blockchain
                mined_block = self.blockchain.mine_pending_transactions(self.current_user.address)
                
                end_time = time.time()
                mining_time = end_time - start_time
                
                self.log_mining(f"✅ Block {mined_block.index} mined successfully!")
                self.log_mining(f"⏱️ Mining time: {mining_time:.2f} seconds")
                self.log_mining(f"🎯 Hash: {mined_block.hash}")
                self.log_mining(f"💰 Reward: {self.blockchain.mining_reward} coins")
                self.log_mining("="*50)
                
                # Update displays
                self.root.after(0, self.update_all_displays)
                
        except Exception as e:
            self.log_mining(f"❌ Mining error: {str(e)}")
        finally:
            self.mining_active = False
            self.root.after(0, lambda: [
                self.start_mining_btn.configure(state="normal"),
                self.stop_mining_btn.configure(state="disabled")
            ])
    
    def stop_mining(self):
        """Stop mining process"""
        self.mining_active = False
        self.log_mining("⏹️ Mining stopped by user")
    
    def auto_mine(self):
        """Auto mine when transactions are available"""
        if len(self.blockchain.pending_transactions) > 0:
            self.start_mining()
        else:
            # Create some sample transactions for demo
            users = list(self.blockchain.users.values())
            if len(users) >= 2:
                sender = random.choice(users)
                receiver = random.choice([u for u in users if u != sender])
                amount = random.uniform(0.1, 5.0)
                
                tx = Transaction(sender.address, receiver.address, amount)
                tx.signature = sender.sign_transaction(tx)
                
                if self.blockchain.add_transaction(tx):
                    self.log_mining(f"🔄 Auto-generated transaction: {amount:.3f} coins")
                    self.start_mining()
    
    def log_mining(self, message):
        """Log mining messages"""
        if not hasattr(self, 'mining_log_text'):
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        def update_log():
            if hasattr(self, 'mining_log_text'):
                self.mining_log_text.insert("end", log_message)
                self.mining_log_text.see("end")
        
        self.root.after(0, update_log)
    
    def update_mining_stats(self):
        """Update mining statistics"""
        if not hasattr(self, 'mining_stats_frame'):
            return
            
        # Clear existing stats
        for widget in self.mining_stats_frame.winfo_children():
            widget.destroy()
        
        if not self.current_user:
            return
        
        # Calculate stats
        blocks_mined = sum(1 for block in self.blockchain.chain if block.miner == self.current_user.address)
        total_rewards = blocks_mined * self.blockchain.mining_reward
        hash_rate = "Inactive" if not self.mining_active else "Active"
        
        stats_text = f"""⛏️ Blocks Mined: {blocks_mined}
💰 Total Rewards: {total_rewards:.3f} coins
📊 Hash Rate: {hash_rate}
🎯 Current Difficulty: {self.difficulty_var.get() if hasattr(self, 'difficulty_var') else 2}
⏸️ Pending Transactions: {len(self.blockchain.pending_transactions)}"""
        
        stats_label = ctk.CTkLabel(self.mining_stats_frame, text=stats_text, justify="left")
        stats_label.pack(padx=10, pady=10)
    
    def search_blockchain(self):
        """Search blockchain for specific data"""
        query = self.search_entry.get().strip()
        if not query:
            return
        
        self.explorer_text.delete("1.0", "end")
        results_found = False
        
        # Search for block by index
        try:
            block_index = int(query)
            if 0 <= block_index < len(self.blockchain.chain):
                block = self.blockchain.chain[block_index]
                self.display_block_details(block)
                results_found = True
        except ValueError:
            pass
        
        # Search for transaction by ID
        for block in self.blockchain.chain:
            for tx in block.transactions:
                if query in tx.tx_id:
                    self.display_transaction_details(tx, block.index)
                    results_found = True
        
        # Search for address
        if query in [user.address for user in self.blockchain.users.values()]:
            self.display_address_details(query)
            results_found = True
        
        if not results_found:
            self.explorer_text.insert("end", f"❌ No results found for: {query}\n")
    
    def view_all_blocks(self):
        """Display all blocks"""
        if not hasattr(self, 'explorer_text'):
            return
            
        self.explorer_text.delete("1.0", "end")
        
        for block in self.blockchain.chain:
            block_info = f"""📦 Block {block.index}
🕒 Timestamp: {block.timestamp}
👤 Miner: {block.miner}
🔗 Previous Hash: {block.previous_hash}
🎯 Hash: {block.hash}
🔢 Nonce: {block.nonce}
🌳 Merkle Root: {block.merkle_root}
📝 Transactions: {len(block.transactions)}
{'='*60}

"""
            self.explorer_text.insert("end", block_info)
    
    def view_all_transactions(self):
        """Display all transactions"""
        self.explorer_text.delete("1.0", "end")
        
        for block in self.blockchain.chain:
            for tx in block.transactions:
                tx_info = f"""💸 Transaction {tx.tx_id}
📤 From: {tx.sender}
📥 To: {tx.receiver}
💰 Amount: {tx.amount:.6f} coins
💳 Fee: {tx.fee:.6f} coins
🕒 Timestamp: {tx.timestamp}
✅ Block: {block.index}
{'='*50}

"""
                self.explorer_text.insert("end", tx_info)
    
    def view_all_users(self):
        """Display all users"""
        self.explorer_text.delete("1.0", "end")
        
        for user in self.blockchain.users.values():
            balance = self.blockchain.get_balance(user.address)
            user_info = f"""👤 User: {user.name}
🏠 Address: {user.address}
💰 Balance: {balance:.6f} coins
🔐 Public Key: {user.public_key.to_string().hex()}
{'='*50}

"""
            self.explorer_text.insert("end", user_info)
    
    def validate_blockchain(self):
        """Validate the entire blockchain"""
        self.explorer_text.delete("1.0", "end")
        
        is_valid = self.blockchain.is_chain_valid()
        
        validation_result = f"""🔍 BLOCKCHAIN VALIDATION REPORT
{'='*50}

✅ Blockchain is {'VALID' if is_valid else 'INVALID'}

📊 Chain Statistics:
• Total Blocks: {len(self.blockchain.chain)}
• Total Transactions: {sum(len(block.transactions) for block in self.blockchain.chain)}
• Pending Transactions: {len(self.blockchain.pending_transactions)}

🔗 Block Validation:
"""
        
        self.explorer_text.insert("end", validation_result)
        
        for i, block in enumerate(self.blockchain.chain):
            calculated_hash = block.calculate_hash()
            hash_valid = calculated_hash == block.hash
            
            if i > 0:
                prev_hash_valid = block.previous_hash == self.blockchain.chain[i-1].hash
            else:
                prev_hash_valid = True
            
            block_status = "✅" if hash_valid and prev_hash_valid else "❌"
            status_text = f"{block_status} Block {i}: Hash {'Valid' if hash_valid else 'Invalid'}, Chain {'Valid' if prev_hash_valid else 'Invalid'}\n"
            self.explorer_text.insert("end", status_text)
    
    def display_block_details(self, block):
        """Display detailed block information"""
        block_details = f"""📦 BLOCK DETAILS
{'='*50}

Block Index: {block.index}
Timestamp: {block.timestamp}
Miner: {block.miner}
Previous Hash: {block.previous_hash}
Current Hash: {block.hash}
Nonce: {block.nonce}
Merkle Root: {block.merkle_root}
Difficulty: {block.difficulty}

📝 TRANSACTIONS ({len(block.transactions)}):
{'='*50}
"""
        
        self.explorer_text.insert("end", block_details)
        
        for i, tx in enumerate(block.transactions):
            tx_info = f"""
{i+1}. Transaction {tx.tx_id}
   From: {tx.sender}
   To: {tx.receiver}
   Amount: {tx.amount:.6f} coins
   Fee: {tx.fee:.6f} coins
   Time: {tx.timestamp}
"""
            self.explorer_text.insert("end", tx_info)
    
    def display_transaction_details(self, transaction, block_index):
        """Display detailed transaction information"""
        tx_details = f"""💸 TRANSACTION DETAILS
{'='*50}

Transaction ID: {transaction.tx_id}
Block: {block_index}
From: {transaction.sender}
To: {transaction.receiver}
Amount: {transaction.amount:.6f} coins
Fee: {transaction.fee:.6f} coins
Timestamp: {transaction.timestamp}
Signature: {transaction.signature[:50] if transaction.signature else 'None'}...
"""
        
        self.explorer_text.insert("end", tx_details)
    
    def display_address_details(self, address):
        """Display detailed address information"""
        balance = self.blockchain.get_balance(address)
        
        # Find user name
        user_name = "Unknown"
        for user in self.blockchain.users.values():
            if user.address == address:
                user_name = user.name
                break
        
        address_details = f"""🏠 ADDRESS DETAILS
{'='*50}

Address: {address}
Name: {user_name}
Balance: {balance:.6f} coins

📤 OUTGOING TRANSACTIONS:
{'='*30}
"""
        
        self.explorer_text.insert("end", address_details)
        
        # Show transactions
        for block in self.blockchain.chain:
            for tx in block.transactions:
                if tx.sender == address:
                    tx_info = f"Block {block.index}: Sent {tx.amount:.6f} to {tx.receiver}\n"
                    self.explorer_text.insert("end", tx_info)
        
        self.explorer_text.insert("end", "\n📥 INCOMING TRANSACTIONS:\n" + "="*30 + "\n")
        
        for block in self.blockchain.chain:
            for tx in block.transactions:
                if tx.receiver == address:
                    tx_info = f"Block {block.index}: Received {tx.amount:.6f} from {tx.sender}\n"
                    self.explorer_text.insert("end", tx_info)
    
    def load_education_content(self, topic):
        """Load educational content for a topic"""
        content = {
            "Hashing": """🔐 CRYPTOGRAPHIC HASHING

Hashing is a fundamental concept in blockchain technology. It's a one-way mathematical function that takes any input and produces a fixed-length output called a hash or digest.

Key Properties:
• Deterministic: Same input always produces same output
• Fast Computation: Quick to calculate
• Pre-image Resistance: Cannot reverse engineer input from output  
• Collision Resistance: Nearly impossible for two inputs to produce same output
• Avalanche Effect: Small change in input dramatically changes output

Example in Bitcoin:
• Uses SHA-256 (Secure Hash Algorithm 256-bit)
• Block headers are hashed to create unique block identifiers
• Hash values serve as digital fingerprints

Try the demo to see how changing even one character completely changes the hash!""",
            
            "Digital Signatures": """🔏 DIGITAL SIGNATURES

Digital signatures provide authentication, integrity, and non-repudiation in blockchain transactions.

How It Works:
1. User generates a key pair (private key + public key)
2. Private key is kept secret and used to sign transactions
3. Public key is shared and used by others to verify signatures
4. Mathematical relationship proves signature authenticity

In Blockchain:
• Every transaction is signed by the sender's private key
• Network participants can verify using the public key
• Prevents double-spending and unauthorized transactions
• Uses elliptic curve cryptography (ECDSA) for efficiency

Security: Your private key = your money. Never share it!""",
            
            "Blocks": """📦 BLOCKCHAIN BLOCKS

A block is a container that holds a batch of verified transactions. Blocks are linked together to form the blockchain.

Block Structure:
• Block Header:
  - Index: Position in the chain
  - Timestamp: When block was created
  - Previous Hash: Links to previous block
  - Merkle Root: Summary of all transactions
  - Nonce: Number used for mining
  - Hash: Unique identifier

• Block Body:
  - List of transactions
  - Each transaction is digitally signed

Merkle Tree:
• Efficient way to summarize all transactions
• Allows quick verification without downloading entire block
• Used in Bitcoin and most cryptocurrencies""",
            
            "Consensus": """🤝 CONSENSUS MECHANISMS

Consensus mechanisms ensure all network participants agree on the blockchain state without a central authority.

Major Types:

1. Proof of Work (PoW):
   • Miners solve computational puzzles
   • First to solve broadcasts the block
   • Energy-intensive but secure
   • Used by Bitcoin

2. Proof of Stake (PoS):
   • Validators chosen based on stake
   • More energy efficient
   • Used by Ethereum 2.0

3. Delegated Proof of Stake (DPoS):
   • Token holders vote for delegates
   • Faster but less decentralized
   • Used by EOS

4. Practical Byzantine Fault Tolerance (pBFT):
   • Handles malicious nodes
   • Good for permissioned networks
   • Used in Hyperledger""",
            
            "Smart Contracts": """📜 SMART CONTRACTS

Smart contracts are self-executing programs stored on the blockchain that automatically enforce agreements.

Key Features:
• Autonomy: Run without human intervention
• Self-sufficiency: Operate independently once deployed
• Immutability: Cannot be changed after deployment
• Trustless: No need to trust other parties
• Transparency: Code is publicly verifiable

Use Cases:
• DeFi (Decentralized Finance): Lending, trading, insurance
• NFTs (Non-Fungible Tokens): Digital art, collectibles
• DAOs (Decentralized Autonomous Organizations): Governance
• Supply Chain: Automated payments and tracking
• Voting: Transparent and tamper-proof elections

Programming Languages:
• Solidity (Ethereum)
• Rust (Solana)
• Go (Hyperledger)""",
            
            "Mining": """⛏️ BLOCKCHAIN MINING

Mining is the process of adding new blocks to the blockchain through computational work.

Mining Process:
1. Collect pending transactions
2. Verify transaction validity
3. Create block header with Merkle root
4. Solve proof-of-work puzzle (find nonce)
5. Broadcast new block to network
6. Receive mining reward

Proof of Work:
• Find nonce that makes block hash start with zeros
• Difficulty adjusts to maintain block time
• Higher difficulty = more zeros required
• Exponentially harder as difficulty increases

Mining Economics:
• Miners compete for block rewards
• Must consider electricity costs
• Mining pools share rewards
• ASIC hardware for efficiency

Environmental Concerns:
• High energy consumption
• Moving toward green energy
• Alternative consensus mechanisms""",
            
            "Transactions": """💸 BLOCKCHAIN TRANSACTIONS

Transactions are the basic units of value transfer in a blockchain network.

Transaction Components:
• Sender: Address sending the value
• Receiver: Address receiving the value
• Amount: Quantity being transferred
• Fee: Payment to miners/validators
• Signature: Cryptographic proof of authorization
• Timestamp: When transaction was created

Transaction Lifecycle:
1. User creates transaction
2. Signs with private key
3. Broadcasts to network
4. Validated by nodes
5. Added to mempool (pending)
6. Miner includes in block
7. Block added to chain
8. Transaction confirmed

UTXO Model (Bitcoin):
• Tracks unspent transaction outputs
• No account balances
• Transactions consume UTXOs and create new ones

Account Model (Ethereum):
• Maintains account balances
• Simpler to understand
• Supports smart contracts better"""
        }
        
        self.education_content.delete("1.0", "end")
        self.education_content.insert("1.0", content.get(topic, "Content not available"))
    
    def run_education_demo(self):
        """Run educational demonstration"""
        topic = self.education_topic_var.get()
        
        demos = {
            "Hashing": self.demo_hashing,
            "Digital Signatures": self.demo_signatures,
            "Blocks": self.demo_blocks,
            "Consensus": self.demo_consensus,
            "Smart Contracts": self.demo_smart_contracts,
            "Mining": self.demo_mining,
            "Transactions": self.demo_transactions
        }
        
        demo_func = demos.get(topic)
        if demo_func:
            demo_func()
        else:
            messagebox.showinfo("Demo", f"Demo for {topic} not implemented yet")
    
    def demo_hashing(self):
        """Demonstrate hashing"""
        demo_window = ctk.CTkToplevel(self.root)
        demo_window.title("🔐 Hashing Demo")
        demo_window.geometry("600x400")
        
        ctk.CTkLabel(demo_window, text="Hash Function Demo", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)
        
        # Input
        ctk.CTkLabel(demo_window, text="Enter text to hash:").pack(pady=5)
        input_entry = ctk.CTkEntry(demo_window, width=400, placeholder_text="Hello, World!")
        input_entry.pack(pady=5)
        
        # Results
        result_frame = ctk.CTkFrame(demo_window)
        result_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        result_text = ctk.CTkTextbox(result_frame)
        result_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        def update_hash():
            text = input_entry.get()
            if text:
                sha256_hash = hashlib.sha256(text.encode()).hexdigest()
                result = f"""Input: {text}
SHA-256: {sha256_hash}

Properties:
• Fixed length: Always 64 characters (256 bits)
• Deterministic: Same input = same output
• Avalanche effect: Small change = big difference

Try changing one character and see how the hash changes completely!"""
                result_text.delete("1.0", "end")
                result_text.insert("1.0", result)
        
        input_entry.bind("<KeyRelease>", lambda e: update_hash())
        ctk.CTkButton(demo_window, text="Calculate Hash", command=update_hash).pack(pady=10)
        
        # Set initial value
        input_entry.insert(0, "Hello, World!")
        update_hash()
    
    def demo_signatures(self):
        """Demonstrate digital signatures"""
        if not self.current_user:
            messagebox.showerror("Error", "Please select a user first")
            return
        
        demo_window = ctk.CTkToplevel(self.root)
        demo_window.title("🔏 Digital Signature Demo")
        demo_window.geometry("700x500")
        
        ctk.CTkLabel(demo_window, text="Digital Signature Demo", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)
        
        # Message input
        ctk.CTkLabel(demo_window, text="Message to sign:").pack(pady=5)
        message_entry = ctk.CTkEntry(demo_window, width=500, placeholder_text="I authorize this transaction")
        message_entry.pack(pady=5)
        
        # Results
        result_frame = ctk.CTkFrame(demo_window)
        result_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        result_text = ctk.CTkTextbox(result_frame)
        result_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        def sign_message():
            message = message_entry.get()
            if message:
                # Sign the message
                signature = self.current_user.private_key.sign(message.encode())
                signature_hex = signature.hex()
                
                # Verify the signature
                try:
                    self.current_user.public_key.verify(signature, message.encode())
                    verification = "✅ VALID"
                except:
                    verification = "❌ INVALID"
                
                result = f"""🔏 DIGITAL SIGNATURE DEMONSTRATION

Signer: {self.current_user.name}
Address: {self.current_user.address}

Message: "{message}"

🔐 Private Key (kept secret):
{self.current_user.private_key.to_string().hex()[:50]}...

🔓 Public Key (shared publicly):
{self.current_user.public_key.to_string().hex()[:50]}...

✍️ Digital Signature:
{signature_hex[:100]}...

🔍 Verification: {verification}

Anyone with the public key can verify this signature proves the message was signed by the private key holder, without revealing the private key!"""
                
                result_text.delete("1.0", "end")
                result_text.insert("1.0", result)
        
        ctk.CTkButton(demo_window, text="Sign & Verify Message", command=sign_message).pack(pady=10)
        
        # Set initial message
        message_entry.insert(0, "I authorize this transaction")
    
    def demo_blocks(self):
        """Demonstrate block structure"""
        demo_window = ctk.CTkToplevel(self.root)
        demo_window.title("📦 Block Structure Demo")
        demo_window.geometry("800x600")
        
        ctk.CTkLabel(demo_window, text="Block Structure Demo", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)
        
        # Show latest block
        latest_block = self.blockchain.get_latest_block()
        
        result_frame = ctk.CTkFrame(demo_window)
        result_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        result_text = ctk.CTkTextbox(result_frame)
        result_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        block_info = f"""📦 BLOCK STRUCTURE DEMONSTRATION

Block #{latest_block.index}

📋 BLOCK HEADER:
{'='*50}
Index: {latest_block.index}
Timestamp: {latest_block.timestamp}
Previous Hash: {latest_block.previous_hash}
Current Hash: {latest_block.hash}
Nonce: {latest_block.nonce}
Miner: {latest_block.miner}
Merkle Root: {latest_block.merkle_root}

📝 BLOCK BODY:
{'='*50}
Transactions: {len(latest_block.transactions)}

Transaction Details:"""
        
        for i, tx in enumerate(latest_block.transactions):
            tx_info = f"""
{i+1}. {tx.tx_id}
   From: {tx.sender}
   To: {tx.receiver}
   Amount: {tx.amount:.6f} coins
   Fee: {tx.fee:.6f} coins"""
            block_info += tx_info
        
        block_info += f"""

🔗 BLOCK LINKING:
{'='*50}
This block's hash becomes the "Previous Hash" of the next block, creating an immutable chain. Changing any data in this block would change its hash, breaking the chain and alerting the network to tampering.

🌳 MERKLE TREE:
{'='*50}
The Merkle Root {latest_block.merkle_root[:20]}... is a summary of all transactions. It allows efficient verification of transaction inclusion without downloading the entire block."""
        
        result_text.insert("1.0", block_info)
    
    def demo_consensus(self):
        """Demonstrate consensus mechanisms"""
        messagebox.showinfo("Consensus Demo", "Start mining to see Proof of Work consensus in action! Check the Mining tab.")
    
    def demo_smart_contracts(self):
        """Demonstrate smart contracts"""
        messagebox.showinfo("Smart Contracts Demo", "Visit the Smart Contracts tab to deploy and interact with contracts!")
    
    def demo_mining(self):
        """Demonstrate mining"""
        messagebox.showinfo("Mining Demo", "Visit the Mining tab to start mining blocks and see the proof-of-work algorithm in action!")
    
    def demo_transactions(self):
        """Demonstrate transactions"""
        messagebox.showinfo("Transactions Demo", "Visit the Transactions tab to create and send transactions between users!")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

# Sample contract templates and additional utility functions
def main():
    """Main function to run the application"""
    try:
        app = BlockchainSimulatorApp()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Error", f"Application failed to start: {e}")

if __name__ == "__main__":
    main()