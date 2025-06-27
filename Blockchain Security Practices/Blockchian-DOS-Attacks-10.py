import hashlib
import time
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# === SIMULATION PARAMETERS ===
NUM_NODES = 15          # Total nodes in the blockchain network
DDOS_ATTACKERS = 0.1    # Percentage of nodes performing DDoS attacks
TRANSACTION_FLOOD = 10 # Number of fake transactions per attack
SIMULATION_ROUNDS = 10  # Number of mining rounds
NORMAL_TX_PER_BLOCK = 5 # Normal transactions per block

# === TRANSACTION CLASS ===
class Transaction:
    def __init__(self, sender, receiver, amount, fee):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.fee = fee  # Higher fee speeds up transaction
        self.timestamp = time.time()
        self.tx_id = hashlib.sha256(f"{sender}{receiver}{amount}{self.timestamp}".encode()).hexdigest()[:10]

    def __str__(self):
        return f"{self.sender} ➡ {self.receiver} | {self.amount} BTC | Fee: {self.fee}"

# === BLOCK CLASS ===
class Block:
    def __init__(self, index, previous_hash, transactions, mined_by):
        self.index = index
        self.timestamp = time.time()
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.mined_by = mined_by
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_data = hashlib.sha256(f"{self.index}{self.previous_hash}{self.timestamp}".encode()).hexdigest()
        return block_data

# === NODE CLASS ===
class Node:
    def __init__(self, node_id, is_attacker=False):
        self.node_id = node_id
        self.is_attacker = is_attacker
        self.transaction_pool = []

    def create_transaction(self):
        """Creates a normal transaction."""
        return Transaction(f"User_{random.randint(1, 100)}", f"User_{random.randint(1, 100)}", random.uniform(0.1, 10), random.uniform(0.0001, 0.01))

    def perform_ddos_attack(self):
        """Attacker node floods the network with dust transactions."""
        for _ in range(TRANSACTION_FLOOD):
            tx = Transaction(f"Bot_{random.randint(1, 100)}", f"Bot_{random.randint(1, 100)}", random.uniform(0.00001, 0.001), 0)
            self.transaction_pool.append(tx)
        print(f"🚨 Node {self.node_id} performed a DDoS attack! Flooded network with {TRANSACTION_FLOOD} transactions.")

# === BLOCKCHAIN CLASS ===
class Blockchain:
    def __init__(self, num_nodes, ddos_attackers):
        self.nodes = []
        self.attacker_nodes = []
        self.honest_nodes = []
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []

        for i in range(num_nodes):
            is_attacker = random.random() < ddos_attackers
            node = Node(i, is_attacker)
            self.nodes.append(node)
            if is_attacker:
                self.attacker_nodes.append(node)
            else:
                self.honest_nodes.append(node)

    def create_genesis_block(self):
        print("\n🌱 Creating Genesis Block...")
        return Block(0, "0", [], "Genesis")

    def simulate_transactions(self):
        """Simulate normal and attack transactions in the network."""
        for node in self.nodes:
            if node.is_attacker:
                node.perform_ddos_attack()
            else:
                self.pending_transactions.append(node.create_transaction())

        # Sorting transactions by fee (higher fee gets prioritized)
        self.pending_transactions.sort(key=lambda tx: tx.fee, reverse=True)

    def mine_block(self):
        """Mines a block, processing only a limited number of transactions."""
        miner = random.choice(self.honest_nodes)  # Honest miners mine blocks
        transactions_to_include = self.pending_transactions[:NORMAL_TX_PER_BLOCK]
        self.pending_transactions = self.pending_transactions[NORMAL_TX_PER_BLOCK:]  # Remaining transactions delayed
        new_block = Block(len(self.chain), self.chain[-1].hash, transactions_to_include, miner.node_id)
        self.chain.append(new_block)
        print(f"✅ Block {new_block.index} mined by Node {miner.node_id}. Processed {len(transactions_to_include)} transactions.")

    def run_simulation(self, rounds):
        """Runs multiple rounds of blockchain operation."""
        print("\n🔍 Starting Blockchain DDoS Attack Simulation...\n")
        for round_num in range(1, rounds + 1):
            print(f"\n🚀 Round {round_num}...\n")
            self.simulate_transactions()
            self.mine_block()

    def visualize_results(self):
        """Graphically display attack impact on the blockchain."""
        block_numbers = [block.index for block in self.chain]
        transactions_per_block = [len(block.transactions) for block in self.chain]
    
        # Transactions Processed per Block
        plt.figure(figsize=(12, 6))
        plt.bar(block_numbers, transactions_per_block, color="blue")
        plt.xlabel("Block Index")
        plt.ylabel("Transactions Processed")
        plt.title("Blockchain Transaction Processing under DDoS Attack")
        plt.show()
    
        # Attacker vs Honest Nodes
        node_ids = [node.node_id for node in self.nodes]
        attacker_flags = [1 if node.is_attacker else 0 for node in self.nodes]
    
        plt.figure(figsize=(12, 6))
        plt.bar(node_ids, attacker_flags, color=["red" if node.is_attacker else "green" for node in self.nodes])
        plt.xlabel("Node ID")
        plt.ylabel("Attacker Status (1 = Attacker, 0 = Honest)")
        plt.title("Blockchain Network - Honest vs. Attacker Nodes")
        plt.show()
    
        # Network Graph Representation
        G = nx.DiGraph()
        for node in self.nodes:
            node_label = f"Node {node.node_id}\nAttacker: {node.is_attacker}"
            G.add_node(node.node_id, label=node_label)
    
        for node in self.nodes:
            if node.is_attacker:
                for _ in range(random.randint(1, 3)):  # Simulate attack connections
                    target = random.choice(self.nodes)
                    if target != node:
                        G.add_edge(node.node_id, target.node_id)
    
        pos = nx.spring_layout(G, seed=42)
        labels = {n: G.nodes[n]['label'] for n in G.nodes}
    
        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", edge_color="black", font_size=10, font_weight="bold")
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)  # ✅ FIXED FUNCTION
        plt.title("Blockchain Network - DDoS Attack Simulation")
        plt.show()

# === RUN DDoS ATTACK SIMULATION ===
blockchain = Blockchain(NUM_NODES, DDOS_ATTACKERS)
blockchain.run_simulation(SIMULATION_ROUNDS)
blockchain.visualize_results()
