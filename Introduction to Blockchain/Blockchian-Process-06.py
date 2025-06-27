import hashlib
import time
import json
import random
import matplotlib.pyplot as plt
import networkx as nx

# === TRANSACTION CLASS ===
class Transaction:
    def __init__(self, sender, receiver, amount):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.timestamp = time.time()

    def to_dict(self):
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "timestamp": self.timestamp
        }

    def __str__(self):
        return json.dumps(self.to_dict(), sort_keys=True)

# === BLOCK CLASS (REPRESENTING TRANSACTION) ===
class Block:
    def __init__(self, index, previous_hash, transactions, nonce=0):
        self.index = index
        self.timestamp = time.time()
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "transactions": [str(tx) for tx in self.transactions],
            "nonce": self.nonce
            }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty):
        print(f"\n⛏️ Mining Block {self.index}... (Difficulty: {difficulty})")
        while self.hash[:difficulty] != "0" * difficulty:
            self.nonce += 1
            self.hash = self.calculate_hash()
        print(f"✅ Block {self.index} mined with hash: {self.hash}")

# === BLOCKCHAIN CLASS (DISTRIBUTED VALIDATION) ===
class Blockchain:
    def __init__(self, difficulty=4):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []
        self.nodes = ["Node1", "Node2", "Node3", "Node4"]  # Simulated network nodes
        self.difficulty = difficulty

    def create_genesis_block(self):
        print("\n🌱 Creating Genesis Block...")
        return Block(0, "0", [])

    def add_transaction(self, sender, receiver, amount):
        transaction = Transaction(sender, receiver, amount)
        self.pending_transactions.append(transaction)
        print(f"\n📨 New Transaction Requested: {sender} ➡ {receiver} | Amount: {amount}")

    def create_block(self):
        if not self.pending_transactions:
            print("\n⚠️ No transactions to process!")
            return None

        previous_block = self.chain[-1]
        new_block = Block(len(self.chain), previous_block.hash, self.pending_transactions)
        self.pending_transactions = []
        return new_block

    def broadcast_block(self, block):
        print("\n🌎 Broadcasting Block to Network Nodes...")
        for node in self.nodes:
            print(f"📡 {node} received Block {block.index} for validation.")

    def validate_block(self, block):
        print("\n✅ Nodes are validating Block...")
        for node in self.nodes:
            validation = random.choice([True, True, True, False])  # 75% chance of success
            if not validation:
                print(f"❌ {node} rejected Block {block.index} (Validation Failed!)")
                return False
        print(f"✅ Block {block.index} validated by all nodes.")
        return True

    def add_block(self, block):
        if self.validate_block(block):
            block.mine_block(self.difficulty)
            self.chain.append(block)
            print(f"📦 Block {block.index} added to Blockchain!")
        else:
            print("❌ Block Validation Failed! Block NOT added.")

    def is_chain_valid(self):
        print("\n🔍 Validating Full Blockchain...")
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash():
                print(f"❌ Block {current_block.index} has been altered!")
                return False
            if current_block.previous_hash != previous_block.hash:
                print(f"❌ Block {current_block.index} has an invalid previous hash!")
                return False

        print("✅ Blockchain is valid and secure!")
        return True

# === VISUALIZATION FUNCTIONS ===
def visualize_blockchain(blockchain):
    G = nx.DiGraph()
    for block in blockchain.chain:
        label = f"Block {block.index}\nHash: {block.hash[:6]}...\nTxs: {len(block.transactions)}"
        G.add_node(block.index, label=label)
        if block.index > 0:
            G.add_edge(block.index - 1, block.index)

    pos = nx.spring_layout(G, seed=42)
    labels = {n: G.nodes[n]['label'] for n in G.nodes}

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", edge_color="black", font_size=8, font_weight="bold")
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    plt.title("Blockchain Transaction & Validation Flow")
    plt.show()

def visualize_network_nodes(blockchain, block):
    G = nx.Graph()
    central_node = "Block " + str(block.index)

    G.add_node(central_node, label=central_node, color="red")

    for node in blockchain.nodes:
        G.add_node(node, label=node, color="green")
        G.add_edge(central_node, node)

    pos = nx.spring_layout(G, seed=42)
    labels = {n: G.nodes[n]['label'] for n in G.nodes}

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", edge_color="black", font_size=8, font_weight="bold")
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    plt.title("Blockchain Network Nodes & Block Broadcast")
    plt.show()

# === RUN BLOCKCHAIN SIMULATION ===
blockchain = Blockchain()

# Step 1: Transaction Request
blockchain.add_transaction("Alice", "Bob", 50)
blockchain.add_transaction("Charlie", "Dave", 20)

# Step 2: Block Creation
new_block = blockchain.create_block()

# Step 3: Broadcasting the Block
if new_block:
    blockchain.broadcast_block(new_block)

# Step 4: Nodes Validate & Add Block
if new_block:
    blockchain.add_block(new_block)

# Final Blockchain Validation
blockchain.is_chain_valid()

# Visualizations
visualize_blockchain(blockchain)
visualize_network_nodes(blockchain, new_block)
