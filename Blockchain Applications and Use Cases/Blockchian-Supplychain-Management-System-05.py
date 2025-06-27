import hashlib
import time
import json
import random
import matplotlib.pyplot as plt
import networkx as nx

# === SUPPLY CHAIN PARTICIPANTS ===
SUPPLY_CHAIN_STAGES = [
    "Farm",
    "Storage",
    "Food Processing",
    "Food Manufacturing",
    "Distribution",
    "Retailer",
    "Consumer"
]

# === SUPPLY CHAIN TRANSACTION CLASS ===
class SupplyTransaction:
    def __init__(self, product_id, stage, timestamp=None):
        self.product_id = product_id
        self.stage = stage
        self.timestamp = timestamp if timestamp else time.time()

    def to_dict(self):
        return {
            "product_id": self.product_id,
            "stage": self.stage,
            "timestamp": self.timestamp
        }

    def __str__(self):
        return json.dumps(self.to_dict(), sort_keys=True)

# === MERKLE TREE CLASS (For Data Integrity) ===
class MerkleTree:
    def __init__(self, transactions):
        self.transactions = transactions
        self.root = self.build_merkle_tree(transactions)

    def build_merkle_tree(self, transactions):
        if not transactions:
            return hashlib.sha256("NO_TRANSACTIONS".encode()).hexdigest()
        hashes = [self.hash_transaction(tx) for tx in transactions]
        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])
            hashes = [self.hash_pair(hashes[i], hashes[i+1]) for i in range(0, len(hashes), 2)]
        return hashes[0]

    def hash_transaction(self, transaction):
        return hashlib.sha256(transaction.encode()).hexdigest()

    def hash_pair(self, hash1, hash2):
        return hashlib.sha256((hash1 + hash2).encode()).hexdigest()

# === BLOCK CLASS (Tracking Supply Chain Stages) ===
class Block:
    def __init__(self, index, previous_hash, transactions, nonce=0):
        self.index = index
        self.timestamp = time.time()
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.merkle_tree = MerkleTree([str(tx) for tx in transactions])
        self.merkle_root = self.merkle_tree.root
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "merkle_root": self.merkle_root,
            "nonce": self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty):
        print(f"\n⛏️ Mining Block {self.index}... (Difficulty: {difficulty})")
        while self.hash[:difficulty] != "0" * difficulty:
            self.nonce += 1
            self.hash = self.calculate_hash()
        print(f"✅ Block {self.index} mined with hash: {self.hash}")

# === BLOCKCHAIN CLASS (Immutable & Transparent Supply Chain) ===
class SupplyBlockchain:
    def __init__(self, difficulty=4):
        self.chain = [self.create_genesis_block()]
        self.products = {}
        self.difficulty = difficulty

    def create_genesis_block(self):
        print("\n🌱 Creating Genesis Block...")
        return Block(0, "0", [])

    def add_product(self, product_id):
        if product_id in self.products:
            print(f"⚠️ Product {product_id} already exists in the supply chain!")
            return False
        self.products[product_id] = []
        print(f"✅ Product {product_id} added to the supply chain.")
        return True

    def update_stage(self, product_id, stage):
        if product_id not in self.products:
            print(f"❌ Product {product_id} is not registered!")
            return False
        if len(self.products[product_id]) >= len(SUPPLY_CHAIN_STAGES):
            print(f"⚠️ Product {product_id} has already reached the final stage.")
            return False

        transaction = SupplyTransaction(product_id, stage)
        self.products[product_id].append(transaction)
        print(f"📦 Product {product_id} moved to '{stage}' stage.")
        return transaction

    def add_block(self, transactions):
        previous_block = self.chain[-1]
        new_block = Block(len(self.chain), previous_block.hash, transactions)
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)

    def track_product(self, product_id):
        if product_id not in self.products:
            print(f"❌ Product {product_id} not found in the blockchain!")
            return None
        return self.products[product_id]

    def is_chain_valid(self):
        print("\n🔍 Validating Blockchain Integrity...")
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
        label = f"Block {block.index}\nHash: {block.hash[:6]}...\nTransactions: {len(block.transactions)}"
        G.add_node(block.index, label=label)
        if block.index > 0:
            G.add_edge(block.index - 1, block.index)

    pos = nx.spring_layout(G, seed=42)
    labels = {n: G.nodes[n]['label'] for n in G.nodes}

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", edge_color="black", font_size=8, font_weight="bold")
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    plt.title("Blockchain Supply Chain Management")
    plt.show()

def visualize_product_flow(product_tracking):
    stages = [tx.stage for tx in product_tracking]
    timestamps = [tx.timestamp for tx in product_tracking]

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, stages, marker="o", linestyle="-", color="green")
    plt.xlabel("Timestamp")
    plt.ylabel("Supply Chain Stages")
    plt.title("Product Movement in Supply Chain")
    plt.xticks(rotation=45)
    plt.show()

# === RUN BLOCKCHAIN SUPPLY CHAIN SYSTEM ===
supply_chain = SupplyBlockchain()

# Register Product
supply_chain.add_product("PRODUCT123")

# Move Product Through Stages
transactions = []
for stage in SUPPLY_CHAIN_STAGES:
    transaction = supply_chain.update_stage("PRODUCT123", stage)
    if transaction:
        transactions.append(transaction)

# Add Transactions to Blockchain
supply_chain.add_block(transactions)

# Validate Blockchain
supply_chain.is_chain_valid()

# Track Product Journey
product_journey = supply_chain.track_product("PRODUCT123")
print("\n🔍 Product Journey:", [tx.to_dict() for tx in product_journey])

# Visualize Blockchain & Product Flow
visualize_blockchain(supply_chain)
visualize_product_flow(product_journey)
