import hashlib
import time
import json
import random
import matplotlib.pyplot as plt
import networkx as nx

# === MERKLE TREE CLASS ===
class MerkleTree:
    def __init__(self, transactions):
        self.transactions = transactions
        self.root = self.build_merkle_tree(transactions)

    def build_merkle_tree(self, transactions):
        if not transactions:
            return hashlib.sha256("GENESIS_BLOCK".encode()).hexdigest()
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

# === BLOCK CLASS ===
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
        self.mining_time = 0

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
        start_time = time.time()
        while self.hash[:difficulty] != "0" * difficulty:
            self.nonce += 1
            self.hash = self.calculate_hash()
        end_time = time.time()
        self.mining_time = end_time - start_time
        print(f"✅ Block {self.index} mined with hash: {self.hash}")

# === BLOCKCHAIN CLASS ===
class Blockchain:
    def __init__(self, difficulty=4):
        self.chain = [self.create_genesis_block()]
        self.difficulty = difficulty

    def create_genesis_block(self):
        print("\n🌱 Creating Genesis Block...")
        return Block(0, "0", [])

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, transactions):
        previous_block = self.chain[-1]
        new_block = Block(len(self.chain), previous_block.hash, transactions)
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)

    def attempt_hack(self, block_index, fake_transaction):
        print("\n⚠️ ATTEMPTED HACK: Modifying Blockchain Data...")
        if block_index >= len(self.chain):
            print("❌ Invalid block index for hacking!")
            return False

        self.chain[block_index].transactions.append(fake_transaction)
        self.chain[block_index].merkle_tree = MerkleTree([str(tx) for tx in self.chain[block_index].transactions])
        self.chain[block_index].merkle_root = self.chain[block_index].merkle_tree.root
        self.chain[block_index].hash = self.chain[block_index].calculate_hash()

        print("🚨 ALERT! Blockchain Tampering Detected!")
        return False  # This proves tampering invalidates the chain

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
def visualize_blockchain(blockchain, hacked=False):
    G = nx.DiGraph()
    for block in blockchain.chain:
        label = f"Block {block.index}\nHash: {block.hash[:6]}...\nMerkle Root: {block.merkle_root[:6]}"
        G.add_node(block.index, label=label)
        if block.index > 0:
            G.add_edge(block.index - 1, block.index, color="red" if hacked and block.index == 2 else "black")

    pos = nx.spring_layout(G, seed=42)
    edges = G.edges()
    edge_colors = ["red" if (hacked and 2 in edge) else "black" for edge in edges]

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", edge_color=edge_colors, font_size=8, font_weight="bold")
    labels = {n: G.nodes[n]['label'] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    title_text = "Blockchain Security Visualization"
    if hacked:
        title_text += " (Tampering Detected)"
    plt.title(title_text)
    plt.show()

def visualize_mining_times(blockchain):
    blocks = [block.index for block in blockchain.chain]
    mining_times = [block.mining_time for block in blockchain.chain]

    plt.figure(figsize=(10, 5))
    plt.bar(blocks, mining_times, color="blue")
    plt.xlabel("Block Index")
    plt.ylabel("Mining Time (seconds)")
    plt.title("Mining Time per Block")
    plt.xticks(blocks)
    plt.show()

# === RUN BLOCKCHAIN SIMULATION ===
blockchain = Blockchain()

# Add legitimate transactions
blockchain.add_block(["A sends 20 BTC to B"])
blockchain.add_block(["B sends 5 BTC to C"])

# Attempt Hack (Fails)
blockchain.attempt_hack(1, "Hacker tries to send 100 BTC to himself")

# Validate Blockchain (Detects Hack)
blockchain.is_chain_valid()

# Visualize Blockchain Process
visualize_blockchain(blockchain, hacked=True)
visualize_mining_times(blockchain)
