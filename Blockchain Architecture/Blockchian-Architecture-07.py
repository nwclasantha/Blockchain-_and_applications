import hashlib
import time
import json
import random
import matplotlib.pyplot as plt
import networkx as nx

# === MERKLE TREE CLASS (Ensures Transaction Integrity) ===
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

# === BLOCK CLASS (Has Previous Hash, Current Hash, Merkle Root) ===
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

# === BLOCKCHAIN CLASS (Ensures Chain Integrity) ===
class Blockchain:
    def __init__(self, difficulty=4):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []
        self.difficulty = difficulty

    def create_genesis_block(self):
        print("\n🌱 Creating Genesis Block...")
        return Block(0, "0", [])

    def add_transaction(self, sender, receiver, amount):
        transaction = Transaction(sender, receiver, amount)
        self.pending_transactions.append(transaction)
        print(f"\n📨 New Transaction: {sender} ➡ {receiver} | Amount: {amount}")

    def create_block(self):
        if not self.pending_transactions:
            print("\n⚠️ No transactions to process!")
            return None

        previous_block = self.chain[-1]
        new_block = Block(len(self.chain), previous_block.hash, self.pending_transactions)
        self.pending_transactions = []
        return new_block

    def add_block(self, block):
        block.mine_block(self.difficulty)
        self.chain.append(block)
        print(f"📦 Block {block.index} added to Blockchain!")

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
        label = f"Block {block.index}\nHash: {block.hash[:6]}...\nMerkle Root: {block.merkle_root[:6]}"
        G.add_node(block.index, label=label)
        if block.index > 0:
            G.add_edge(block.index - 1, block.index)

    pos = nx.spring_layout(G, seed=42)
    labels = {n: G.nodes[n]['label'] for n in G.nodes}

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", edge_color="black", font_size=8, font_weight="bold")
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    plt.title("Blockchain Structure & Integrity")
    plt.show()

# === RUN BLOCKCHAIN SIMULATION ===
blockchain = Blockchain()

# Add Transactions
blockchain.add_transaction("Alice", "Bob", 50)
blockchain.add_transaction("Charlie", "Dave", 30)

# Create and Add New Block
new_block = blockchain.create_block()
if new_block:
    blockchain.add_block(new_block)

# Add More Transactions
blockchain.add_transaction("Eve", "Frank", 100)
blockchain.add_transaction("George", "Helen", 25)

# Create and Add Another Block
new_block = blockchain.create_block()
if new_block:
    blockchain.add_block(new_block)

# Validate Blockchain
blockchain.is_chain_valid()

# Visualize Blockchain
visualize_blockchain(blockchain)
