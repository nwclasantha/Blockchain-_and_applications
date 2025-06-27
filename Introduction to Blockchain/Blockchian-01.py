import hashlib
import time
import json
import random
import matplotlib.pyplot as plt
import networkx as nx

# Transaction class
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

# Block class
class Block:
    def __init__(self, index, previous_hash, transactions, nonce=0):
        self.index = index
        self.timestamp = time.time()
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.nonce = nonce
        self.hash = self.calculate_hash()
        self.mining_time = 0  # Track mining time

    def calculate_hash(self):
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "transactions": [tx.to_dict() for tx in self.transactions],
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
        self.mining_time = end_time - start_time  # Save mining time
        print(f"✅ Block {self.index} mined in {self.mining_time:.2f}s with hash: {self.hash}")

# Blockchain class
class Blockchain:
    def __init__(self, difficulty=3):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []
        self.mining_reward = 10
        self.difficulty = difficulty

    def create_genesis_block(self):
        print("\n🌱 Creating Genesis Block...")
        return Block(0, "0", [], nonce=0)

    def get_latest_block(self):
        return self.chain[-1]

    def add_transaction(self, sender, receiver, amount):
        print(f"\n💳 New Transaction: {sender} ➡ {receiver} | Amount: {amount}")
        transaction = Transaction(sender, receiver, amount)
        self.pending_transactions.append(transaction)

    def mine_pending_transactions(self, miner_address):
        if not self.pending_transactions:
            print("\n⚠️ No transactions to mine.")
            return

        print("\n🚀 Transactions ready for mining:")
        for tx in self.pending_transactions:
            print(f" - {tx.sender} ➡ {tx.receiver} | Amount: {tx.amount}")

        new_block = Block(len(self.chain), self.get_latest_block().hash, self.pending_transactions)
        new_block.mine_block(self.difficulty)

        self.chain.append(new_block)
        print(f"📦 Block {new_block.index} added to blockchain!")

        self.pending_transactions = [Transaction("System", miner_address, self.mining_reward)]
        print(f"🎉 Miner {miner_address} received {self.mining_reward} as reward!")

        self.is_chain_valid()  # Auto-check blockchain integrity after every block

    def is_chain_valid(self):
        print("\n🔍 Validating Blockchain Integrity...")
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash():
                print(f"❌ Block {current_block.index} has been tampered!")
                return False
            if current_block.previous_hash != previous_block.hash:
                print(f"❌ Block {current_block.index} has an invalid previous hash!")
                return False

        print("✅ Blockchain is valid and secure!")
        return True

    def display_chain(self):
        print("\n📜 Full Blockchain:")
        for block in self.chain:
            print(f"\n🧩 Block {block.index}:")
            print(f"   ⏳ Timestamp: {block.timestamp}")
            print(f"   🔗 Previous Hash: {block.previous_hash}")
            print(f"   🔑 Hash: {block.hash}")
            print(f"   ⏳ Mining Time: {block.mining_time:.2f}s")
            print(f"   📝 Transactions: {json.dumps([tx.to_dict() for tx in block.transactions], indent=4)}")

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
    plt.title("Blockchain Structure")
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

def visualize_transactions_per_block(blockchain):
    blocks = [block.index for block in blockchain.chain]
    transactions_count = [len(block.transactions) for block in blockchain.chain]

    plt.figure(figsize=(10, 5))
    plt.bar(blocks, transactions_count, color="green")
    plt.xlabel("Block Index")
    plt.ylabel("Number of Transactions")
    plt.title("Transactions Per Block")
    plt.xticks(blocks)
    plt.show()

# === SIMULATION START ===
print("🚀 Initializing Blockchain...")
blockchain = Blockchain(difficulty=4)

# Simulating transactions
blockchain.add_transaction("Alice", "Bob", 50)
blockchain.add_transaction("Bob", "Charlie", 30)

# Mining a new block
blockchain.mine_pending_transactions("Miner1")

# More transactions
blockchain.add_transaction("Charlie", "Dave", 20)
blockchain.add_transaction("Dave", "Eve", 10)

# Mining another block
blockchain.mine_pending_transactions("Miner2")

# Display full blockchain
blockchain.display_chain()

# Visualizing the blockchain
visualize_blockchain(blockchain)
visualize_mining_times(blockchain)
visualize_transactions_per_block(blockchain)
