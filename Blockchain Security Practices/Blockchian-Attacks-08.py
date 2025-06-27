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
        self.tx_id = hashlib.sha256(f"{sender}{receiver}{amount}{self.timestamp}".encode()).hexdigest()[:10]

    def to_dict(self):
        return {
            "tx_id": self.tx_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "timestamp": self.timestamp
        }

    def __str__(self):
        return json.dumps(self.to_dict(), sort_keys=True)

# === BLOCK CLASS ===
class Block:
    def __init__(self, index, previous_hash, transactions, nonce=0):
        self.index = index
        self.timestamp = time.time()
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_data = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "transactions": [str(tx) for tx in self.transactions],
            "nonce": self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_data.encode()).hexdigest()

    def mine_block(self, difficulty):
        print(f"\n⛏️ Mining Block {self.index}... (Difficulty: {difficulty})")
        while self.hash[:difficulty] != "0" * difficulty:
            self.nonce += 1
            self.hash = self.calculate_hash()
        print(f"✅ Block {self.index} mined with hash: {self.hash}")

# === BLOCKCHAIN CLASS (With Attack Simulation) ===
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

    # === ATTACK SIMULATIONS ===

    def double_spending_attack(self, sender, receiver):
        print("\n🚨 Simulating Double Spending Attack...")
        fake_tx = Transaction(sender, receiver, 1000)
        self.pending_transactions.append(fake_tx)
        print(f"⚠️ Fake Transaction Created: {sender} ➡ {receiver} | 1000 BTC")
        return fake_tx

    def fifty_one_percent_attack(self):
        print("\n🚨 Simulating 51% Attack...")
        attacker_blocks = random.randint(2, 4)
        for _ in range(attacker_blocks):
            fake_block = Block(len(self.chain), "FakeHash", [])
            fake_block.hash = "0000FakeBlock"
            self.chain.append(fake_block)
        print("⚠️ Attackers gained majority control and created fake blocks!")

    def selfish_mining_attack(self):
        print("\n🚨 Simulating Selfish Mining Attack...")
        print("⚠️ Malicious miners withhold blocks and delay network updates.")

    def sybil_attack(self):
        print("\n🚨 Simulating Sybil Attack...")
        fake_nodes = [f"FakeNode{i}" for i in range(10)]
        print(f"⚠️ Attackers create {len(fake_nodes)} fake nodes to manipulate consensus.")

    def reentrancy_attack(self):
        print("\n🚨 Simulating Reentrancy Attack...")
        print("⚠️ Smart contract exploited to drain funds multiple times before update.")

    def eclipse_attack(self):
        print("\n🚨 Simulating Eclipse Attack...")
        victim_node = random.choice(["Node1", "Node2", "Node3"])
        print(f"⚠️ {victim_node} is isolated from the network, receiving false data.")

    def dao_attack(self):
        print("\n🚨 Simulating DAO Attack...")
        print("⚠️ Smart contract vulnerability exploited to withdraw extra funds.")

    def defi_attack(self):
        print("\n🚨 Simulating DeFi Attack...")
        print("⚠️ Exploit used to manipulate DeFi liquidity pools.")

    def identity_privacy_attack(self):
        print("\n🚨 Simulating Identity Privacy Attack...")
        print("⚠️ Attackers attempt to trace transactions and de-anonymize users.")

    def transaction_information_attack(self):
        print("\n🚨 Simulating Transaction Information Attack...")
        print("⚠️ Data leaks allow tracking of blockchain transactions.")

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
    plt.title("Blockchain Security - Attack Simulations")
    plt.show()

# === RUN BLOCKCHAIN ATTACK SIMULATION ===
blockchain = Blockchain()

# Step 1: Normal Transactions
blockchain.add_transaction("Alice", "Bob", 50)
blockchain.add_transaction("Charlie", "Dave", 30)

# Step 2: Create and Add New Block
new_block = blockchain.create_block()
if new_block:
    blockchain.add_block(new_block)

# Step 3: Simulate Attacks
blockchain.fifty_one_percent_attack()
blockchain.selfish_mining_attack()
blockchain.sybil_attack()
blockchain.reentrancy_attack()
blockchain.eclipse_attack()
blockchain.dao_attack()
blockchain.defi_attack()
blockchain.identity_privacy_attack()
blockchain.transaction_information_attack()

# Step 4: Validate Blockchain
blockchain.is_chain_valid()

# Step 5: Visualize Blockchain
visualize_blockchain(blockchain)
