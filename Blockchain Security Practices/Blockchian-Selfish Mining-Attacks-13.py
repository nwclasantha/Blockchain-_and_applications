import hashlib
import time
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# === SIMULATION PARAMETERS ===
NUM_MINERS = 20          # Total miners in the network
SELFISH_PERCENTAGE = 0.3 # Percentage of selfish miners
DIFFICULTY = 4           # Proof-of-Work difficulty level
BLOCK_REWARD = 6.25      # Standard mining reward per block
SIMULATION_ROUNDS = 20   # Number of mining rounds
BRIBE_CHANCE = 0.5       # 50% chance that honest miners switch to selfish chain

# === TRANSACTION CLASS ===
class Transaction:
    def __init__(self, sender, receiver, amount):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.timestamp = time.time()
        self.tx_id = hashlib.sha256(f"{sender}{receiver}{amount}{self.timestamp}".encode()).hexdigest()[:10]

    def __str__(self):
        return f"{self.sender} ➡ {self.receiver} | {self.amount} BTC"

# === MINER CLASS ===
class Miner:
    def __init__(self, miner_id, is_selfish=False):
        self.miner_id = miner_id
        self.is_selfish = is_selfish
        self.computational_power = random.uniform(0.5, 2.0)  # Random power
        self.valid_blocks_mined = 0
        self.rewards = 0
        self.secret_chain = [] if is_selfish else None  # Selfish miners maintain a private chain

    def mine_block(self):
        """Simulate Proof-of-Work mining."""
        nonce = 0
        while True:
            hash_attempt = hashlib.sha256(f"{self.miner_id}{nonce}".encode()).hexdigest()
            if hash_attempt[:DIFFICULTY] == "0" * DIFFICULTY:
                return nonce, hash_attempt
            nonce += 1

# === BLOCK CLASS ===
class Block:
    def __init__(self, index, previous_hash, transactions, mined_by, nonce=0):
        self.index = index
        self.timestamp = time.time()
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.mined_by = mined_by
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_data = hashlib.sha256(f"{self.index}{self.previous_hash}{self.nonce}".encode()).hexdigest()
        return block_data

# === BLOCKCHAIN CLASS ===
class Blockchain:
    def __init__(self, num_miners, selfish_percentage):
        self.miners = []
        self.selfish_miners = []
        self.honest_miners = []
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []

        for i in range(num_miners):
            is_selfish = (random.random() < selfish_percentage)
            miner = Miner(i, is_selfish)
            self.miners.append(miner)
            if is_selfish:
                self.selfish_miners.append(miner)
            else:
                self.honest_miners.append(miner)

    def create_genesis_block(self):
        print("\n🌱 Creating Genesis Block...")
        return Block(0, "0", [], "Genesis")

    def add_transaction(self, sender, receiver, amount):
        transaction = Transaction(sender, receiver, amount)
        self.pending_transactions.append(transaction)
        print(f"\n📨 New Transaction: {sender} ➡ {receiver} | Amount: {amount}")

    def simulate_mining_round(self):
        """Simulate a mining round where both honest and selfish miners compete."""
        # Selfish miners mine in secret
        for selfish_miner in self.selfish_miners:
            if random.random() < 0.7:  # 70% chance selfish miners mine a secret block
                nonce, hash_value = selfish_miner.mine_block()
                selfish_miner.secret_chain.append(Block(len(self.chain) + len(self.selfish_miners), self.chain[-1].hash, [], selfish_miner.miner_id, nonce))
                print(f"⚠️ Selfish Miner {selfish_miner.miner_id} mined a SECRET block!")

        # Honest miners mine publicly
        winner = max(self.honest_miners, key=lambda m: m.computational_power * random.random())  
        nonce, hash_value = winner.mine_block()
        new_block = Block(len(self.chain), self.chain[-1].hash, self.pending_transactions, winner.miner_id, nonce)

        # Compare chains: Does the selfish chain overtake the public chain?
        if any(len(miner.secret_chain) > len(self.chain) for miner in self.selfish_miners):
            if random.random() < BRIBE_CHANCE:  # 50% chance that miners switch to selfish chain
                self.chain = max(self.selfish_miners, key=lambda m: len(m.secret_chain)).secret_chain  # Adopt the longest chain
                print(f"💰 Bribery Success! Honest miners switched to Selfish Chain!")
                return

        # Otherwise, add the honest block
        self.chain.append(new_block)
        self.pending_transactions = []
        winner.valid_blocks_mined += 1
        winner.rewards += BLOCK_REWARD

        print(f"✅ Block Successfully Added by Honest Miner {winner.miner_id}")

    def run_simulation(self, rounds):
        """Run multiple mining rounds with possible attacks."""
        print("\n🔍 Starting Selfish Mining Attack Simulation...\n")
        for round_num in range(1, rounds + 1):
            print(f"\n🚀 Mining Round {round_num}...\n")
            self.simulate_mining_round()

    def visualize_results(self):
        """Graphically display mining performance."""
        miner_ids = [miner.miner_id for miner in self.miners]
        rewards = [miner.rewards for miner in self.miners]

        # Rewards Earned by Honest vs. Selfish Miners
        plt.figure(figsize=(12, 6))
        plt.bar(miner_ids, rewards, color=["red" if miner.is_selfish else "green" for miner in self.miners])
        plt.xlabel("Miner ID")
        plt.ylabel("Total Rewards Earned")
        plt.title("Mining Rewards (Honest vs. Selfish Miners)")
        plt.show()

        # Network Graph Representation
        G = nx.DiGraph()
        for miner in self.miners:
            G.add_node(f"Miner {miner.miner_id}", label=f"Miner {miner.miner_id}\nRewards: {miner.rewards:.2f}")
            if miner.valid_blocks_mined > 0:
                G.add_edge(f"Miner {miner.miner_id}", "Blockchain")

        pos = nx.spring_layout(G, seed=42)

        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", edge_color="black", font_size=10, font_weight="bold")
        plt.title("Blockchain - Honest vs. Selfish Miners")
        plt.show()

# === RUN SELFISH MINING ATTACK SIMULATION ===
blockchain = Blockchain(NUM_MINERS, SELFISH_PERCENTAGE)
blockchain.add_transaction("Alice", "Bob", 50)
blockchain.add_transaction("Charlie", "Dave", 30)
blockchain.run_simulation(SIMULATION_ROUNDS)
blockchain.visualize_results()
