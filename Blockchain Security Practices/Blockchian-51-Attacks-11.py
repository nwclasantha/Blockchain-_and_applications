import hashlib
import time
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import json

# === SIMULATION PARAMETERS ===
NUM_MINERS = 10        # Total miners in the network
ATTACKER_PERCENTAGE = 0.55  # 55% computational power owned by attacker (simulating 51%+ control)
DIFFICULTY = 4         # Proof-of-Work difficulty level
BLOCK_REWARD = 6.25    # Standard reward per mined block
SIMULATION_ROUNDS = 20 # Number of mining rounds

# === TRANSACTION CLASS ===
class Transaction:
    def __init__(self, sender, receiver, amount, tx_id=None):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.timestamp = time.time()
        self.tx_id = tx_id if tx_id else hashlib.sha256(f"{sender}{receiver}{amount}{self.timestamp}".encode()).hexdigest()[:10]

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

# === MINER CLASS ===
class Miner:
    def __init__(self, miner_id, is_attacker=False):
        self.miner_id = miner_id
        self.is_attacker = is_attacker
        self.computational_power = random.uniform(0.5, 2.0)  # Random computational power
        if is_attacker:
            self.computational_power *= 3  # Attackers have more power
        self.valid_blocks_mined = 0
        self.rewards = 0

    def mine_block(self):
        """Simulate Proof-of-Work mining."""
        nonce = 0
        while True:
            hash_attempt = hashlib.sha256(f"{self.miner_id}{nonce}".encode()).hexdigest()
            if hash_attempt[:DIFFICULTY] == "0" * DIFFICULTY:
                return nonce, hash_attempt  # Successfully mined block
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
        block_data = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "transactions": [str(tx) for tx in self.transactions],
            "nonce": self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_data.encode()).hexdigest()

# === BLOCKCHAIN CLASS ===
class Blockchain:
    def __init__(self, num_miners, attacker_percentage):
        self.miners = []
        self.attacker_miners = []
        self.honest_miners = []
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []
        self.total_hash_power = 0

        for i in range(num_miners):
            is_attacker = (random.random() < attacker_percentage)
            miner = Miner(i, is_attacker)
            self.miners.append(miner)
            if is_attacker:
                self.attacker_miners.append(miner)
            else:
                self.honest_miners.append(miner)

            self.total_hash_power += miner.computational_power

    def create_genesis_block(self):
        print("\n🌱 Creating Genesis Block...")
        return Block(0, "0", [], "Genesis")

    def add_transaction(self, sender, receiver, amount):
        transaction = Transaction(sender, receiver, amount)
        self.pending_transactions.append(transaction)
        print(f"\n📨 New Transaction: {sender} ➡ {receiver} | Amount: {amount}")

    def simulate_mining_round(self):
        """Simulate a mining round where miners compete to add a block."""
        winner = max(self.miners, key=lambda m: m.computational_power * random.random())  # Select winner based on power
        nonce, hash_value = winner.mine_block()

        # Attackers may manipulate the network
        if winner.is_attacker:
            if random.random() < 0.7:  # 70% chance attacker manipulates network
                print(f"🚨 51% Attack! Malicious Miner {winner.miner_id} controls block creation!")
                self.double_spending_attack()
                self.block_transactions()
                return None

        # Honest miner adds a block
        new_block = Block(len(self.chain), self.chain[-1].hash, self.pending_transactions, winner.miner_id, nonce)
        self.chain.append(new_block)
        self.pending_transactions = []
        winner.valid_blocks_mined += 1
        winner.rewards += BLOCK_REWARD

        return winner  # Return the honest miner who successfully mined a block

    def double_spending_attack(self):
        """Simulates a double-spending attack where the attacker reverses transactions."""
        if len(self.chain) > 1:
            target_block = random.choice(self.chain[1:])
            target_block.transactions = []  # Erase transactions (rollback)
            print(f"⚠️ Double Spending Attack! Transactions in Block {target_block.index} reversed!")

    def block_transactions(self):
        """Prevents certain users from performing transactions."""
        blocked_user = f"User_{random.randint(1, 100)}"
        self.pending_transactions = [tx for tx in self.pending_transactions if tx.sender != blocked_user]
        print(f"⚠️ 51% Attack! Transactions from {blocked_user} blocked!")

    def run_simulation(self, rounds):
        """Run multiple mining rounds with possible attacks."""
        print("\n🔍 Starting 51% Attack Simulation...\n")
        for round_num in range(1, rounds + 1):
            print(f"\n🚀 Mining Round {round_num}...\n")
            winner = self.simulate_mining_round()
            if winner:
                print(f"✅ Block Successfully Added by Miner {winner.miner_id}!")

    def visualize_results(self):
        """Graphically display mining pool performance."""
        miner_ids = [miner.miner_id for miner in self.miners]
        rewards = [miner.rewards for miner in self.miners]

        # Visualizing Miner Rewards
        plt.figure(figsize=(12, 6))
        plt.bar(miner_ids, rewards, color=["red" if miner.is_attacker else "green" for miner in self.miners])
        plt.xlabel("Miner ID")
        plt.ylabel("Total Rewards Earned")
        plt.title("Mining Rewards (Honest vs. Attackers)")
        plt.show()

# === RUN 51% ATTACK SIMULATION ===
blockchain = Blockchain(NUM_MINERS, ATTACKER_PERCENTAGE)
blockchain.add_transaction("Alice", "Bob", 50)
blockchain.add_transaction("Charlie", "Dave", 30)
blockchain.run_simulation(SIMULATION_ROUNDS)
blockchain.visualize_results()
