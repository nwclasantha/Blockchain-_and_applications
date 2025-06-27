import hashlib
import time
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# === SIMULATION PARAMETERS ===
NUM_MINERS = 30        # Total number of miners
NUM_POOLS = 6          # Number of mining pools
POOL_HOPPERS = 0.9     # Percentage of miners performing Pool Hopping Attack
DIFFICULTY = 5         # Proof-of-Work difficulty
BLOCK_REWARD = 6.25    # Standard mining reward per block
SIMULATION_ROUNDS = 20 # Number of mining rounds

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
    def __init__(self, miner_id, is_hopper=False):
        self.miner_id = miner_id
        self.is_hopper = is_hopper
        self.computational_power = random.uniform(0.5, 2.0)  # Random power
        self.valid_blocks_mined = 0
        self.rewards = 0
        self.current_pool = None  # Assigned dynamically

    def mine_block(self):
        """Simulate Proof-of-Work mining."""
        nonce = 0
        while True:
            hash_attempt = hashlib.sha256(f"{self.miner_id}{nonce}".encode()).hexdigest()
            if hash_attempt[:DIFFICULTY] == "0" * DIFFICULTY:
                return nonce, hash_attempt
            nonce += 1

# === MINING POOL CLASS ===
class MiningPool:
    def __init__(self, pool_id):
        self.pool_id = pool_id
        self.miners = []
        self.computational_power = 0
        self.total_rewards = 0

    def add_miner(self, miner):
        """Assign a miner to this pool."""
        self.miners.append(miner)
        miner.current_pool = self
        self.computational_power += miner.computational_power

    def remove_miner(self, miner):
        """Remove a miner from this pool when they hop."""
        if miner in self.miners:
            self.miners.remove(miner)
            self.computational_power -= miner.computational_power

    def mine_block(self):
        """Simulate a mining process in this pool."""
        if not self.miners:
            return None  # No miners available

        winner = max(self.miners, key=lambda m: m.computational_power * random.random())  # Select winner
        nonce, hash_value = winner.mine_block()
        winner.valid_blocks_mined += 1
        self.total_rewards += BLOCK_REWARD
        return winner

    def distribute_rewards(self):
        """Distribute rewards based on computational contribution."""
        if self.total_rewards == 0 or not self.miners:
            return
        
        total_power = sum(m.computational_power for m in self.miners)
        for miner in self.miners:
            miner.rewards += (miner.computational_power / total_power) * self.total_rewards
        
        self.total_rewards = 0  # Reset rewards

# === NETWORK CLASS (SIMULATING MULTIPLE MINING POOLS) ===
class BlockchainNetwork:
    def __init__(self, num_miners, num_pools, pool_hopper_percentage):
        self.pools = [MiningPool(i) for i in range(num_pools)]
        self.miners = []
        self.hoppers = []
        self.honest_miners = []

        for i in range(num_miners):
            is_hopper = (random.random() < pool_hopper_percentage)
            miner = Miner(i, is_hopper)
            self.miners.append(miner)
            if is_hopper:
                self.hoppers.append(miner)
            else:
                self.honest_miners.append(miner)

        self.assign_miners_to_pools()

    def assign_miners_to_pools(self):
        """Distribute miners evenly among pools initially."""
        for miner in self.miners:
            pool = random.choice(self.pools)
            pool.add_miner(miner)

    def perform_pool_hopping(self):
        """Malicious miners switch pools to maximize earnings."""
        for hopper in self.hoppers:
            # Find the pool with the highest computational power
            best_pool = max(self.pools, key=lambda p: p.computational_power)
            if hopper.current_pool != best_pool:
                # Leave old pool and join the best one
                hopper.current_pool.remove_miner(hopper)
                best_pool.add_miner(hopper)
                print(f"⚠️ Pool Hopper {hopper.miner_id} switched to Pool {best_pool.pool_id}")

    def simulate_mining_round(self):
        """Each pool runs a mining round, hoppers may switch pools."""
        self.perform_pool_hopping()
        for pool in self.pools:
            winner = pool.mine_block()
            if winner:
                print(f"✅ Block Mined in Pool {pool.pool_id} by Miner {winner.miner_id}")

        for pool in self.pools:
            pool.distribute_rewards()

    def run_simulation(self, rounds):
        """Run multiple rounds of mining."""
        print("\n🔍 Starting Pool Hopping Attack Simulation...\n")
        for round_num in range(1, rounds + 1):
            print(f"\n🚀 Mining Round {round_num}...\n")
            self.simulate_mining_round()

    def visualize_results(self):
        """Graphically display mining pool performance."""
        pool_ids = [pool.pool_id for pool in self.pools]
        computational_powers = [pool.computational_power for pool in self.pools]
        total_rewards = [sum(m.rewards for m in pool.miners) for pool in self.pools]

        # Computational Power of Each Pool
        plt.figure(figsize=(12, 6))
        plt.bar(pool_ids, computational_powers, color="blue")
        plt.xlabel("Pool ID")
        plt.ylabel("Computational Power")
        plt.title("Computational Power Distribution Across Pools")
        plt.show()

        # Rewards Earned by Each Pool
        plt.figure(figsize=(12, 6))
        plt.bar(pool_ids, total_rewards, color="green")
        plt.xlabel("Pool ID")
        plt.ylabel("Total Mining Rewards")
        plt.title("Mining Rewards Across Pools")
        plt.show()

        # Network Graph Representation
        G = nx.DiGraph()
        for pool in self.pools:
            G.add_node(f"Pool {pool.pool_id}", label=f"Pool {pool.pool_id}\nPower: {pool.computational_power:.2f}")

        for miner in self.miners:
            if miner.current_pool:
                G.add_node(f"Miner {miner.miner_id}", label=f"Miner {miner.miner_id}\nRewards: {miner.rewards:.2f}")
                G.add_edge(f"Miner {miner.miner_id}", f"Pool {miner.current_pool.pool_id}")

        pos = nx.spring_layout(G, seed=42)

        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", edge_color="black", font_size=10, font_weight="bold")
        plt.title("Mining Pool - Honest vs. Pool Hopping Miners")
        plt.show()

# === RUN POOL HOPPING ATTACK SIMULATION ===
network = BlockchainNetwork(NUM_MINERS, NUM_POOLS, POOL_HOPPERS)
network.run_simulation(SIMULATION_ROUNDS)
network.visualize_results()
