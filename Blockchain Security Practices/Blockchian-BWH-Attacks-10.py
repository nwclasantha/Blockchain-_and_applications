import hashlib
import time
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# === SIMULATION PARAMETERS ===
NUM_MINERS = 10        # Total number of miners
BWH_PERCENTAGE = 0.6   # Percentage of miners performing Block Withholding
DIFFICULTY = 5         # Proof-of-Work difficulty
BLOCK_REWARD = 6.25    # Standard mining reward per block
SIMULATION_ROUNDS = 20 # Number of mining rounds

# === MINER CLASS ===
class Miner:
    def __init__(self, miner_id, is_malicious=False):
        self.miner_id = miner_id
        self.is_malicious = is_malicious
        self.computational_power = random.uniform(0.5, 2.0)  # Random computational power
        self.valid_blocks_mined = 0
        self.fake_blocks_submitted = 0
        self.rewards = 0

    def mine_block(self):
        """Simulate Proof-of-Work mining."""
        nonce = 0
        while True:
            hash_attempt = hashlib.sha256(f"{self.miner_id}{nonce}".encode()).hexdigest()
            if hash_attempt[:DIFFICULTY] == "0" * DIFFICULTY:
                return nonce, hash_attempt  # Successfully mined block
            nonce += 1

    def submit_partial_proof(self):
        """Malicious miner submits a fake Proof-of-Work to mislead the pool."""
        return hashlib.sha256(f"{self.miner_id}{random.randint(1, 100000)}".encode()).hexdigest()

# === MINING POOL CLASS ===
class MiningPool:
    def __init__(self, num_miners, bwh_percentage):
        self.miners = []
        self.malicious_miners = []
        self.honest_miners = []
        self.global_chain = []
        self.pool_rewards = 0

        for i in range(num_miners):
            is_malicious = random.random() < bwh_percentage
            miner = Miner(i, is_malicious)
            self.miners.append(miner)
            if is_malicious:
                self.malicious_miners.append(miner)
            else:
                self.honest_miners.append(miner)

    def simulate_mining_round(self):
        """Simulate a full mining round in the pool."""
        winner = max(self.miners, key=lambda m: m.computational_power * random.random())  # Select winner based on power
        nonce, hash_value = winner.mine_block()

        # If the winner is malicious, they may withhold the block
        if winner.is_malicious:
            if random.random() < 0.7:  # 70% chance of withholding the block
                print(f"⚠️ Malicious Miner {winner.miner_id} performed BWH Attack: Block Withheld!")
                winner.fake_blocks_submitted += 1
                return None  # No valid block added to blockchain
            else:
                print(f"⚠️ Malicious Miner {winner.miner_id} decided to submit the block!")

        # Add the block to the blockchain
        self.global_chain.append(hash_value)
        winner.valid_blocks_mined += 1
        self.pool_rewards += BLOCK_REWARD

        return winner  # Return miner who successfully added the block

    def distribute_rewards(self):
        """Distribute mining rewards among all participating miners."""
        if self.pool_rewards == 0:
            return

        total_power = sum(m.computational_power for m in self.miners)
        for miner in self.miners:
            miner.rewards += (miner.computational_power / total_power) * self.pool_rewards

        self.pool_rewards = 0  # Reset reward pool

    def run_simulation(self, rounds):
        """Run multiple rounds of the mining process."""
        print("\n🔍 Starting BWH Attack Simulation on Mining Pool...\n")
        for round_num in range(1, rounds + 1):
            print(f"\n🚀 Mining Round {round_num}...\n")
            winner = self.simulate_mining_round()
            if winner:
                print(f"✅ Block Successfully Added by Miner {winner.miner_id}!")
            self.distribute_rewards()

    def visualize_results(self):
        """Graphically display mining pool performance."""
        miner_ids = [miner.miner_id for miner in self.miners]
        honest_blocks = [miner.valid_blocks_mined for miner in self.miners]
        fake_submissions = [miner.fake_blocks_submitted for miner in self.miners]
        rewards = [miner.rewards for miner in self.miners]
    
        # Visualizing Valid Blocks vs. Withheld Blocks
        plt.figure(figsize=(12, 6))
        bar_width = 0.4
        plt.bar(np.array(miner_ids) - bar_width/2, honest_blocks, width=bar_width, label="Valid Blocks Mined", color="blue")
        plt.bar(np.array(miner_ids) + bar_width/2, fake_submissions, width=bar_width, label="Fake PoW Submitted", color="red")
        plt.xlabel("Miner ID")
        plt.ylabel("Number of Blocks")
        plt.title("Valid vs. Withheld Blocks in BWH Attack")
        plt.legend()
        plt.show()
    
        # Visualizing Miner Rewards
        plt.figure(figsize=(12, 6))
        plt.bar(miner_ids, rewards, color="green")
        plt.xlabel("Miner ID")
        plt.ylabel("Total Rewards Earned")
        plt.title("Mining Pool Rewards Distribution")
        plt.show()
    
        # Network Graph Representation
        G = nx.DiGraph()
        
        node_labels = {}
        
        for miner in self.miners:
            node_name = f"Miner {miner.miner_id}"
            label_text = f"Miner {miner.miner_id}\nRewards: {miner.rewards:.2f}"
            
            G.add_node(node_name)
            node_labels[node_name] = label_text  # Store labels manually
            
            if miner.valid_blocks_mined > 0:
                G.add_edge(node_name, "Blockchain", weight=miner.valid_blocks_mined)
    
        pos = nx.spring_layout(G, seed=42)
    
        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", edge_color="black", font_size=10, font_weight="bold")
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        plt.title("Mining Pool - Honest vs. Malicious Miners in BWH Attack")
        plt.show()

# === RUN BWH ATTACK SIMULATION ===
mining_pool = MiningPool(NUM_MINERS, BWH_PERCENTAGE)
mining_pool.run_simulation(SIMULATION_ROUNDS)
mining_pool.visualize_results()
