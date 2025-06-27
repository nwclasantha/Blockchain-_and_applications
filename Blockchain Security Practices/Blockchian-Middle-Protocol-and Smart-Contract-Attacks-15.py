import hashlib
import time
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from web3 import Web3

# === SIMULATION PARAMETERS ===
NUM_NODES = 15          # Total nodes in the blockchain network
ATTACK_PROBABILITY = 0.1 # Probability of a smart contract or network attack
SIMULATION_ROUNDS = 10   # Number of attack rounds
DDoS_IMPACT = 0.1        # Percentage of network affected by DDoS attack

# === SMART CONTRACT CLASS ===
class SmartContract:
    def __init__(self, contract_id):
        self.contract_id = contract_id
        self.balance = random.uniform(10, 100)  # Smart contract balance
        self.vulnerable = random.random() < 0.5  # 50% chance of having a vulnerability

    def withdraw(self, amount, attacker):
        """Simulates a reentrancy attack on a vulnerable smart contract."""
        if self.vulnerable:
            print(f"⚠️ Smart Contract {self.contract_id} is vulnerable! Reentrancy attack initiated by {attacker}.")
            while self.balance > 0 and amount <= self.balance:
                print(f"💰 Attacker withdrawing {amount} from Smart Contract {self.contract_id}.")
                self.balance -= amount  # Drain contract funds
        else:
            print(f"✅ Smart Contract {self.contract_id} is secure. No attack possible.")

    def __str__(self):
        return f"SmartContract {self.contract_id}: Balance={self.balance:.2f}, Vulnerable={self.vulnerable}"

# === BLOCKCHAIN NODE CLASS ===
class Node:
    def __init__(self, node_id, is_attacker=False):
        self.node_id = node_id
        self.is_attacker = is_attacker
        self.smart_contracts = [SmartContract(i) for i in range(random.randint(1, 3))]  # Assign contracts

    def execute_smart_contract(self):
        """Randomly executes a smart contract transaction."""
        if not self.smart_contracts:
            return
        contract = random.choice(self.smart_contracts)
        print(f"📝 Node {self.node_id} interacts with Smart Contract {contract.contract_id}.")

    def launch_attack(self, attack_type):
        """Simulate different smart contract and network attacks."""
        if attack_type == "Reentrancy":
            vulnerable_contract = next((sc for sc in self.smart_contracts if sc.vulnerable), None)
            if vulnerable_contract:
                vulnerable_contract.withdraw(random.uniform(1, 5), self.node_id)
            else:
                print(f"❌ No vulnerable smart contract found for Node {self.node_id}.")

        elif attack_type == "DDoS":
            print(f"🚨 Node {self.node_id} launches a DDoS Attack! {DDoS_IMPACT * 100:.0f}% of the network is affected.")

        elif attack_type == "Sybil":
            fake_nodes = [f"FakeNode_{i}" for i in range(random.randint(3, 10))]
            print(f"⚠️ Node {self.node_id} launches a Sybil Attack! Created {len(fake_nodes)} fake identities.")

        elif attack_type == "DAO Exploit":
            print(f"💸 Node {self.node_id} attempts a DAO Exploit! Redirecting funds to attacker.")

    def __str__(self):
        return f"Node {self.node_id} - {'Attacker' if self.is_attacker else 'Honest'}"

# === BLOCKCHAIN NETWORK CLASS ===
class BlockchainNetwork:
    def __init__(self, num_nodes, attack_probability):
        self.nodes = []
        self.attacker_nodes = []
        self.honest_nodes = []

        for i in range(num_nodes):
            is_attacker = random.random() < attack_probability
            node = Node(i, is_attacker)
            self.nodes.append(node)
            if is_attacker:
                self.attacker_nodes.append(node)
            else:
                self.honest_nodes.append(node)

    def simulate_round(self):
        """Simulates a round where nodes interact with smart contracts and some launch attacks."""
        for node in self.nodes:
            if node.is_attacker:
                attack_type = random.choice(["Reentrancy", "DDoS", "Sybil", "DAO Exploit"])
                node.launch_attack(attack_type)
            else:
                node.execute_smart_contract()

    def run_simulation(self, rounds):
        """Runs multiple rounds of attack simulation."""
        print("\n🔍 Starting Smart Contract and Network Attack Simulation...\n")
        for round_num in range(1, rounds + 1):
            print(f"\n🚀 Attack Round {round_num}...\n")
            self.simulate_round()

    def visualize_results(self):
        """Graphically display attack impact on the blockchain."""
        node_ids = [node.node_id for node in self.nodes]
        attacker_flags = [1 if node.is_attacker else 0 for node in self.nodes]
        vulnerabilities = [sum(1 for sc in node.smart_contracts if sc.vulnerable) for node in self.nodes]

        # Smart Contract Vulnerabilities
        plt.figure(figsize=(12, 6))
        plt.bar(node_ids, vulnerabilities, color="red")
        plt.xlabel("Node ID")
        plt.ylabel("Number of Vulnerable Smart Contracts")
        plt.title("Smart Contract Vulnerabilities Across Nodes")
        plt.show()

        # Attacker Distribution
        plt.figure(figsize=(12, 6))
        plt.bar(node_ids, attacker_flags, color=["red" if node.is_attacker else "green" for node in self.nodes])
        plt.xlabel("Node ID")
        plt.ylabel("Attacker Status (1 = Attacker, 0 = Honest)")
        plt.title("Blockchain Network - Honest vs. Attacker Nodes")
        plt.show()

        # Network Graph Representation
        G = nx.DiGraph()
        for node in self.nodes:
            node_label = f"Node {node.node_id}\nAttacker: {node.is_attacker}"
            G.add_node(node.node_id, label=node_label)

        for node in self.nodes:
            if node.is_attacker:
                for _ in range(random.randint(1, 3)):  # Simulate attack connections
                    target = random.choice(self.nodes)
                    if target != node:
                        G.add_edge(node.node_id, target.node_id)

        pos = nx.spring_layout(G, seed=42)
        labels = {n: G.nodes[n]['label'] for n in G.nodes}

        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", edge_color="black", font_size=10, font_weight="bold")
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
        plt.title("Blockchain Network - Attack Simulation")
        plt.show()

# === RUN SMART CONTRACT ATTACK SIMULATION ===
network = BlockchainNetwork(NUM_NODES, ATTACK_PROBABILITY)
network.run_simulation(SIMULATION_ROUNDS)
network.visualize_results()
