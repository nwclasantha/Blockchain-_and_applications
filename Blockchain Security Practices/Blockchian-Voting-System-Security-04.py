import hashlib
import time
import json
import random
import matplotlib.pyplot as plt
import networkx as nx

# === VOTER CLASS (Registration & Verification) ===
class Voter:
    def __init__(self, voter_id, name):
        self.voter_id = voter_id
        self.name = name
        self.has_voted = False

# === MERKLE TREE CLASS (For Vote Integrity) ===
class MerkleTree:
    def __init__(self, votes):
        self.votes = votes
        self.root = self.build_merkle_tree(votes)

    def build_merkle_tree(self, votes):
        if not votes:
            return hashlib.sha256("NO_VOTES".encode()).hexdigest()
        hashes = [self.hash_vote(vote) for vote in votes]
        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])
            hashes = [self.hash_pair(hashes[i], hashes[i+1]) for i in range(0, len(hashes), 2)]
        return hashes[0]

    def hash_vote(self, vote):
        return hashlib.sha256(vote.encode()).hexdigest()

    def hash_pair(self, hash1, hash2):
        return hashlib.sha256((hash1 + hash2).encode()).hexdigest()

# === VOTE CLASS (Cast Votes Anonymously) ===
class Vote:
    def __init__(self, voter_id, candidate):
        self.voter_id = voter_id
        self.candidate = candidate
        self.timestamp = time.time()

    def to_dict(self):
        return {
            "voter_id": hashlib.sha256(self.voter_id.encode()).hexdigest(),  # Encrypt voter ID
            "candidate": self.candidate,
            "timestamp": self.timestamp
        }

    def __str__(self):
        return json.dumps(self.to_dict(), sort_keys=True)

# === BLOCK CLASS (Immutable Storage of Votes) ===
class Block:
    def __init__(self, index, previous_hash, votes, nonce=0):
        self.index = index
        self.timestamp = time.time()
        self.previous_hash = previous_hash
        self.votes = votes
        self.merkle_tree = MerkleTree([str(vote) for vote in votes])
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

# === BLOCKCHAIN CLASS (Tamper-Proof Voting Storage) ===
class Blockchain:
    def __init__(self, difficulty=4):
        self.chain = [self.create_genesis_block()]
        self.voters = {}  # Store voter information
        self.difficulty = difficulty

    def create_genesis_block(self):
        print("\n🌱 Creating Genesis Block...")
        return Block(0, "0", [])

    def register_voter(self, voter_id, name):
        if voter_id in self.voters:
            print(f"⚠️ Voter {name} is already registered.")
            return False
        self.voters[voter_id] = Voter(voter_id, name)
        print(f"✅ Voter {name} registered successfully.")
        return True

    def cast_vote(self, voter_id, candidate):
        if voter_id not in self.voters:
            print("❌ Unregistered voter!")
            return False
        if self.voters[voter_id].has_voted:
            print("❌ Voter has already voted!")
            return False

        vote = Vote(voter_id, candidate)
        self.voters[voter_id].has_voted = True
        print(f"🗳️ Vote Cast: {voter_id} voted for {candidate}")
        return vote

    def add_block(self, votes):
        previous_block = self.chain[-1]
        new_block = Block(len(self.chain), previous_block.hash, votes)
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)

    def count_votes(self):
        tally = {}
        for block in self.chain[1:]:
            for vote in block.votes:
                vote_data = json.loads(str(vote))
                candidate = vote_data["candidate"]
                tally[candidate] = tally.get(candidate, 0) + 1
        return tally

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
        label = f"Block {block.index}\nHash: {block.hash[:6]}...\nVotes: {len(block.votes)}"
        G.add_node(block.index, label=label)
        if block.index > 0:
            G.add_edge(block.index - 1, block.index)

    pos = nx.spring_layout(G, seed=42)
    labels = {n: G.nodes[n]['label'] for n in G.nodes}

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", edge_color="black", font_size=8, font_weight="bold")
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    plt.title("Blockchain Voting System")
    plt.show()

def visualize_vote_tally(vote_counts):
    candidates = list(vote_counts.keys())
    votes = list(vote_counts.values())

    plt.figure(figsize=(10, 5))
    plt.bar(candidates, votes, color="green")
    plt.xlabel("Candidates")
    plt.ylabel("Number of Votes")
    plt.title("Election Results")
    plt.show()

# === RUN BLOCKCHAIN VOTING SYSTEM ===
blockchain = Blockchain()

# Register voters
blockchain.register_voter("VOTER001", "Alice")
blockchain.register_voter("VOTER002", "Bob")
blockchain.register_voter("VOTER003", "Charlie")

# Cast votes
votes = []
votes.append(blockchain.cast_vote("VOTER001", "Candidate A"))
votes.append(blockchain.cast_vote("VOTER002", "Candidate B"))
votes.append(blockchain.cast_vote("VOTER003", "Candidate A"))

# Add votes to blockchain
blockchain.add_block([v for v in votes if v])

# Validate Blockchain
blockchain.is_chain_valid()

# Count votes
vote_counts = blockchain.count_votes()
print("\n📊 Final Election Results:", vote_counts)

# Visualize Blockchain & Results
visualize_blockchain(blockchain)
visualize_vote_tally(vote_counts)
