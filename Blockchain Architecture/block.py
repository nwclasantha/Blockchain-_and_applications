import hashlib
import json
import time
from merkle_tree import MerkleTree

class Block:
    def __init__(self, transactions, previous_hash, block_number):
        self.block_number = block_number
        self.timestamp = time.time()
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.nonce = 0
        self.merkle_root = MerkleTree(transactions).root
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = json.dumps({
            'block_number': self.block_number,
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'nonce': self.nonce
            }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def mine_block(self, difficulty=2):
        prefix = '0' * difficulty
        while not self.hash.startswith(prefix):
            self.nonce += 1
            self.hash = self.compute_hash()
