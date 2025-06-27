import hashlib
import json

class Transaction:
    def __init__(self, data):
        self.data = data

    def to_dict(self):
        return {'data': self.data}

class MerkleTree:
    def __init__(self, transactions):
        self.transactions = transactions
        self.root = self.build_merkle_root()

    def build_merkle_root(self):
        hashes = [self.hash_transaction(tx) for tx in self.transactions]

        while len(hashes) > 1:
            if len(hashes) % 2 != 0:
                hashes.append(hashes[-1])
            new_level = []
            for i in range(0, len(hashes), 2):
                new_level.append(self.hash_pair(hashes[i], hashes[i+1]))
            hashes = new_level

        return hashes[0] if hashes else None

    @staticmethod
    def hash_transaction(transaction):
        tx_string = json.dumps(transaction.to_dict(), sort_keys=True)
        return hashlib.sha256(tx_string.encode()).hexdigest()

    @staticmethod
    def hash_pair(a, b):
        return hashlib.sha256((a + b).encode()).hexdigest()

if __name__ == "__main__":
    transactions = [Transaction(f"Tx {i}") for i in range(1, 5)]
    tree = MerkleTree(transactions)
    print("Merkle Root:", tree.root)
