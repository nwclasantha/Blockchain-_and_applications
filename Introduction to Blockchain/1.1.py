import hashlib
import time
import json

class Transaction:
    def __init__(self, sender, receiver, amount):
        try:
            if not isinstance(sender, str) or not isinstance(receiver, str):
                raise ValueError("Sender and receiver must be strings.")
            if not isinstance(amount, (int, float)) or amount <= 0:
                raise ValueError("Amount must be a positive number.")
            
            self.sender = sender
            self.receiver = receiver
            self.amount = amount
            self.timestamp = time.time()
        except Exception as e:
            print(f"[Transaction Error] {e}")
            raise

    def to_dict(self):
        return {
            'sender': self.sender,
            'receiver': self.receiver,
            'amount': self.amount,
            'timestamp': self.timestamp
        }

class Block:
    def __init__(self, transactions, previous_hash):
        try:
            if not isinstance(transactions, list):
                raise ValueError("Transactions must be a list.")
            if not all(isinstance(tx, Transaction) for tx in transactions):
                raise ValueError("Each transaction must be a Transaction object.")
            if not isinstance(previous_hash, str):
                raise ValueError("Previous hash must be a string.")
            
            self.timestamp = time.time()
            self.transactions = transactions
            self.previous_hash = previous_hash
            self.nonce = 0
            self.hash = self.compute_hash()
        except Exception as e:
            print(f"[Block Initialization Error] {e}")
            raise

    def compute_hash(self):
        try:
            block_string = json.dumps({
                'timestamp': self.timestamp,
                'transactions': [tx.to_dict() for tx in self.transactions],
                'previous_hash': self.previous_hash,
                'nonce': self.nonce
            }, sort_keys=True).encode()
            return hashlib.sha256(block_string).hexdigest()
        except Exception as e:
            print(f"[Compute Hash Error] {e}")
            raise

    def mine_block(self, difficulty=2):
        try:
            prefix = '0' * difficulty
            while not self.hash.startswith(prefix):
                self.nonce += 1
                self.hash = self.compute_hash()
        except Exception as e:
            print(f"[Mining Error] {e}")
            raise

class Blockchain:
    def __init__(self):
        try:
            self.chain = [self.create_genesis_block()]
            self.pending_transactions = []
            self.difficulty = 2
        except Exception as e:
            print(f"[Blockchain Init Error] {e}")
            raise

    def create_genesis_block(self):
        try:
            genesis_block = Block([], "0")
            genesis_block.hash = genesis_block.compute_hash()
            return genesis_block
        except Exception as e:
            print(f"[Genesis Block Error] {e}")
            raise

    def add_transaction(self, transaction):
        try:
            if not isinstance(transaction, Transaction):
                raise TypeError("Only Transaction objects can be added.")
            self.pending_transactions.append(transaction)
        except Exception as e:
            print(f"[Add Transaction Error] {e}")
            raise

    def mine_pending_transactions(self, miner_address):
        try:
            if not self.pending_transactions:
                print("No transactions to mine.")
                return
            
            new_block = Block(self.pending_transactions, self.chain[-1].hash)
            new_block.mine_block(self.difficulty)
            self.chain.append(new_block)

            # Reward miner
            reward_tx = Transaction("Network", miner_address, 1)
            self.pending_transactions = [reward_tx]
        except Exception as e:
            print(f"[Mining Transactions Error] {e}")
            raise

    def is_chain_valid(self):
        try:
            for i in range(1, len(self.chain)):
                current = self.chain[i]
                previous = self.chain[i - 1]

                if current.hash != current.compute_hash():
                    print(f"Invalid block at index {i}: Hash mismatch.")
                    return False

                if current.previous_hash != previous.hash:
                    print(f"Invalid block at index {i}: Previous hash mismatch.")
                    return False
            return True
        except Exception as e:
            print(f"[Chain Validation Error] {e}")
            return False

# ----------------- DEMO -----------------
try:
    my_blockchain = Blockchain()

    my_blockchain.add_transaction(Transaction("Alice", "Bob", 50))
    my_blockchain.add_transaction(Transaction("Bob", "Charlie", 30))

    print("\n--- Mining Block 1 ---")
    my_blockchain.mine_pending_transactions("Miner1")

    my_blockchain.add_transaction(Transaction("Charlie", "Dave", 25))
    my_blockchain.add_transaction(Transaction("Dave", "Eve", 10))

    print("\n--- Mining Block 2 ---")
    my_blockchain.mine_pending_transactions("Miner2")

    print("\nBlockchain valid?", my_blockchain.is_chain_valid())

    for i, block in enumerate(my_blockchain.chain):
        print(f"\nBlock {i}:")
        print("Hash:", block.hash)
        print("Previous Hash:", block.previous_hash)
        print("Transactions:", [tx.to_dict() for tx in block.transactions])

except Exception as main_err:
    print(f"[Fatal Error] {main_err}")
