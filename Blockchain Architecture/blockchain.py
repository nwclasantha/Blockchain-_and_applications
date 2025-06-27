from block import Block
from transaction import Transaction

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []
        self.difficulty = 2

    def create_genesis_block(self):
        genesis_block = Block([], "0", 0)
        return genesis_block

    def add_transaction(self, transaction):
        self.pending_transactions.append(transaction)

    def mine_pending_transactions(self, miner_address):
        if not self.pending_transactions:
            print("No transactions to mine.")
            return

        block_number = len(self.chain)
        new_block = Block(self.pending_transactions, self.chain[-1].hash, block_number)
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)

        reward_tx = Transaction("Network", miner_address, 1)
        self.pending_transactions = [reward_tx]

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            if current.hash != current.compute_hash():
                return False
            if current.previous_hash != previous.hash:
                return False
        return True

    def print_chain(self):
        for block in self.chain:
            print(f"Block {block.block_number} | Hash: {block.hash} | Prev: {block.previous_hash}")
