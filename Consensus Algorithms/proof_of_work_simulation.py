import hashlib
import time
import random

class Miner:
    def __init__(self, name):
        self.name = name

    def mine_block(self, data, difficulty=4):
        nonce = 0
        prefix = '0' * difficulty
        start_time = time.time()

        while True:
            block_content = f"{data}{nonce}"
            block_hash = hashlib.sha256(block_content.encode()).hexdigest()

            if block_hash.startswith(prefix):
                duration = time.time() - start_time
                return nonce, block_hash, duration
            nonce += 1

if __name__ == "__main__":
    miners = [Miner("Alice"), Miner("Bob"), Miner("Charlie")]

    data = "Transaction Data"

    print("\nStarting mining competition...\n")
    results = []

    for miner in miners:
        nonce, block_hash, duration = miner.mine_block(data)
        results.append((duration, miner.name, nonce, block_hash))

    results.sort()
    winner = results[0]

    print(f"🏆 Winner: {winner[1]}")
    print(f"Nonce: {winner[2]}")
    print(f"Hash: {winner[3]}")
    print(f"Time Taken: {winner[0]:.2f} seconds")
