def simulate_attack(miner_hashrate, network_hashrate):
    if miner_hashrate / network_hashrate > 0.5:
        print("🚨 51% Attack Possible!")
        print("Actions: Double spend, reverse transactions, block validation")
    else:
        print("✅ Network is secure against 51% attack")

if __name__ == "__main__":
    simulate_attack(miner_hashrate=600, network_hashrate=1000)  # 60% control
    simulate_attack(miner_hashrate=400, network_hashrate=1000)  # 40% control
