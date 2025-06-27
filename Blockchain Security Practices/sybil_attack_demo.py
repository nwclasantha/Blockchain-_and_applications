def simulate_sybil_attack(real_nodes, fake_nodes):
    total_nodes = real_nodes + fake_nodes
    if fake_nodes / total_nodes > 0.5:
        print("❌ Sybil Attack: Attacker controls majority of identities")
    else:
        print("✅ Network safe: Honest majority maintained")

if __name__ == "__main__":
    simulate_sybil_attack(real_nodes=5, fake_nodes=7)
    simulate_sybil_attack(real_nodes=10, fake_nodes=3)
