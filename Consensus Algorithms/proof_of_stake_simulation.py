import random
import matplotlib.pyplot as plt

class Validator:
    def __init__(self, name, stake):
        self.name = name
        self.stake = stake

def visualize_validators(validators, selected_name=None):
    names = [v.name for v in validators]
    stakes = [v.stake for v in validators]
    colors = ['green' if name == selected_name else 'blue' for name in names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, stakes, color=colors)

    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 1, f'{int(height)}', ha='center', va='bottom')

    plt.title("Validator Stakes")
    if selected_name:
        plt.suptitle(f"🏆 Selected Validator: {selected_name}", fontsize=12, y=0.95, color='darkred')
    plt.xlabel("Validators")
    plt.ylabel("Stake")
    plt.ylim(0, max(stakes) + 10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    validators = [
        Validator("Alice", 50),
        Validator("Bob", 30),
        Validator("Charlie", 20)
    ]

    # Create weighted pool
    weighted_pool = []
    for validator in validators:
        weighted_pool.extend([validator.name] * validator.stake)

    print("\nSelecting validator based on stake...\n")
    selected = random.choice(weighted_pool)

    print(f"🏆 Selected Validator: {selected}")

    # Visualize the result
    visualize_validators(validators, selected_name=selected)
