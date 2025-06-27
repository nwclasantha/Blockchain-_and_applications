import hashlib
import rsa

# Simulating Blockchain Security Features

# Step 1: Hashing (Data Integrity)
def hash_transaction(tx_data):
    return hashlib.sha256(tx_data.encode()).hexdigest()

# Step 2: Public/Private Keys (Authentication)
(pub_key, priv_key) = rsa.newkeys(512)

# Step 3: Digital Signature (Non-repudiation)
def sign_data(private_key, data):
    return rsa.sign(data.encode(), private_key, 'SHA-256')

def verify_signature(public_key, data, signature):
    try:
        rsa.verify(data.encode(), signature, public_key)
        return True
    except rsa.VerificationError:
        return False

if __name__ == "__main__":
    transaction = "Alice pays Bob 5 BTC"
    print("\nOriginal Transaction:", transaction)

    tx_hash = hash_transaction(transaction)
    print("Hashed Transaction:", tx_hash)

    signature = sign_data(priv_key, transaction)
    print("Signature:", signature.hex())

    result = verify_signature(pub_key, transaction, signature)
    print("Signature Verified?" , result)
