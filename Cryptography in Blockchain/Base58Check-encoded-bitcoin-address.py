import hashlib
import base58
from ecdsa import SigningKey, SECP256k1
import binascii

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def ripemd160(data: bytes) -> bytes:
    h = hashlib.new('ripemd160')
    h.update(data)
    return h.digest()

def hash160(data: bytes) -> bytes:
    """SHA256 followed by RIPEMD160 (used in Bitcoin addresses)"""
    return ripemd160(sha256(data))

def double_sha256(data: bytes) -> bytes:
    return sha256(sha256(data))

def base58check_encode(prefix: bytes, payload: bytes) -> str:
    full_data = prefix + payload
    checksum = double_sha256(full_data)[:4]
    return base58.b58encode(full_data + checksum).decode()

def generate_btc_address(hex_private_key: str, compressed: bool = True, testnet: bool = False):
    print("🔐 Generating Bitcoin Address...")
    print(f"• Private Key (hex): {hex_private_key}")
    
    private_key_bytes = bytes.fromhex(hex_private_key)
    if len(private_key_bytes) != 32:
        raise ValueError("Private key must be 32 bytes (64 hex characters).")

    # Derive public key from private key
    sk = SigningKey.from_string(private_key_bytes, curve=SECP256k1)
    vk = sk.verifying_key

    if compressed:
        # Compressed public key format
        pk_bytes = vk.to_string()
        prefix = b'\x02' if pk_bytes[31] % 2 == 0 else b'\x03'
        public_key = prefix + pk_bytes[:32]
    else:
        # Uncompressed format (0x04 + X + Y)
        public_key = b'\x04' + vk.to_string()

    print(f"• Public Key ({'compressed' if compressed else 'uncompressed'}): {public_key.hex()}")

    # Perform HASH160
    pubkey_hash160 = hash160(public_key)
    print(f"• HASH160 (RIPEMD160(SHA256(pubkey))): {pubkey_hash160.hex()}")

    # Choose network prefix
    version_byte = b'\x6f' if testnet else b'\x00'  # 0x6f for testnet, 0x00 for mainnet

    # Encode in Base58Check
    btc_address = base58check_encode(version_byte, pubkey_hash160)
    print(f"✅ Bitcoin Address ({'testnet' if testnet else 'mainnet'}): {btc_address}\n")
    return btc_address


# Example usage
if __name__ == "__main__":
    hex_privkey = "038109007313a5807b2eccc082c8c3fbb988a973cacf1a7df9ce725c31b14776"
    generate_btc_address(hex_privkey, compressed=True, testnet=False)
