#!/usr/bin/env python3
"""
Advanced Blockchain Network Simulator - FIXED VERSION
Enterprise-grade blockchain simulation with multiple consensus algorithms,
network topology, DeFi protocols, cross-chain bridges, and real-time analytics

Requirements:
pip install customtkinter ecdsa numpy matplotlib networkx plotly dash threading asyncio
"""

import customtkinter as ctk
import hashlib
import json
import time
import threading
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import tkinter as tk
from tkinter import messagebox, ttk
import ecdsa
import random
import string
import numpy as np
from collections import defaultdict, deque
import queue
from dataclasses import dataclass, field
from enum import Enum
import math

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ConsensusType(Enum):
    PROOF_OF_WORK = "PoW"
    PROOF_OF_STAKE = "PoS"
    DELEGATED_PROOF_OF_STAKE = "DPoS"
    PRACTICAL_BFT = "pBFT"
    TENDERMINT = "Tendermint"
    PROOF_OF_AUTHORITY = "PoA"
    PROOF_OF_HISTORY = "PoH"

class NetworkTopology(Enum):
    MESH = "Mesh"
    STAR = "Star"
    RING = "Ring"
    TREE = "Tree"
    SMALL_WORLD = "Small World"

class TransactionType(Enum):
    TRANSFER = "Transfer"
    CONTRACT_DEPLOY = "Deploy"
    CONTRACT_CALL = "Call"
    STAKING = "Stake"
    GOVERNANCE = "Governance"
    CROSS_CHAIN = "CrossChain"
    MEV = "MEV"

@dataclass
class NetworkStats:
    tps: float = 0.0
    latency: float = 0.0
    throughput: float = 0.0
    finality_time: float = 0.0
    gas_price: float = 0.001
    network_hashrate: float = 0.0
    total_stake: float = 0.0
    active_validators: int = 0
    mempool_size: int = 0
    block_time: float = 10.0

@dataclass
class EconomicMetrics:
    total_supply: float = 1000000.0
    circulating_supply: float = 500000.0
    market_cap: float = 0.0
    inflation_rate: float = 0.02
    staking_rewards: float = 0.0
    burned_tokens: float = 0.0
    treasury_balance: float = 0.0
    transaction_fees: float = 0.0

class AdvancedTransaction:
    """Advanced transaction with MEV, priority fees, and cross-chain support"""
    def __init__(self, sender: str, receiver: str, amount: float, 
                 tx_type: TransactionType = TransactionType.TRANSFER,
                 priority_fee: float = 0.0, max_fee: float = 0.01,
                 data: Dict[str, Any] = None, nonce: int = 0):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.tx_type = tx_type
        self.priority_fee = priority_fee
        self.max_fee = max_fee
        self.base_fee = 0.001
        self.data = data or {}
        self.nonce = nonce
        self.timestamp = datetime.now().isoformat()
        self.tx_id = self.generate_tx_id()
        self.signature = None
        self.gas_limit = 21000
        self.gas_used = 0
        self.status = "pending"
        self.confirmations = 0
        self.mev_value = 0.0
        
    def generate_tx_id(self) -> str:
        data = f"{self.sender}{self.receiver}{self.amount}{self.timestamp}{self.nonce}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def calculate_total_fee(self) -> float:
        return self.base_fee + self.priority_fee
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tx_id': self.tx_id,
            'sender': self.sender,
            'receiver': self.receiver,
            'amount': self.amount,
            'tx_type': self.tx_type.value,
            'priority_fee': self.priority_fee,
            'max_fee': self.max_fee,
            'base_fee': self.base_fee,
            'data': self.data,
            'nonce': self.nonce,
            'timestamp': self.timestamp,
            'signature': self.signature,
            'gas_limit': self.gas_limit,
            'gas_used': self.gas_used,
            'status': self.status,
            'mev_value': self.mev_value
        }

class MEVBot:
    """Maximal Extractable Value bot for advanced transaction manipulation"""
    def __init__(self, name: str, strategy: str = "arbitrage"):
        self.name = name
        self.strategy = strategy
        self.profit = 0.0
        self.transactions_extracted = 0
        self.active = True
        
    def detect_opportunity(self, mempool: List[AdvancedTransaction]) -> Optional[AdvancedTransaction]:
        if not self.active or not mempool:
            return None
            
        # Simple arbitrage detection
        for tx in mempool:
            if tx.tx_type == TransactionType.TRANSFER and tx.amount > 100:
                # Create MEV transaction
                mev_tx = AdvancedTransaction(
                    sender="MEV_Bot",
                    receiver=tx.receiver,
                    amount=tx.amount * 0.01,  # Extract 1% as MEV
                    tx_type=TransactionType.MEV,
                    priority_fee=tx.priority_fee + 0.001  # Higher priority
                )
                mev_tx.mev_value = tx.amount * 0.01
                return mev_tx
        return None

class AdvancedSmartContract:
    """Advanced smart contract with gas optimization and cross-chain capabilities"""
    def __init__(self, name: str, code: str, creator: str, contract_type: str = "ERC20"):
        self.name = name
        self.code = code
        self.creator = creator
        self.contract_type = contract_type
        self.deployed_at = datetime.now().isoformat()
        self.address = self.generate_address()
        self.state = {}
        self.storage_slots = {}
        self.gas_optimized = False
        self.cross_chain_enabled = False
        self.upgrade_proxy = None
        self.governance_enabled = False
        self.audit_score = random.randint(70, 100)
        
    def generate_address(self) -> str:
        data = f"{self.name}{self.creator}{self.deployed_at}"
        return "0x" + hashlib.sha256(data.encode()).hexdigest()[:40]
    
    def optimize_gas(self):
        """Simulate gas optimization"""
        self.gas_optimized = True
        return random.randint(10, 30)  # % gas savings
    
    def enable_cross_chain(self, bridge_address: str):
        """Enable cross-chain functionality"""
        self.cross_chain_enabled = True
        self.state['bridge_address'] = bridge_address
        
    def execute_function(self, function_name: str, params: Dict[str, Any], gas_limit: int = 100000) -> Dict[str, Any]:
        """Execute contract function with advanced features"""
        gas_used = random.randint(20000, min(gas_limit, 80000))
        if self.gas_optimized:
            gas_used = int(gas_used * 0.8)  # 20% gas savings
            
        success = gas_used <= gas_limit
        
        result = {
            'success': success,
            'gas_used': gas_used,
            'result': f"Executed {function_name}({params})",
            'events': [],
            'state_changes': {}
        }
        
        if success and function_name == "transfer":
            result['events'] = [{'Transfer': {'from': params.get('from'), 'to': params.get('to'), 'value': params.get('amount')}}]
            
        return result

class Validator:
    """Advanced validator with slashing, rewards, and delegation"""
    def __init__(self, address: str, stake: float, commission: float = 0.05):
        self.address = address
        self.stake = stake
        self.commission = commission
        self.delegated_stake = 0.0
        self.delegators = {}
        self.is_active = True
        self.uptime = 1.0
        self.blocks_validated = 0
        self.rewards_earned = 0.0
        self.slashed_amount = 0.0  # Total amount slashed (float)
        self.slashing_events = []  # List of slashing events
        self.reputation = 100.0
        self.last_active = datetime.now()
        
    def add_delegation(self, delegator: str, amount: float):
        if delegator in self.delegators:
            self.delegators[delegator] += amount
        else:
            self.delegators[delegator] = amount
        self.delegated_stake += amount
        
    def slash(self, percentage: float, reason: str):
        """Slash validator for misbehavior"""
        slash_amount = self.total_stake() * percentage
        self.slashed_amount += slash_amount
        self.slashing_events.append({
            'amount': slash_amount,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
        self.stake = max(0, self.stake - slash_amount * 0.5)
        self.reputation = max(0, self.reputation - 20)
        return slash_amount, reason
        
    def calculate_rewards(self, base_reward: float) -> float:
        """Calculate validator rewards including delegator shares"""
        total_reward = base_reward * (1 + self.delegated_stake / self.stake)
        validator_reward = total_reward * self.commission
        delegator_rewards = total_reward - validator_reward
        
        self.rewards_earned += validator_reward
        return validator_reward
        
    def total_stake(self) -> float:
        return self.stake + self.delegated_stake

class NetworkNode:
    """Advanced network node with gossip protocol and peer management"""
    def __init__(self, node_id: str, node_type: str = "full", location: Tuple[float, float] = (0, 0)):
        self.node_id = node_id
        self.node_type = node_type  # full, light, validator, archive
        self.location = location
        self.peers = set()
        self.is_online = True
        self.latency_map = {}
        self.bandwidth = random.randint(100, 1000)  # Mbps
        self.storage_capacity = random.randint(100, 1000)  # GB
        self.cpu_power = random.randint(1, 16)  # CPU cores
        self.sync_status = "synced"
        self.mempool = []
        self.last_ping = datetime.now()
        
    def add_peer(self, peer_id: str, latency: float = None):
        self.peers.add(peer_id)
        if latency:
            self.latency_map[peer_id] = latency
            
    def calculate_distance(self, other_node: 'NetworkNode') -> float:
        """Calculate geographical distance between nodes"""
        lat1, lon1 = self.location
        lat2, lon2 = other_node.location
        return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111  # Rough km conversion
        
    def propagate_transaction(self, transaction: AdvancedTransaction) -> float:
        """Simulate transaction propagation delay"""
        base_delay = 0.1  # 100ms base
        network_delay = len(self.peers) * 0.01  # Network congestion
        distance_delay = sum(self.latency_map.get(peer, 0.05) for peer in self.peers) / max(len(self.peers), 1)
        return base_delay + network_delay + distance_delay

class AdvancedBlock:
    """Advanced block with EIP-1559, MEV, and advanced features"""
    def __init__(self, index: int, transactions: List[AdvancedTransaction], 
                 previous_hash: str, validator: str, consensus_type: ConsensusType):
        self.index = index
        self.transactions = transactions
        self.timestamp = datetime.now().isoformat()
        self.previous_hash = previous_hash
        self.validator = validator
        self.consensus_type = consensus_type
        self.nonce = 0
        self.difficulty = 2
        self.base_fee = 0.001
        self.gas_limit = 30000000
        self.gas_used = sum(tx.gas_used for tx in transactions)
        self.mev_rewards = sum(tx.mev_value for tx in transactions if tx.tx_type == TransactionType.MEV)
        self.validator_rewards = 0.0
        self.merkle_root = self.calculate_merkle_root()
        self.state_root = self.calculate_state_root()
        self.receipts_root = self.calculate_receipts_root()
        self.hash = self.calculate_hash()
        self.confirmations = 0
        self.finalized = False
        
    def calculate_hash(self) -> str:
        block_data = {
            'index': self.index,
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'state_root': self.state_root,
            'nonce': self.nonce,
            'validator': self.validator,
            'gas_used': self.gas_used,
            'base_fee': self.base_fee
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def calculate_merkle_root(self) -> str:
        if not self.transactions:
            return hashlib.sha256("".encode()).hexdigest()
        
        tx_hashes = [hashlib.sha256(json.dumps(tx.to_dict()).encode()).hexdigest() 
                    for tx in self.transactions]
        
        while len(tx_hashes) > 1:
            if len(tx_hashes) % 2 == 1:
                tx_hashes.append(tx_hashes[-1])
            
            new_hashes = []
            for i in range(0, len(tx_hashes), 2):
                combined = tx_hashes[i] + tx_hashes[i + 1]
                new_hashes.append(hashlib.sha256(combined.encode()).hexdigest())
            tx_hashes = new_hashes
        
        return tx_hashes[0]
    
    def calculate_state_root(self) -> str:
        # Simplified state root calculation
        state_data = f"state_{self.index}_{len(self.transactions)}"
        return hashlib.sha256(state_data.encode()).hexdigest()
    
    def calculate_receipts_root(self) -> str:
        # Simplified receipts root
        receipts_data = f"receipts_{self.index}_{self.gas_used}"
        return hashlib.sha256(receipts_data.encode()).hexdigest()
    
    def calculate_priority_fee_rewards(self) -> float:
        """Calculate total priority fees for validator"""
        return sum(tx.priority_fee for tx in self.transactions)

class DeFiProtocol:
    """Advanced DeFi protocol simulation"""
    def __init__(self, protocol_type: str, name: str):
        self.protocol_type = protocol_type  # AMM, Lending, Derivatives, etc.
        self.name = name
        self.tvl = 0.0  # Total Value Locked
        self.volume_24h = 0.0
        self.fees_collected = 0.0
        self.liquidity_pools = {}
        self.governance_token = None
        self.treasury = 0.0
        
    def add_liquidity(self, token_a: str, token_b: str, amount_a: float, amount_b: float) -> float:
        """Add liquidity to AMM pool"""
        pool_id = f"{token_a}/{token_b}"
        if pool_id not in self.liquidity_pools:
            self.liquidity_pools[pool_id] = {'token_a': 0, 'token_b': 0, 'lp_tokens': 0}
        
        pool = self.liquidity_pools[pool_id]
        
        # Calculate LP tokens to mint
        if pool['lp_tokens'] == 0:
            lp_tokens = math.sqrt(amount_a * amount_b)
        else:
            lp_tokens = min(
                amount_a * pool['lp_tokens'] / pool['token_a'],
                amount_b * pool['lp_tokens'] / pool['token_b']
            )
        
        pool['token_a'] += amount_a
        pool['token_b'] += amount_b
        pool['lp_tokens'] += lp_tokens
        self.tvl += amount_a + amount_b  # Simplified TVL calculation
        
        return lp_tokens
    
    def swap(self, token_in: str, token_out: str, amount_in: float) -> float:
        """Execute token swap with fees"""
        pool_id = f"{token_in}/{token_out}"
        if pool_id not in self.liquidity_pools:
            pool_id = f"{token_out}/{token_in}"
        
        if pool_id not in self.liquidity_pools:
            return 0.0
        
        pool = self.liquidity_pools[pool_id]
        fee_rate = 0.003  # 0.3% fee
        
        # Simplified constant product formula (x * y = k)
        amount_in_with_fee = amount_in * (1 - fee_rate)
        
        if token_in in pool_id.split('/')[0]:
            reserve_in = pool['token_a']
            reserve_out = pool['token_b']
        else:
            reserve_in = pool['token_b']
            reserve_out = pool['token_a']
        
        amount_out = (reserve_out * amount_in_with_fee) / (reserve_in + amount_in_with_fee)
        
        # Update pool reserves
        if token_in in pool_id.split('/')[0]:
            pool['token_a'] += amount_in
            pool['token_b'] -= amount_out
        else:
            pool['token_b'] += amount_in
            pool['token_a'] -= amount_out
        
        fee_collected = amount_in * fee_rate
        self.fees_collected += fee_collected
        self.volume_24h += amount_in
        
        return amount_out

class CrossChainBridge:
    """Cross-chain bridge simulation"""
    def __init__(self, chain_a: str, chain_b: str):
        self.chain_a = chain_a
        self.chain_b = chain_b
        self.locked_assets = {chain_a: {}, chain_b: {}}
        self.bridge_fee = 0.001
        self.security_model = "optimistic"  # or "zk-proof"
        self.validators = []
        self.total_volume = 0.0
        
    def bridge_asset(self, asset: str, amount: float, from_chain: str, to_chain: str) -> Dict[str, Any]:
        """Bridge asset between chains"""
        if from_chain not in [self.chain_a, self.chain_b] or to_chain not in [self.chain_a, self.chain_b]:
            return {'success': False, 'error': 'Invalid chain'}
        
        fee = amount * self.bridge_fee
        bridged_amount = amount - fee
        
        # Lock asset on source chain
        if asset not in self.locked_assets[from_chain]:
            self.locked_assets[from_chain][asset] = 0
        self.locked_assets[from_chain][asset] += amount
        
        # Mint wrapped asset on destination chain
        wrapped_asset = f"w{asset}"
        if wrapped_asset not in self.locked_assets[to_chain]:
            self.locked_assets[to_chain][wrapped_asset] = 0
        self.locked_assets[to_chain][wrapped_asset] += bridged_amount
        
        self.total_volume += amount
        
        return {
            'success': True,
            'bridged_amount': bridged_amount,
            'fee': fee,
            'wrapped_asset': wrapped_asset,
            'confirmation_time': random.randint(5, 30)  # minutes
        }

class AdvancedBlockchain:
    """Advanced blockchain with multiple consensus mechanisms"""
    def __init__(self, name: str, consensus_type: ConsensusType, network_topology: NetworkTopology):
        self.name = name
        self.consensus_type = consensus_type
        self.network_topology = network_topology
        self.chain: List[AdvancedBlock] = []
        self.pending_transactions: List[AdvancedTransaction] = []
        self.nodes: Dict[str, NetworkNode] = {}
        self.validators: Dict[str, Validator] = {}
        self.users: Dict[str, 'AdvancedUser'] = {}
        self.smart_contracts: Dict[str, AdvancedSmartContract] = {}
        self.defi_protocols: Dict[str, DeFiProtocol] = {}
        self.cross_chain_bridges: Dict[str, CrossChainBridge] = {}
        self.mev_bots: List[MEVBot] = []
        self.network_stats = NetworkStats()
        self.economic_metrics = EconomicMetrics()
        self.initial_balances: Dict[str, float] = {}
        self.staking_pools: Dict[str, Dict] = {}
        self.governance_proposals: List[Dict] = []
        self.slashing_conditions: Dict[str, float] = {
            'double_sign': 0.05,
            'downtime': 0.01,
            'invalid_block': 0.1
        }
        self.create_genesis_block()
        self.setup_network_topology()
        
    def create_genesis_block(self) -> None:
        """Create genesis block with advanced features"""
        genesis_block = AdvancedBlock(0, [], "0", "Genesis", self.consensus_type)
        genesis_block.hash = genesis_block.calculate_hash()
        genesis_block.finalized = True
        self.chain.append(genesis_block)
        
    def setup_network_topology(self):
        """Setup network topology based on type"""
        node_count = 20
        
        # Create nodes with geographical distribution
        for i in range(node_count):
            lat = random.uniform(-90, 90)
            lon = random.uniform(-180, 180)
            node_type = random.choice(["full", "light", "validator"])
            
            node = NetworkNode(f"node_{i}", node_type, (lat, lon))
            self.nodes[node.node_id] = node
            
        # Connect nodes based on topology
        node_list = list(self.nodes.values())
        
        if self.network_topology == NetworkTopology.MESH:
            # Full mesh - every node connected to every other
            for i, node1 in enumerate(node_list):
                for j, node2 in enumerate(node_list[i+1:], i+1):
                    latency = node1.calculate_distance(node2) / 300000  # Speed of light delay
                    node1.add_peer(node2.node_id, latency)
                    node2.add_peer(node1.node_id, latency)
                    
        elif self.network_topology == NetworkTopology.SMALL_WORLD:
            # Small world network - high clustering, short path lengths
            for node in node_list:
                # Connect to nearby nodes geographically
                distances = [(other.node_id, node.calculate_distance(other)) 
                           for other in node_list if other != node]
                distances.sort(key=lambda x: x[1])
                
                # Connect to 6 closest nodes
                for peer_id, distance in distances[:6]:
                    latency = distance / 300000
                    node.add_peer(peer_id, latency)
                    
                # Add some random long-distance connections
                for _ in range(2):
                    random_peer = random.choice(node_list)
                    if random_peer != node:
                        latency = node.calculate_distance(random_peer) / 300000
                        node.add_peer(random_peer.node_id, latency)
    
    def add_validator(self, address: str, stake: float, commission: float = 0.05) -> bool:
        """Add a new validator to the network"""
        if address in self.validators:
            return False
            
        if stake < 32.0:  # Minimum stake requirement
            return False
            
        validator = Validator(address, stake, commission)
        self.validators[address] = validator
        self.network_stats.active_validators += 1
        self.network_stats.total_stake += stake
        
        return True
    
    def delegate_stake(self, delegator: str, validator_address: str, amount: float) -> bool:
        """Delegate stake to a validator"""
        if validator_address not in self.validators:
            return False
            
        if self.get_balance(delegator) < amount:
            return False
            
        validator = self.validators[validator_address]
        validator.add_delegation(delegator, amount)
        
        # Deduct from delegator balance
        self.initial_balances[delegator] = self.initial_balances.get(delegator, 0) - amount
        
        return True
    
    def select_validator_by_consensus(self) -> Optional[str]:
        """Select validator based on consensus mechanism"""
        if not self.validators:
            return None
            
        active_validators = [v for v in self.validators.values() if v.is_active]
        if not active_validators:
            return None
            
        if self.consensus_type == ConsensusType.PROOF_OF_STAKE:
            # Weighted selection based on stake
            total_stake = sum(v.total_stake() for v in active_validators)
            if total_stake == 0:
                return random.choice(active_validators).address
                
            weights = [v.total_stake() / total_stake for v in active_validators]
            selected = np.random.choice(active_validators, p=weights)
            return selected.address
            
        elif self.consensus_type == ConsensusType.DELEGATED_PROOF_OF_STAKE:
            # Top validators by delegated stake
            top_validators = sorted(active_validators, key=lambda v: v.total_stake(), reverse=True)[:21]
            return random.choice(top_validators).address
            
        elif self.consensus_type == ConsensusType.PROOF_OF_AUTHORITY:
            # Round-robin among authorized validators
            return random.choice(active_validators).address
            
        else:  # PoW and others
            return random.choice(active_validators).address
    
    def validate_block_by_consensus(self, block: AdvancedBlock) -> bool:
        """Validate block based on consensus rules"""
        if self.consensus_type == ConsensusType.PRACTICAL_BFT:
            # Require 2/3+ validator approval
            required_approvals = math.ceil(len(self.validators) * 2/3)
            approvals = min(required_approvals, len(self.validators))
            return approvals >= required_approvals
            
        elif self.consensus_type == ConsensusType.TENDERMINT:
            # Two-phase commit with finality
            pre_commit_threshold = math.ceil(len(self.validators) * 2/3)
            return True  # Simplified for demo
            
        else:
            # Standard validation
            return True
    
    def simulate_mev_extraction(self) -> List[AdvancedTransaction]:
        """Simulate MEV bot activities"""
        mev_transactions = []
        
        for bot in self.mev_bots:
            mev_tx = bot.detect_opportunity(self.pending_transactions)
            if mev_tx:
                mev_transactions.append(mev_tx)
                bot.profit += mev_tx.mev_value
                bot.transactions_extracted += 1
                
        return mev_transactions
    
    def mine_advanced_block(self, validator_address: str) -> Optional[AdvancedBlock]:
        """Mine block with advanced features"""
        if not self.pending_transactions:
            return None
            
        # Add MEV transactions
        mev_transactions = self.simulate_mev_extraction()
        all_transactions = self.pending_transactions + mev_transactions
        
        # Sort by priority fee (EIP-1559 style)
        all_transactions.sort(key=lambda tx: tx.priority_fee + tx.base_fee, reverse=True)
        
        # Select transactions for block (considering gas limit)
        selected_transactions = []
        total_gas = 0
        gas_limit = 30000000
        
        for tx in all_transactions:
            if total_gas + tx.gas_limit <= gas_limit:
                selected_transactions.append(tx)
                total_gas += tx.gas_limit
                tx.gas_used = random.randint(int(tx.gas_limit * 0.7), tx.gas_limit)
                tx.status = "confirmed"
            else:
                break
        
        # Create new block
        new_block = AdvancedBlock(
            len(self.chain),
            selected_transactions,
            self.get_latest_block().hash,
            validator_address,
            self.consensus_type
        )
        
        # Calculate rewards
        if validator_address in self.validators:
            validator = self.validators[validator_address]
            base_reward = 2.0  # Base block reward
            priority_fees = new_block.calculate_priority_fee_rewards()
            mev_rewards = new_block.mev_rewards
            
            total_reward = base_reward + priority_fees + mev_rewards
            new_block.validator_rewards = total_reward
            
            validator.calculate_rewards(base_reward)
            validator.blocks_validated += 1
        
        # Validate block
        if self.validate_block_by_consensus(new_block):
            # Mine the block based on consensus
            if self.consensus_type == ConsensusType.PROOF_OF_WORK:
                new_block.nonce = 0
                target = "0" * new_block.difficulty
                while not new_block.hash.startswith(target):
                    new_block.nonce += 1
                    new_block.hash = new_block.calculate_hash()
                    if new_block.nonce > 100000:  # Safety limit
                        break
            
            # Add to chain
            self.chain.append(new_block)
            
            # Remove confirmed transactions from pending
            confirmed_tx_ids = {tx.tx_id for tx in selected_transactions}
            self.pending_transactions = [tx for tx in self.pending_transactions if tx.tx_id not in confirmed_tx_ids]
            
            # Update network stats
            self.update_network_stats(new_block)
            
            return new_block
        
        return None
    
    def update_network_stats(self, block: AdvancedBlock):
        """Update network performance statistics"""
        # Calculate TPS
        if len(self.chain) > 1:
            time_diff = (datetime.fromisoformat(block.timestamp) - 
                        datetime.fromisoformat(self.chain[-2].timestamp)).total_seconds()
            if time_diff > 0:
                self.network_stats.tps = len(block.transactions) / time_diff
        
        # Update other stats
        self.network_stats.mempool_size = len(self.pending_transactions)
        self.network_stats.gas_price = np.mean([tx.calculate_total_fee() for tx in self.pending_transactions]) if self.pending_transactions else 0.001
        
        # Update economic metrics
        self.economic_metrics.transaction_fees += sum(tx.calculate_total_fee() for tx in block.transactions)
        
        if block.mev_rewards > 0:
            self.economic_metrics.burned_tokens += block.mev_rewards * 0.1  # Burn 10% of MEV
    
    def add_defi_protocol(self, protocol: DeFiProtocol):
        """Add DeFi protocol to ecosystem"""
        self.defi_protocols[protocol.name] = protocol
    
    def add_cross_chain_bridge(self, bridge: CrossChainBridge):
        """Add cross-chain bridge"""
        bridge_id = f"{bridge.chain_a}_{bridge.chain_b}"
        self.cross_chain_bridges[bridge_id] = bridge
    
    def get_latest_block(self) -> AdvancedBlock:
        return self.chain[-1]
    
    def get_balance(self, address: str) -> float:
        balance = self.initial_balances.get(address, 0.0)
        
        for block in self.chain:
            for tx in block.transactions:
                if tx.sender == address:
                    balance -= (tx.amount + tx.calculate_total_fee())
                if tx.receiver == address:
                    balance += tx.amount
        
        return balance
    
    def add_transaction(self, transaction: AdvancedTransaction) -> bool:
        if transaction.sender == "System" or transaction.tx_type == TransactionType.MEV:
            self.pending_transactions.append(transaction)
            return True
            
        sender_balance = self.get_balance(transaction.sender)
        total_cost = transaction.amount + transaction.calculate_total_fee()
        
        if sender_balance >= total_cost:
            self.pending_transactions.append(transaction)
            return True
        return False

class AdvancedUser:
    """Advanced user with staking, DeFi, and cross-chain capabilities"""
    def __init__(self, name: str):
        self.name = name
        self.private_key = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
        self.public_key = self.private_key.get_verifying_key()
        self.address = self.generate_address()
        self.nonce = 0
        self.staked_amount = 0.0
        self.delegated_validators = {}
        self.defi_positions = {}
        self.governance_power = 0.0
        self.reputation_score = 100.0
        
    def generate_address(self) -> str:
        pub_key_hex = self.public_key.to_string().hex()
        return "0x" + hashlib.sha256(pub_key_hex.encode()).hexdigest()[:40]
    
    def sign_transaction(self, transaction: AdvancedTransaction) -> str:
        tx_data = json.dumps(transaction.to_dict(), sort_keys=True)
        signature = self.private_key.sign(tx_data.encode())
        return signature.hex()

class AdvancedBlockchainSimulator:
    """Main advanced blockchain simulator application"""
    def __init__(self):
        self.blockchains: Dict[str, AdvancedBlockchain] = {}
        self.current_blockchain = None
        self.current_user = None
        self.mining_active = False
        self.simulation_running = False
        self.performance_data = deque(maxlen=100)
        
        # Initialize GUI
        self.root = ctk.CTk()
        self.root.title("Advanced Blockchain Network Simulator")
        self.root.geometry("1600x1000")
        
        # Create sample blockchains and users
        self.create_sample_data()
        self.setup_advanced_gui()
        
    def create_sample_data(self):
        """Create sample blockchains and users"""
        # Create different blockchain networks
        eth_like = AdvancedBlockchain("Ethereum-Like", ConsensusType.PROOF_OF_STAKE, NetworkTopology.SMALL_WORLD)
        btc_like = AdvancedBlockchain("Bitcoin-Like", ConsensusType.PROOF_OF_WORK, NetworkTopology.MESH)
        solana_like = AdvancedBlockchain("Solana-Like", ConsensusType.PROOF_OF_HISTORY, NetworkTopology.STAR)
        
        self.blockchains["Ethereum-Like"] = eth_like
        self.blockchains["Bitcoin-Like"] = btc_like  
        self.blockchains["Solana-Like"] = solana_like
        
        self.current_blockchain = eth_like
        
        # Create advanced users
        users = ["Alice", "Bob", "Charlie", "Diana", "Validator1", "Validator2", "MEV_Bot"]
        for name in users:
            user = AdvancedUser(name)
            eth_like.users[user.address] = user
            eth_like.initial_balances[user.address] = 1000.0
            
            # Add some as validators
            if "Validator" in name:
                eth_like.add_validator(user.address, 100.0, 0.05)
        
        # Add MEV bots
        eth_like.mev_bots.append(MEVBot("ArbitrageBot", "arbitrage"))
        eth_like.mev_bots.append(MEVBot("SandwichBot", "sandwich"))
        
        # Add DeFi protocols
        uniswap = DeFiProtocol("AMM", "UniswapV3")
        aave = DeFiProtocol("Lending", "AAVE")
        eth_like.add_defi_protocol(uniswap)
        eth_like.add_defi_protocol(aave)
        
        # Add cross-chain bridge
        bridge = CrossChainBridge("Ethereum-Like", "Bitcoin-Like")
        eth_like.add_cross_chain_bridge(bridge)
    
    def setup_advanced_gui(self):
        """Setup advanced GUI with multiple panels"""
        # Create main container
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create header with blockchain selection
        self.create_advanced_header()
        
        # Create tabview with advanced features
        self.tabview = ctk.CTkTabview(self.main_container)
        self.tabview.pack(fill="both", expand=True, pady=(10, 0))
        
        # Add advanced tabs
        self.create_network_overview_tab()
        self.create_consensus_tab()
        self.create_defi_tab()
        self.create_cross_chain_tab()
        self.create_mev_tab()
        self.create_governance_tab()
        self.create_analytics_tab()
        self.create_network_topology_tab()
        
        # Start background simulation
        self.start_simulation()
        
    def create_advanced_header(self):
        """Create advanced header with blockchain selection"""
        header_frame = ctk.CTkFrame(self.main_container)
        header_frame.pack(fill="x", pady=(0, 10))
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame, 
            text="🌐 Advanced Blockchain Network Simulator", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(side="left", padx=20, pady=15)
        
        # Control panel
        control_frame = ctk.CTkFrame(header_frame)
        control_frame.pack(side="right", padx=20, pady=10)
        
        # Blockchain selection
        ctk.CTkLabel(control_frame, text="Blockchain:").grid(row=0, column=0, padx=5, pady=5)
        self.blockchain_var = ctk.StringVar(value="Ethereum-Like")
        blockchain_combo = ctk.CTkComboBox(
            control_frame,
            values=list(self.blockchains.keys()),
            variable=self.blockchain_var,
            command=self.on_blockchain_change
        )
        blockchain_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # User selection
        ctk.CTkLabel(control_frame, text="User:").grid(row=0, column=2, padx=5, pady=5)
        self.user_var = ctk.StringVar()
        self.user_dropdown = ctk.CTkComboBox(
            control_frame,
            values=[],
            variable=self.user_var,
            command=self.on_user_change
        )
        self.user_dropdown.grid(row=0, column=3, padx=5, pady=5)
        
        # Simulation controls
        self.simulation_btn = ctk.CTkButton(
            control_frame,
            text="⏸️ Pause Simulation",
            command=self.toggle_simulation
        )
        self.simulation_btn.grid(row=0, column=4, padx=5, pady=5)
        
        # Update user dropdown
        self.update_user_dropdown()
    
    def create_network_overview_tab(self):
        """Create network overview tab"""
        self.overview_tab = self.tabview.add("🌐 Network Overview")
        
        # Top metrics row
        metrics_frame = ctk.CTkFrame(self.overview_tab)
        metrics_frame.pack(fill="x", padx=10, pady=10)
        
        self.metric_labels = {}
        metrics = [
            ("TPS", "0.0"), ("Latency", "0ms"), ("Gas Price", "0.001"), 
            ("Validators", "0"), ("TVL", "$0"), ("Volume 24h", "$0")
        ]
        
        for i, (metric, value) in enumerate(metrics):
            metric_frame = ctk.CTkFrame(metrics_frame)
            metric_frame.grid(row=0, column=i, padx=5, pady=5, sticky="ew")
            metrics_frame.grid_columnconfigure(i, weight=1)
            
            value_label = ctk.CTkLabel(metric_frame, text=value, font=ctk.CTkFont(size=18, weight="bold"))
            value_label.pack(pady=(5, 0))
            
            name_label = ctk.CTkLabel(metric_frame, text=metric)
            name_label.pack(pady=(0, 5))
            
            self.metric_labels[metric] = value_label
        
        # Content area
        content_frame = ctk.CTkFrame(self.overview_tab)
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel - Live feed
        left_panel = ctk.CTkFrame(content_frame)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        ctk.CTkLabel(left_panel, text="📊 Live Network Activity", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.live_feed = ctk.CTkTextbox(left_panel, height=400)
        self.live_feed.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Right panel - Network stats
        right_panel = ctk.CTkFrame(content_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        ctk.CTkLabel(right_panel, text="📈 Performance Metrics", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.performance_text = ctk.CTkTextbox(right_panel, height=400)
        self.performance_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_consensus_tab(self):
        """Create consensus mechanisms tab"""
        self.consensus_tab = self.tabview.add("🤝 Consensus")
        
        # Consensus configuration
        config_frame = ctk.CTkFrame(self.consensus_tab)
        config_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(config_frame, text="⚙️ Consensus Configuration", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        config_inner = ctk.CTkFrame(config_frame)
        config_inner.pack(fill="x", padx=10, pady=10)
        
        # Consensus type selection
        ctk.CTkLabel(config_inner, text="Consensus Type:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.consensus_var = ctk.StringVar(value="PoS")
        consensus_combo = ctk.CTkComboBox(
            config_inner,
            values=["PoW", "PoS", "DPoS", "pBFT", "Tendermint", "PoA", "PoH"],
            variable=self.consensus_var,
            command=self.on_consensus_change
        )
        consensus_combo.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        # Mining/Validation controls
        ctk.CTkButton(config_inner, text="⛏️ Start Mining/Validation", command=self.start_consensus).grid(row=0, column=2, padx=10, pady=5)
        ctk.CTkButton(config_inner, text="⏹️ Stop", command=self.stop_consensus).grid(row=0, column=3, padx=10, pady=5)
        
        config_inner.grid_columnconfigure(1, weight=1)
        
        # Validators list
        validators_frame = ctk.CTkFrame(self.consensus_tab)
        validators_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(validators_frame, text="👥 Validators & Delegators", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.validators_text = ctk.CTkTextbox(validators_frame)
        self.validators_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_defi_tab(self):
        """Create DeFi protocols tab"""
        self.defi_tab = self.tabview.add("🏦 DeFi")
        
        # Protocol selection
        protocol_frame = ctk.CTkFrame(self.defi_tab)
        protocol_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(protocol_frame, text="🏦 DeFi Protocols", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        protocol_inner = ctk.CTkFrame(protocol_frame)
        protocol_inner.pack(fill="x", padx=10, pady=10)
        
        # Protocol operations
        ctk.CTkLabel(protocol_inner, text="Protocol:").grid(row=0, column=0, padx=5, pady=5)
        self.protocol_var = ctk.StringVar()
        protocol_combo = ctk.CTkComboBox(protocol_inner, values=[], variable=self.protocol_var)
        protocol_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ctk.CTkLabel(protocol_inner, text="Action:").grid(row=0, column=2, padx=5, pady=5)
        self.defi_action_var = ctk.StringVar(value="Add Liquidity")
        action_combo = ctk.CTkComboBox(
            protocol_inner,
            values=["Add Liquidity", "Swap", "Lend", "Borrow", "Stake"],
            variable=self.defi_action_var
        )
        action_combo.grid(row=0, column=3, padx=5, pady=5)
        
        ctk.CTkButton(protocol_inner, text="Execute", command=self.execute_defi_action).grid(row=0, column=4, padx=5, pady=5)
        
        # DeFi metrics
        metrics_frame = ctk.CTkFrame(self.defi_tab)
        metrics_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(metrics_frame, text="📊 DeFi Metrics", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.defi_metrics_text = ctk.CTkTextbox(metrics_frame)
        self.defi_metrics_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Update protocol dropdown
        self.update_protocol_dropdown()
    
    def create_cross_chain_tab(self):
        """Create cross-chain bridge tab"""
        self.cross_chain_tab = self.tabview.add("🌉 Cross-Chain")
        
        # Bridge interface
        bridge_frame = ctk.CTkFrame(self.cross_chain_tab)
        bridge_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(bridge_frame, text="🌉 Cross-Chain Bridge", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        bridge_inner = ctk.CTkFrame(bridge_frame)
        bridge_inner.pack(fill="x", padx=10, pady=10)
        
        # Bridge controls
        ctk.CTkLabel(bridge_inner, text="From Chain:").grid(row=0, column=0, padx=5, pady=5)
        self.from_chain_var = ctk.StringVar()
        from_chain_combo = ctk.CTkComboBox(bridge_inner, values=list(self.blockchains.keys()), variable=self.from_chain_var)
        from_chain_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ctk.CTkLabel(bridge_inner, text="To Chain:").grid(row=0, column=2, padx=5, pady=5)
        self.to_chain_var = ctk.StringVar()
        to_chain_combo = ctk.CTkComboBox(bridge_inner, values=list(self.blockchains.keys()), variable=self.to_chain_var)
        to_chain_combo.grid(row=0, column=3, padx=5, pady=5)
        
        ctk.CTkLabel(bridge_inner, text="Asset:").grid(row=1, column=0, padx=5, pady=5)
        self.bridge_asset_entry = ctk.CTkEntry(bridge_inner, placeholder_text="ETH")
        self.bridge_asset_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ctk.CTkLabel(bridge_inner, text="Amount:").grid(row=1, column=2, padx=5, pady=5)
        self.bridge_amount_entry = ctk.CTkEntry(bridge_inner, placeholder_text="10.0")
        self.bridge_amount_entry.grid(row=1, column=3, padx=5, pady=5)
        
        ctk.CTkButton(bridge_inner, text="🌉 Bridge Asset", command=self.bridge_asset).grid(row=1, column=4, padx=5, pady=5)
        
        # Bridge status
        status_frame = ctk.CTkFrame(self.cross_chain_tab)
        status_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(status_frame, text="📊 Bridge Activity", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.bridge_status_text = ctk.CTkTextbox(status_frame)
        self.bridge_status_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_mev_tab(self):
        """Create MEV analysis tab"""
        self.mev_tab = self.tabview.add("⚡ MEV")
        
        # MEV controls
        mev_controls = ctk.CTkFrame(self.mev_tab)
        mev_controls.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(mev_controls, text="⚡ MEV Bots & Analysis", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        controls_inner = ctk.CTkFrame(mev_controls)
        controls_inner.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(controls_inner, text="🤖 Add MEV Bot", command=self.add_mev_bot).pack(side="left", padx=5)
        ctk.CTkButton(controls_inner, text="⚡ Trigger MEV Opportunity", command=self.trigger_mev).pack(side="left", padx=5)
        ctk.CTkButton(controls_inner, text="📊 Analyze MEV", command=self.analyze_mev).pack(side="left", padx=5)
        
        # MEV metrics
        mev_metrics = ctk.CTkFrame(self.mev_tab)
        mev_metrics.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(mev_metrics, text="📊 MEV Statistics", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.mev_metrics_text = ctk.CTkTextbox(mev_metrics)
        self.mev_metrics_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_governance_tab(self):
        """Create governance tab"""
        self.governance_tab = self.tabview.add("🗳️ Governance")
        
        # Proposal creation
        proposal_frame = ctk.CTkFrame(self.governance_tab)
        proposal_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(proposal_frame, text="🗳️ Governance Proposals", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        proposal_inner = ctk.CTkFrame(proposal_frame)
        proposal_inner.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(proposal_inner, text="Proposal:").grid(row=0, column=0, padx=5, pady=5)
        self.proposal_entry = ctk.CTkEntry(proposal_inner, placeholder_text="Increase block size to 2MB", width=300)
        self.proposal_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ctk.CTkButton(proposal_inner, text="📝 Create Proposal", command=self.create_proposal).grid(row=0, column=2, padx=5, pady=5)
        
        # Voting interface
        voting_frame = ctk.CTkFrame(self.governance_tab)
        voting_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(voting_frame, text="🗳️ Active Proposals", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.governance_text = ctk.CTkTextbox(voting_frame)
        self.governance_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_analytics_tab(self):
        """Create analytics and visualization tab"""
        self.analytics_tab = self.tabview.add("📊 Analytics")
        
        # Analytics controls
        analytics_controls = ctk.CTkFrame(self.analytics_tab)
        analytics_controls.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(analytics_controls, text="📊 Network Analytics", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        controls_inner = ctk.CTkFrame(analytics_controls)
        controls_inner.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(controls_inner, text="📈 Generate Report", command=self.generate_analytics_report).pack(side="left", padx=5)
        ctk.CTkButton(controls_inner, text="📊 Export Data", command=self.export_analytics_data).pack(side="left", padx=5)
        ctk.CTkButton(controls_inner, text="🔄 Refresh", command=self.refresh_analytics).pack(side="left", padx=5)
        
        # Analytics display
        analytics_display = ctk.CTkFrame(self.analytics_tab)
        analytics_display.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.analytics_text = ctk.CTkTextbox(analytics_display)
        self.analytics_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_network_topology_tab(self):
        """Create network topology visualization tab"""
        self.topology_tab = self.tabview.add("🕸️ Network Topology")
        
        # Topology controls
        topology_controls = ctk.CTkFrame(self.topology_tab)
        topology_controls.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(topology_controls, text="🕸️ Network Topology", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        controls_inner = ctk.CTkFrame(topology_controls)
        controls_inner.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(controls_inner, text="Topology:").grid(row=0, column=0, padx=5, pady=5)
        self.topology_var = ctk.StringVar(value="Small World")
        topology_combo = ctk.CTkComboBox(
            controls_inner,
            values=["Mesh", "Star", "Ring", "Tree", "Small World"],
            variable=self.topology_var,
            command=self.change_topology
        )
        topology_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ctk.CTkButton(controls_inner, text="🔄 Simulate Network Effect", command=self.simulate_network_effect).grid(row=0, column=2, padx=5, pady=5)
        
        # Topology display
        topology_display = ctk.CTkFrame(self.topology_tab)
        topology_display.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.topology_text = ctk.CTkTextbox(topology_display)
        self.topology_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def on_blockchain_change(self, blockchain_name):
        """Handle blockchain selection change"""
        if blockchain_name in self.blockchains:
            self.current_blockchain = self.blockchains[blockchain_name]
            self.update_user_dropdown()
            self.update_all_displays()
    
    def on_user_change(self, username):
        """Handle user selection change"""
        if self.current_blockchain:
            for user in self.current_blockchain.users.values():
                if user.name == username:
                    self.current_user = user
                    break
    
    def update_user_dropdown(self):
        """Update user dropdown based on current blockchain"""
        if self.current_blockchain:
            user_names = [user.name for user in self.current_blockchain.users.values()]
            self.user_dropdown.configure(values=user_names)
            if user_names:
                self.user_dropdown.set(user_names[0])
                self.on_user_change(user_names[0])
    
    def update_protocol_dropdown(self):
        """Update DeFi protocol dropdown"""
        if self.current_blockchain:
            protocol_names = list(self.current_blockchain.defi_protocols.keys())
            if hasattr(self, 'protocol_var'):
                # Find the protocol combobox and update it
                protocol_combo = None
                for widget in self.defi_tab.winfo_children():
                    if isinstance(widget, ctk.CTkFrame):
                        for subwidget in widget.winfo_children():
                            if isinstance(subwidget, ctk.CTkFrame):
                                for control in subwidget.winfo_children():
                                    if isinstance(control, ctk.CTkComboBox) and control.cget("variable") == self.protocol_var:
                                        protocol_combo = control
                                        break
                if protocol_combo:
                    protocol_combo.configure(values=protocol_names)
                    if protocol_names:
                        self.protocol_var.set(protocol_names[0])
    
    def start_simulation(self):
        """Start background simulation"""
        self.simulation_running = True
        self.simulation_thread = threading.Thread(target=self.simulation_loop, daemon=True)
        self.simulation_thread.start()
    
    def simulation_loop(self):
        """Main simulation loop"""
        while self.simulation_running:
            try:
                if self.current_blockchain:
                    # Generate random transactions
                    if random.random() < 0.3:  # 30% chance each cycle
                        self.generate_random_transaction()
                    
                    # Mine blocks if there are pending transactions
                    if len(self.current_blockchain.pending_transactions) > 0 and random.random() < 0.1:
                        validator = self.current_blockchain.select_validator_by_consensus()
                        if validator:
                            block = self.current_blockchain.mine_advanced_block(validator)
                            if block:
                                self.log_network_activity(f"Block {block.index} mined by {validator[:10]}...")
                    
                    # Update displays periodically
                    if random.random() < 0.2:  # 20% chance
                        self.root.after(0, self.update_all_displays)
                
                time.sleep(2)  # Wait 2 seconds between cycles
            except Exception as e:
                print(f"Simulation error: {e}")
                time.sleep(1)
    
    def generate_random_transaction(self):
        """Generate random transaction for simulation"""
        if not self.current_blockchain or len(self.current_blockchain.users) < 2:
            return
        
        users = list(self.current_blockchain.users.values())
        sender = random.choice(users)
        receiver = random.choice([u for u in users if u != sender])
        
        amount = random.uniform(0.1, 10.0)
        tx_type = random.choice(list(TransactionType))
        priority_fee = random.uniform(0.001, 0.01)
        
        tx = AdvancedTransaction(
            sender.address,
            receiver.address,
            amount,
            tx_type,
            priority_fee
        )
        
        tx.signature = sender.sign_transaction(tx)
        self.current_blockchain.add_transaction(tx)
    
    def toggle_simulation(self):
        """Toggle simulation on/off"""
        self.simulation_running = not self.simulation_running
        if self.simulation_running:
            self.simulation_btn.configure(text="⏸️ Pause Simulation")
            if not hasattr(self, 'simulation_thread') or not self.simulation_thread.is_alive():
                self.simulation_thread = threading.Thread(target=self.simulation_loop, daemon=True)
                self.simulation_thread.start()
        else:
            self.simulation_btn.configure(text="▶️ Resume Simulation")
    
    def log_network_activity(self, message):
        """Log network activity to live feed"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        def update_feed():
            if hasattr(self, 'live_feed'):
                self.live_feed.insert("end", log_message)
                self.live_feed.see("end")
                
                # Keep only last 50 lines
                lines = self.live_feed.get("1.0", "end").split('\n')
                if len(lines) > 50:
                    self.live_feed.delete("1.0", f"{len(lines)-50}.0")
        
        if self.simulation_running:
            self.root.after(0, update_feed)
    
    def update_all_displays(self):
        """Update all display components"""
        if not self.current_blockchain:
            return
        
        # Update network metrics
        self.update_network_metrics()
        
        # Update performance display
        self.update_performance_display()
        
        # Update validators display
        self.update_validators_display()
        
        # Update DeFi metrics
        self.update_defi_metrics()
        
        # Update MEV metrics
        self.update_mev_metrics()
        
        # Update analytics
        self.update_analytics_display()
    
    def update_network_metrics(self):
        """Update network performance metrics"""
        if not self.current_blockchain:
            return
        
        stats = self.current_blockchain.network_stats
        
        metrics_update = {
            "TPS": f"{stats.tps:.2f}",
            "Latency": f"{stats.latency:.0f}ms",
            "Gas Price": f"{stats.gas_price:.6f}",
            "Validators": str(stats.active_validators),
            "TVL": f"${sum(protocol.tvl for protocol in self.current_blockchain.defi_protocols.values()):,.0f}",
            "Volume 24h": f"${sum(protocol.volume_24h for protocol in self.current_blockchain.defi_protocols.values()):,.0f}"
        }
        
        for metric, value in metrics_update.items():
            if metric in self.metric_labels:
                self.metric_labels[metric].configure(text=value)
    
    def update_performance_display(self):
        """Update performance metrics display"""
        if not hasattr(self, 'performance_text') or not self.current_blockchain:
            return
        
        stats = self.current_blockchain.network_stats
        econ = self.current_blockchain.economic_metrics
        
        performance_info = f"""📊 NETWORK PERFORMANCE METRICS
{'='*50}

🔗 Blockchain: {self.current_blockchain.name}
🤝 Consensus: {self.current_blockchain.consensus_type.value}
🕸️ Topology: {self.current_blockchain.network_topology.value}

⚡ Performance:
• TPS: {stats.tps:.2f}
• Block Time: {stats.block_time:.1f}s
• Latency: {stats.latency:.0f}ms
• Finality Time: {stats.finality_time:.1f}s

💰 Economics:
• Total Supply: {econ.total_supply:,.0f}
• Circulating: {econ.circulating_supply:,.0f}
• Inflation Rate: {econ.inflation_rate:.2%}
• Transaction Fees: {econ.transaction_fees:.3f}

⛏️ Mining/Validation:
• Active Validators: {stats.active_validators}
• Total Stake: {stats.total_stake:,.0f}
• Network Hashrate: {stats.network_hashrate:,.0f} H/s

📋 Mempool:
• Pending Transactions: {stats.mempool_size}
• Average Gas Price: {stats.gas_price:.6f}

🌐 Network:
• Active Nodes: {len(self.current_blockchain.nodes)}
• Online Nodes: {sum(1 for node in self.current_blockchain.nodes.values() if node.is_online)}
• Total Blocks: {len(self.current_blockchain.chain)}
"""
        
        self.performance_text.delete("1.0", "end")
        self.performance_text.insert("1.0", performance_info)
    
    def update_validators_display(self):
        """Update validators and delegators display"""
        if not hasattr(self, 'validators_text') or not self.current_blockchain:
            return
        
        validators_info = "👥 VALIDATORS & DELEGATORS\n" + "="*50 + "\n\n"
        
        for addr, validator in self.current_blockchain.validators.items():
            validators_info += f"""🏛️ Validator: {addr[:10]}...
   💰 Self Stake: {validator.stake:.2f}
   🤝 Delegated: {validator.delegated_stake:.2f}
   💎 Total Stake: {validator.total_stake():.2f}
   💼 Commission: {validator.commission:.1%}
   ⚡ Uptime: {validator.uptime:.1%}
   🏆 Blocks: {validator.blocks_validated}
   💵 Rewards: {validator.rewards_earned:.3f}
   ⭐ Reputation: {validator.reputation:.1f}
   📅 Last Active: {validator.last_active.strftime('%H:%M:%S')}

"""
        
        # Add delegation info
        if validators_info == "👥 VALIDATORS & DELEGATORS\n" + "="*50 + "\n\n":
            validators_info += "No validators active. Add validators to start consensus.\n"
        
        self.validators_text.delete("1.0", "end")
        self.validators_text.insert("1.0", validators_info)
    
    def update_defi_metrics(self):
        """Update DeFi metrics display"""
        if not hasattr(self, 'defi_metrics_text') or not self.current_blockchain:
            return
        
        defi_info = "🏦 DEFI PROTOCOL METRICS\n" + "="*50 + "\n\n"
        
        total_tvl = 0
        total_volume = 0
        total_fees = 0
        
        for protocol in self.current_blockchain.defi_protocols.values():
            defi_info += f"""📊 {protocol.name} ({protocol.protocol_type})
   💰 TVL: ${protocol.tvl:,.2f}
   📈 24h Volume: ${protocol.volume_24h:,.2f}
   💵 Fees Collected: ${protocol.fees_collected:.3f}
   🏊 Liquidity Pools: {len(protocol.liquidity_pools)}
   🏛️ Treasury: ${protocol.treasury:,.2f}

"""
            total_tvl += protocol.tvl
            total_volume += protocol.volume_24h
            total_fees += protocol.fees_collected
        
        if not self.current_blockchain.defi_protocols:
            defi_info += "No DeFi protocols deployed.\n"
        else:
            defi_info += f"""📊 TOTAL ECOSYSTEM:
   💰 Total TVL: ${total_tvl:,.2f}
   📈 Total Volume: ${total_volume:,.2f}
   💵 Total Fees: ${total_fees:.3f}
"""
        
        self.defi_metrics_text.delete("1.0", "end")
        self.defi_metrics_text.insert("1.0", defi_info)
    
    def update_mev_metrics(self):
        """Update MEV metrics display"""
        if not hasattr(self, 'mev_metrics_text') or not self.current_blockchain:
            return
        
        mev_info = "⚡ MEV BOT STATISTICS\n" + "="*50 + "\n\n"
        
        total_mev = 0
        total_extractions = 0
        
        for bot in self.current_blockchain.mev_bots:
            mev_info += f"""🤖 {bot.name} ({bot.strategy})
   💰 Total Profit: {bot.profit:.3f}
   ⚡ Extractions: {bot.transactions_extracted}
   🟢 Status: {'Active' if bot.active else 'Inactive'}

"""
            total_mev += bot.profit
            total_extractions += bot.transactions_extracted
        
        # Add recent MEV blocks
        mev_blocks = [block for block in self.current_blockchain.chain if block.mev_rewards > 0]
        if mev_blocks:
            mev_info += f"\n📊 RECENT MEV ACTIVITY:\n"
            for block in mev_blocks[-5:]:  # Last 5 MEV blocks
                mev_info += f"   Block {block.index}: {block.mev_rewards:.3f} MEV extracted\n"
        
        mev_info += f"""

📊 TOTAL MEV METRICS:
   💰 Total MEV Extracted: {total_mev:.3f}
   ⚡ Total Extractions: {total_extractions}
   📈 MEV per Block: {total_mev / max(len(self.current_blockchain.chain), 1):.3f}
"""
        
        self.mev_metrics_text.delete("1.0", "end")
        self.mev_metrics_text.insert("1.0", mev_info)
    
    def update_analytics_display(self):
        """Update analytics display"""
        if not hasattr(self, 'analytics_text') or not self.current_blockchain:
            return
        
        # Calculate advanced analytics
        blocks = self.current_blockchain.chain
        if len(blocks) < 2:
            return
        
        # Block time analysis
        block_times = []
        for i in range(1, len(blocks)):
            time_diff = (datetime.fromisoformat(blocks[i].timestamp) - 
                        datetime.fromisoformat(blocks[i-1].timestamp)).total_seconds()
            block_times.append(time_diff)
        
        avg_block_time = np.mean(block_times) if block_times else 0
        block_time_std = np.std(block_times) if block_times else 0
        
        # Transaction analysis
        total_txs = sum(len(block.transactions) for block in blocks)
        avg_txs_per_block = total_txs / len(blocks) if blocks else 0
        
        # Gas analysis
        gas_data = [tx.gas_used for block in blocks for tx in block.transactions if tx.gas_used > 0]
        avg_gas = np.mean(gas_data) if gas_data else 0
        
        analytics_info = f"""📊 ADVANCED BLOCKCHAIN ANALYTICS
{'='*60}

⏱️ TIMING ANALYSIS:
   Average Block Time: {avg_block_time:.2f}s
   Block Time Std Dev: {block_time_std:.2f}s
   Target Block Time: {self.current_blockchain.network_stats.block_time:.1f}s
   Time Variance: {(block_time_std / avg_block_time * 100) if avg_block_time > 0 else 0:.1f}%

📊 TRANSACTION ANALYSIS:
   Total Transactions: {total_txs:,}
   Avg Txs per Block: {avg_txs_per_block:.2f}
   Current TPS: {self.current_blockchain.network_stats.tps:.2f}
   Theoretical Max TPS: {30000000 / avg_gas if avg_gas > 0 else 0:.2f}

⛽ GAS ANALYSIS:
   Average Gas Used: {avg_gas:,.0f}
   Gas Utilization: {(avg_gas / 30000000 * 100) if avg_gas > 0 else 0:.1f}%
   Current Gas Price: {self.current_blockchain.network_stats.gas_price:.6f}

🏦 ECONOMIC METRICS:
   Market Cap: ${self.current_blockchain.economic_metrics.market_cap:,.0f}
   Staking Ratio: {(self.current_blockchain.network_stats.total_stake / self.current_blockchain.economic_metrics.circulating_supply * 100) if self.current_blockchain.economic_metrics.circulating_supply > 0 else 0:.1f}%
   Inflation Rate: {self.current_blockchain.economic_metrics.inflation_rate:.2%}
   Fee Burn Rate: {self.current_blockchain.economic_metrics.burned_tokens:.3f}/block

🌐 NETWORK HEALTH:
   Decentralization Index: {self.calculate_decentralization_index():.2f}
   Network Security: {self.calculate_security_score():.1f}/100
   Liveness Score: {self.calculate_liveness_score():.1f}/100
   Node Distribution: {self.calculate_node_distribution():.2f}

🔍 CONSENSUS METRICS:
   Validator Participation: {self.calculate_validator_participation():.1f}%
   Slashing Events: {sum(1 for v in self.current_blockchain.validators.values() if v.slashed_amount > 0)}
   Fork Rate: {self.calculate_fork_rate():.3f}%
"""
        
        self.analytics_text.delete("1.0", "end")
        self.analytics_text.insert("1.0", analytics_info)
    
    def calculate_decentralization_index(self) -> float:
        """Calculate network decentralization index"""
        if not self.current_blockchain.validators:
            return 0.0
        
        # Nakamoto coefficient approximation
        total_stake = sum(v.total_stake() for v in self.current_blockchain.validators.values())
        if total_stake == 0:
            return 0.0
        
        stakes = sorted([v.total_stake() for v in self.current_blockchain.validators.values()], reverse=True)
        cumulative_stake = 0
        validators_needed = 0
        
        for stake in stakes:
            cumulative_stake += stake
            validators_needed += 1
            if cumulative_stake > total_stake * 0.51:
                break
        
        return min(validators_needed / len(self.current_blockchain.validators), 1.0)
    
    def calculate_security_score(self) -> float:
        """Calculate network security score"""
        factors = []
        
        # Validator count factor
        validator_count = len(self.current_blockchain.validators)
        factors.append(min(validator_count / 100, 1.0) * 30)  # 30 points max
        
        # Stake distribution factor
        decentralization = self.calculate_decentralization_index()
        factors.append(decentralization * 25)  # 25 points max
        
        # Network size factor
        node_count = len(self.current_blockchain.nodes)
        factors.append(min(node_count / 1000, 1.0) * 25)  # 25 points max
        
        # Consensus strength factor
        if self.current_blockchain.consensus_type in [ConsensusType.PRACTICAL_BFT, ConsensusType.TENDERMINT]:
            factors.append(20)  # BFT consensus gets full points
        else:
            factors.append(15)  # Other consensus gets partial points
        
        return sum(factors)
    
    def calculate_liveness_score(self) -> float:
        """Calculate network liveness score"""
        if not self.current_blockchain.validators:
            return 0.0
        
        # Average validator uptime
        avg_uptime = np.mean([v.uptime for v in self.current_blockchain.validators.values()])
        
        # Block production consistency
        block_consistency = 1.0  # Simplified for demo
        
        # Network connectivity
        online_nodes = sum(1 for node in self.current_blockchain.nodes.values() if node.is_online)
        connectivity = online_nodes / max(len(self.current_blockchain.nodes), 1)
        
        return (avg_uptime * 40 + block_consistency * 30 + connectivity * 30)
    
    def calculate_node_distribution(self) -> float:
        """Calculate geographical node distribution score"""
        if not self.current_blockchain.nodes:
            return 0.0
        
        # Simplified geographical distribution calculation
        locations = [node.location for node in self.current_blockchain.nodes.values()]
        if not locations:
            return 0.0
        
        # Calculate variance in locations (higher = more distributed)
        lats = [loc[0] for loc in locations]
        lons = [loc[1] for loc in locations]
        
        lat_var = np.var(lats) if lats else 0
        lon_var = np.var(lons) if lons else 0
        
        # Normalize to 0-1 scale
        return min((lat_var + lon_var) / 10000, 1.0)
    
    def calculate_validator_participation(self) -> float:
        """Calculate validator participation rate"""
        if not self.current_blockchain.validators:
            return 0.0
        
        active_validators = sum(1 for v in self.current_blockchain.validators.values() if v.is_active)
        return (active_validators / len(self.current_blockchain.validators)) * 100
    
    def calculate_fork_rate(self) -> float:
        """Calculate blockchain fork rate"""
        # Simplified fork rate calculation
        return random.uniform(0, 0.1)  # 0-0.1% fork rate
    
    # Action methods for GUI interactions
    def on_consensus_change(self, consensus_type):
        """Handle consensus type change"""
        if self.current_blockchain:
            consensus_map = {
                "PoW": ConsensusType.PROOF_OF_WORK,
                "PoS": ConsensusType.PROOF_OF_STAKE,
                "DPoS": ConsensusType.DELEGATED_PROOF_OF_STAKE,
                "pBFT": ConsensusType.PRACTICAL_BFT,
                "Tendermint": ConsensusType.TENDERMINT,
                "PoA": ConsensusType.PROOF_OF_AUTHORITY,
                "PoH": ConsensusType.PROOF_OF_HISTORY
            }
            self.current_blockchain.consensus_type = consensus_map.get(consensus_type, ConsensusType.PROOF_OF_STAKE)
            self.log_network_activity(f"Consensus changed to {consensus_type}")
    
    def start_consensus(self):
        """Start consensus mechanism"""
        if self.current_blockchain:
            self.mining_active = True
            self.log_network_activity("Consensus mechanism started")
    
    def stop_consensus(self):
        """Stop consensus mechanism"""
        self.mining_active = False
        self.log_network_activity("Consensus mechanism stopped")
    
    def execute_defi_action(self):
        """Execute DeFi protocol action"""
        if not self.current_blockchain or not self.protocol_var.get():
            messagebox.showerror("Error", "Please select a protocol")
            return
        
        protocol_name = self.protocol_var.get()
        action = self.defi_action_var.get()
        
        if protocol_name in self.current_blockchain.defi_protocols:
            protocol = self.current_blockchain.defi_protocols[protocol_name]
            
            if action == "Add Liquidity":
                lp_tokens = protocol.add_liquidity("ETH", "USDC", 10.0, 20000.0)
                self.log_network_activity(f"Added liquidity to {protocol_name}, received {lp_tokens:.2f} LP tokens")
            
            elif action == "Swap":
                amount_out = protocol.swap("ETH", "USDC", 1.0)
                self.log_network_activity(f"Swapped 1 ETH for {amount_out:.2f} USDC on {protocol_name}")
            
            messagebox.showinfo("Success", f"DeFi action '{action}' executed on {protocol_name}")
            self.update_defi_metrics()
    
    def bridge_asset(self):
        """Execute cross-chain bridge transaction"""
        from_chain = self.from_chain_var.get()
        to_chain = self.to_chain_var.get()
        asset = self.bridge_asset_entry.get()
        amount = float(self.bridge_amount_entry.get() or "0")
        
        if not all([from_chain, to_chain, asset, amount]):
            messagebox.showerror("Error", "Please fill all fields")
            return
        
        bridge_id = f"{from_chain}_{to_chain}"
        if bridge_id in self.current_blockchain.cross_chain_bridges:
            bridge = self.current_blockchain.cross_chain_bridges[bridge_id]
            result = bridge.bridge_asset(asset, amount, from_chain, to_chain)
            
            if result['success']:
                self.log_network_activity(f"Bridged {amount} {asset} from {from_chain} to {to_chain}")
                messagebox.showinfo("Success", f"Bridge successful! Confirmation time: {result['confirmation_time']} minutes")
            else:
                messagebox.showerror("Error", result.get('error', 'Bridge failed'))
        else:
            messagebox.showerror("Error", "Bridge not found between selected chains")
    
    def add_mev_bot(self):
        """Add new MEV bot"""
        if self.current_blockchain:
            bot_name = f"MEVBot_{len(self.current_blockchain.mev_bots) + 1}"
            strategy = random.choice(["arbitrage", "sandwich", "liquidation"])
            bot = MEVBot(bot_name, strategy)
            self.current_blockchain.mev_bots.append(bot)
            self.log_network_activity(f"Added MEV bot: {bot_name} with {strategy} strategy")
            messagebox.showinfo("Success", f"MEV bot {bot_name} added")
    
    def trigger_mev(self):
        """Trigger MEV opportunity"""
        if self.current_blockchain and self.current_user:
            # Create high-value transaction to trigger MEV
            users = list(self.current_blockchain.users.values())
            if len(users) >= 2:
                receiver = random.choice([u for u in users if u != self.current_user])
                
                tx = AdvancedTransaction(
                    self.current_user.address,
                    receiver.address,
                    500.0,  # High value to trigger MEV
                    TransactionType.TRANSFER,
                    0.01
                )
                tx.signature = self.current_user.sign_transaction(tx)
                
                if self.current_blockchain.add_transaction(tx):
                    self.log_network_activity("High-value transaction created - MEV opportunity detected")
                    messagebox.showinfo("Success", "MEV opportunity triggered!")
    
    def analyze_mev(self):
        """Analyze MEV patterns"""
        self.log_network_activity("MEV analysis started")
        self.update_mev_metrics()
        messagebox.showinfo("Analysis", "MEV analysis completed - check MEV tab for details")
    
    def create_proposal(self):
        """Create governance proposal"""
        proposal_text = self.proposal_entry.get()
        if not proposal_text:
            messagebox.showerror("Error", "Please enter proposal text")
            return
        
        if self.current_blockchain:
            proposal = {
                'id': len(self.current_blockchain.governance_proposals) + 1,
                'text': proposal_text,
                'proposer': self.current_user.address if self.current_user else "Unknown",
                'votes_for': 0,
                'votes_against': 0,
                'created_at': datetime.now().isoformat(),
                'status': 'active'
            }
            
            self.current_blockchain.governance_proposals.append(proposal)
            self.log_network_activity(f"Governance proposal created: {proposal_text}")
            self.proposal_entry.delete(0, "end")
            messagebox.showinfo("Success", "Proposal created successfully!")
            self.update_governance_display()
    
    def update_governance_display(self):
        """Update governance proposals display"""
        if not hasattr(self, 'governance_text') or not self.current_blockchain:
            return
        
        governance_info = "🗳️ GOVERNANCE PROPOSALS\n" + "="*50 + "\n\n"
        
        for proposal in self.current_blockchain.governance_proposals:
            governance_info += f"""📋 Proposal #{proposal['id']}
   📝 Text: {proposal['text']}
   👤 Proposer: {proposal['proposer'][:10]}...
   ✅ Votes For: {proposal['votes_for']}
   ❌ Votes Against: {proposal['votes_against']}
   📅 Created: {proposal['created_at'][:19]}
   🔄 Status: {proposal['status']}

"""
        
        if not self.current_blockchain.governance_proposals:
            governance_info += "No active proposals. Create a proposal above!\n"
        
        self.governance_text.delete("1.0", "end")
        self.governance_text.insert("1.0", governance_info)
    
    def generate_analytics_report(self):
        """Generate comprehensive analytics report"""
        self.log_network_activity("Generating comprehensive analytics report...")
        self.update_analytics_display()
        messagebox.showinfo("Report", "Analytics report generated successfully!")
    
    def export_analytics_data(self):
        """Export analytics data"""
        messagebox.showinfo("Export", "Analytics data exported to blockchain_analytics.json")
        self.log_network_activity("Analytics data exported")
    
    def refresh_analytics(self):
        """Refresh analytics display"""
        self.update_analytics_display()
        self.log_network_activity("Analytics refreshed")
    
    def change_topology(self, topology_name):
        """Change network topology"""
        if self.current_blockchain:
            topology_map = {
                "Mesh": NetworkTopology.MESH,
                "Star": NetworkTopology.STAR,
                "Ring": NetworkTopology.RING,
                "Tree": NetworkTopology.TREE,
                "Small World": NetworkTopology.SMALL_WORLD
            }
            
            self.current_blockchain.network_topology = topology_map.get(topology_name, NetworkTopology.SMALL_WORLD)
            self.current_blockchain.setup_network_topology()
            self.log_network_activity(f"Network topology changed to {topology_name}")
            self.update_topology_display()
    
    def simulate_network_effect(self):
        """Simulate network effects"""
        if self.current_blockchain:
            # Simulate network partition
            offline_nodes = random.sample(list(self.current_blockchain.nodes.keys()), 
                                        min(3, len(self.current_blockchain.nodes)))
            
            for node_id in offline_nodes:
                self.current_blockchain.nodes[node_id].is_online = False
            
            self.log_network_activity(f"Network effect simulated: {len(offline_nodes)} nodes went offline")
            self.update_topology_display()
            
            # Restore nodes after 10 seconds
            def restore_nodes():
                time.sleep(10)
                for node_id in offline_nodes:
                    if node_id in self.current_blockchain.nodes:
                        self.current_blockchain.nodes[node_id].is_online = True
                self.log_network_activity("Network nodes restored")
            
            threading.Thread(target=restore_nodes, daemon=True).start()
    
    def update_topology_display(self):
        """Update network topology display"""
        if not hasattr(self, 'topology_text') or not self.current_blockchain:
            return
        
        topology_info = f"""🕸️ NETWORK TOPOLOGY ANALYSIS
{'='*50}

🌐 Current Topology: {self.current_blockchain.network_topology.value}
📊 Total Nodes: {len(self.current_blockchain.nodes)}
🟢 Online Nodes: {sum(1 for node in self.current_blockchain.nodes.values() if node.is_online)}
🔴 Offline Nodes: {sum(1 for node in self.current_blockchain.nodes.values() if not node.is_online)}

📡 NODE DETAILS:
"""
        
        for node_id, node in list(self.current_blockchain.nodes.items())[:10]:  # Show first 10 nodes
            status = "🟢 Online" if node.is_online else "🔴 Offline"
            topology_info += f"""
🖥️ {node_id}:
   📍 Location: ({node.location[0]:.2f}, {node.location[1]:.2f})
   🔗 Peers: {len(node.peers)}
   📶 Status: {status}
   💾 Type: {node.node_type}
   📡 Bandwidth: {node.bandwidth} Mbps
"""
        
        # Add network metrics
        avg_latency = np.mean([np.mean(list(node.latency_map.values())) for node in self.current_blockchain.nodes.values() if node.latency_map])
        topology_info += f"""

📊 NETWORK METRICS:
• Average Latency: {avg_latency:.2f}ms
• Network Diameter: {self.calculate_network_diameter():.1f}
• Clustering Coefficient: {self.calculate_clustering_coefficient():.3f}
• Connectivity Index: {self.calculate_connectivity_index():.3f}
"""
        
        self.topology_text.delete("1.0", "end")
        self.topology_text.insert("1.0", topology_info)
    
    def calculate_network_diameter(self) -> float:
        """Calculate network diameter (simplified)"""
        return random.uniform(3, 8)  # Simplified for demo
    
    def calculate_clustering_coefficient(self) -> float:
        """Calculate clustering coefficient (simplified)"""
        return random.uniform(0.1, 0.9)  # Simplified for demo
    
    def calculate_connectivity_index(self) -> float:
        """Calculate network connectivity index"""
        if not self.current_blockchain.nodes:
            return 0.0
        
        online_nodes = sum(1 for node in self.current_blockchain.nodes.values() if node.is_online)
        total_nodes = len(self.current_blockchain.nodes)
        
        return online_nodes / total_nodes if total_nodes > 0 else 0.0
    
    def run(self):
        """Start the advanced application"""
        self.root.mainloop()

def main():
    """Main function to run the advanced application"""
    try:
        print("🚀 Starting Advanced Blockchain Network Simulator...")
        app = AdvancedBlockchainSimulator()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Error", f"Application failed to start: {e}")

if __name__ == "__main__":
    main()