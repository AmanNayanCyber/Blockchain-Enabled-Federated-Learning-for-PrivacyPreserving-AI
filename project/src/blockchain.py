import hashlib
import json
from datetime import datetime


class BlockchainLedger:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = {
            "index": 0,
            "timestamp": str(datetime.utcnow()),
            "client_id": "genesis",
            "round": 0,
            "update_hash": "0",
            "previous_hash": "0"
        }
        genesis_block["block_hash"] = self.compute_hash(genesis_block)
        self.chain.append(genesis_block)

    def compute_hash(self, block):
        block_copy = block.copy()
        block_copy.pop("block_hash", None)
        block_string = json.dumps(block_copy, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def add_block(self, client_id, round_num, update_state_dict):
        update_string = json.dumps(
            {k: v.tolist() if hasattr(v, "tolist") else str(v) for k, v in update_state_dict.items()},
            sort_keys=True
        ).encode()
        update_hash = hashlib.sha256(update_string).hexdigest()

        previous_block = self.chain[-1]
        block = {
            "index": len(self.chain),
            "timestamp": str(datetime.utcnow()),
            "client_id": client_id,
            "round": round_num,
            "update_hash": update_hash,
            "previous_hash": previous_block["block_hash"]
        }
        block["block_hash"] = self.compute_hash(block)
        self.chain.append(block)
        return block

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            prev = self.chain[i - 1]

            if current["previous_hash"] != prev["block_hash"]:
                return False

            if current["block_hash"] != self.compute_hash(current):
                return False

        return True
