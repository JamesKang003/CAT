from dataclasses import dataclass


@dataclass
class CATConfig:
    # Architecture (설계 그대로)
    block_size: int = 256
    vocab_size: int = 16384
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False
