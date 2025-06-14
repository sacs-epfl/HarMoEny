from dataclasses import dataclass, field
from typing import List, Optional, Any


@dataclass
class FetchingStrategyConfig:
    rank: int
    world_size: int
    experts: List[Any]
    cache: List[Any]
    cached_experts: List[Any]
    expert_loaded_events: Any
    cache_size: int
    num_experts: int
    first_slot_expert_idx: int
    last_slot_expert_idx: int
    buffer_expert: Optional[Any] = None
