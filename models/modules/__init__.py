"""RootSAM-Pro modules."""

from .memory_bank import MemoryBank

# PP-FSRD++ core (phase-preserving Fourier domain washer).
from .ppfsrd_core import FourierWashingUnit

# from .sam2_compat import Sam2Compat

from .bhfi import BHFI
from .bkmc import apply_delta_trust_ceiling

# Router + dual-bank state manager
from .memory_router import MemoryRouter
from .dual_memory_bank import DualMemoryBank

from .memory_packer import pack_memory_bank
from .sam2_decode_compat import decode_masks_compat

# Memory attention wrapper (signature + runtime safety)
from .memory_attn_compat import memory_attention_compat

# Dimension radar utilities
from .memory_dim_radar import MemoryDims, detect_memory_dims, detect_kv_in_dim, detect_mem_store_dim, detect_ptr_dim, has_official_obj_ptr_proj, get_official_obj_ptr_proj
