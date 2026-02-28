# RootSAM-Pro adapter collection (Tri-Adapter paradigm)
from .adapter_utils import VanillaAdapter, VanillaKMRAdapter, SRDSafeWrapper
from .sfa_adapter import SFAAdapter
from .asta_adapter import ASTAAdapter
from .pra_adapter import PRAAdapter

# Existing task-specific adapters (kept for compatibility)
from .srd_adapter import SRDAdapter
from .kmr_adapter import KMRAdapter
