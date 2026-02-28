import torch


class MemoryBank:
    """Explicit, paper-grade memory bank wrapper for SAM2 memory attention.

    Why this exists:
      - Keep `inference_state` as a plain dict of tensors (checkpoint-safe),
        but operate on it via an explicit, well-scoped class.
      - Centralize: flush/clear, sliding window trimming, and SAM2 input adaptation.

    State schema (stored in `inference_state`):
      - output_dict: Dict[int, Tensor]        (memory features per frame)
      - obj_ptr_tks: Dict[int, Tensor]      (optional object pointer tokens per frame)
      - value_dict: Dict[int, Tensor]        (optional per-frame utility for priority packing/eviction)
      - is_mem_empty: BoolTensor[B]     (whether to bypass memory attention)
    """

    KEY_MEM = "output_dict"
    KEY_PTR = "obj_ptr_tks"
    KEY_VAL = "value_dict"
    KEY_EMPTY = "is_mem_empty"

    def __init__(self, state: dict, max_frames: int):
        self.state = state
        self.max_frames = int(max_frames)

    @staticmethod
    def bootstrap(state: dict, batch_size: int, device: torch.device) -> dict:
        """Ensure required keys exist."""
        if state is None:
            state = {}
        if MemoryBank.KEY_MEM not in state or not isinstance(state.get(MemoryBank.KEY_MEM), dict):
            state[MemoryBank.KEY_MEM] = {}
        if MemoryBank.KEY_PTR not in state or not isinstance(state.get(MemoryBank.KEY_PTR), dict):
            state[MemoryBank.KEY_PTR] = {}
        if MemoryBank.KEY_EMPTY not in state or state.get(MemoryBank.KEY_EMPTY) is None:
            state[MemoryBank.KEY_EMPTY] = torch.ones(batch_size, dtype=torch.bool, device=device)
        return state

    @property
    def mem(self) -> dict:
        return self.state[self.KEY_MEM]

    @property
    def ptr(self) -> dict:
        return self.state[self.KEY_PTR]

    @property
    def is_empty(self) -> torch.Tensor:
        return self.state[self.KEY_EMPTY]

    @is_empty.setter
    def is_empty(self, v: torch.Tensor):
        self.state[self.KEY_EMPTY] = v

    def all_empty(self) -> bool:
        try:
            return bool(self.is_empty.all().item())
        except Exception:
            return True

    def as_sam2_inputs(self):
        """Return (memory_dict, obj_ptr_tks) in the *current* SAM2-friendly format.

        Default/compatible format used in your project so far:
          memory_dict: {frame_idx: Tensor[B, C, H, W]}
          obj_ptr_tks: {frame_idx: Tensor[B, ...]}
        """
        return self.mem, self.ptr

    def apply_flush(self, flush_flags: torch.Tensor):
        """Flush history for samples where flush_flags==True.

        Implementation detail:
          - We mask out history per-sample (batch dimension) to avoid resurrecting old artifacts.
          - If *all* samples flush, we hard-clear dicts to avoid keeping all-zero junk.
        """
        if flush_flags is None:
            return
        if flush_flags.dtype != torch.bool:
            flush_flags = flush_flags.bool()

        B = int(flush_flags.shape[0])

        # mark empty for next step
        self.is_empty = flush_flags.detach().clone()

        if not flush_flags.any():
            return

        if len(self.mem) == 0 and len(self.ptr) == 0:
            return

        keep = (~flush_flags).view(B, 1, 1, 1)

        # mask memory feats
        for k in list(self.mem.keys()):
            v = self.mem[k]
            if torch.is_tensor(v) and v.dim() >= 4 and v.shape[0] == B:
                self.mem[k] = v * keep.to(dtype=v.dtype, device=v.device)

        # mask obj_ptr tokens
        for k in list(self.ptr.keys()):
            v = self.ptr[k]
            if not torch.is_tensor(v) or v.shape[0] != B:
                continue
            if v.dim() == 2:
                self.ptr[k] = v * (~flush_flags).view(B, 1).to(dtype=v.dtype, device=v.device)
            elif v.dim() == 3:
                self.ptr[k] = v * (~flush_flags).view(B, 1, 1).to(dtype=v.dtype, device=v.device)
            else:
                # fallback: broadcast along remaining dims
                shape = [B] + [1] * (v.dim() - 1)
                self.ptr[k] = v * (~flush_flags).view(*shape).to(dtype=v.dtype, device=v.device)

        # if all flush: hard clear to avoid reading all-zero history later
        if flush_flags.all():
            self.mem.clear()
            self.ptr.clear()

    def add(self, frame_idx: int, mem_feat: torch.Tensor, obj_ptr: torch.Tensor | None = None):
        """Add memory for a frame (expects mem_feat already causal-masked)."""
        self.mem[int(frame_idx)] = mem_feat
        if obj_ptr is not None:
            self.ptr[int(frame_idx)] = obj_ptr
        self.trim()

    def trim(self):
        """Keep at most `max_frames` latest entries by frame_idx."""
        if self.max_frames <= 0:
            return
        if len(self.mem) <= self.max_frames:
            return
        # drop oldest frame indices
        keys = sorted(self.mem.keys())
        while len(keys) > self.max_frames:
            k = keys.pop(0)
            if k in self.mem:
                del self.mem[k]
            if k in self.ptr:
                del self.ptr[k]
