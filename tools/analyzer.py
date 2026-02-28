import torch
import time
from configs.root_sam_pro_cfg import Config
from models.root_sam_pro import RootSAMPro


@torch.no_grad()
def profile_model_efficiency():
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("⏳ Building RootSAM-Pro for profiling...")
    model = RootSAMPro(cfg).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 50)
    print("🔬 RootSAM-Pro Parameter & Efficiency Analysis")
    print("=" * 50)
    print(f"🧠 Total Foundation Parameters : {total_params / 1e6:.2f} M")
    print(f"🔥 Trainable Physical Adapters : {trainable_params / 1e6:.2f} M")
    print(f"⚡ Trainable Parameter Ratio   : {(trainable_params / total_params) * 100:.3f} %")

    dummy_img = torch.randn(1, 3, cfg.TARGET_SIZE, cfg.TARGET_SIZE).to(device)
    dummy_dt = torch.tensor([1.0]).to(device)
    inf_state = {
        'output_dict': {}, 'obj_ptr_tks': {}, 'time_dict': {}, 'value_dict': {},
        'prompted_output_dict': {}, 'prompted_obj_ptr_tks': {}, 'prompted_time_dict': {}, 'prompted_value_dict': {},
        'time_days': torch.zeros(1, dtype=torch.float32, device=device),
        'prev_present': torch.zeros(1, dtype=torch.float32, device=device),
        'is_mem_empty': torch.ones(1, dtype=torch.bool, device=device),
    }

    print("\n🔥 Warming up GPU...")
    for t in range(5): model(dummy_img, dummy_dt, inf_state, t)

    print("⏱️ Running latency test...")
    torch.cuda.synchronize()
    start_time = time.time()
    iters = 50
    for t in range(iters):
        model(dummy_img, dummy_dt, inf_state, t + 5)
    torch.cuda.synchronize()

    avg_time = (time.time() - start_time) / iters
    print(f"⏱️ Inference Latency per frame: {avg_time * 1000:.2f} ms")
    print(f"🎞️ FPS (Frames Per Second)    : {1.0 / avg_time:.1f}")
    print("=" * 50)


if __name__ == "__main__":
    profile_model_efficiency()