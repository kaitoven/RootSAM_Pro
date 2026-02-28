确认模块启用与 trainable：
python - << 'PY'
import importlib, sys
sys.path.insert(0,".")
mod = importlib.import_module("configs.root_sam_pro_cfg")
cfg = getattr(mod, "Config")
from models.root_sam_pro import RootSAMPro

for mode in ["SFA_ONLY","SFA_ASTA","FULL"]:
    cfg.ABLATION_MODE = mode
    m = RootSAMPro(cfg).cuda()
    trainable = [(n,p.numel()) for n,p in m.named_parameters() if p.requires_grad]
    total = sum(k for _,k in trainable)
    srd_img = sum(1 for n,_ in trainable if ".mlp.adapter." in n or "srd_img_blocks" in n)
    asta = sum(1 for n,_ in trainable if n.startswith("asta."))
    pra = sum(1 for n,_ in trainable if n.startswith("pra."))
    print(f"\n[{mode}] trainable={total:,} | srd_img_tensors={srd_img} | asta_tensors={asta} | pra_tensors={pra}")
    for n,k in sorted(trainable, key=lambda x:-x[1])[:8]:
        print(" ", n, k)
PY

消融实验指令：
SFA_ONLY
python main.py \
  --subset Papaya_736x552_DPI150 \
  --ablation SFA_ONLY \
  --run_tag abla_sfa_only_fair_s0 \
  --set TRAIN_MODE=SEQUENCE \
  --set TBPTT_CHUNK=0 \
  --set TRAIN_MAX_SEQ_LEN=0 \
  --set EPOCHS=30 \
  --set BATCH_SIZE=4 \
  --set LR=1e-4 \
  --set AMP=1 \
  --set USE_TASK_UNCERTAINTY=0 \

  --set SOIL_PENALTY_ALL_MODES=1 \
  --set SOIL_TOPK_RATIO=0.03 \
  --set SOIL_LAMBDA_MAX=50

SFA_ONLY
python main.py \
  --subset Papaya_736x552_DPI150 \
  --ablation SFA_ASTA \
  --run_tag abla_sfa_asta_fair_s0 \
  --set TRAIN_MODE=SEQUENCE \
  --set TBPTT_CHUNK=4 \
  --set TRAIN_MAX_SEQ_LEN=0 \
  --set EPOCHS=30 \
  --set BATCH_SIZE=4 \
  --set LR=1e-4 \
  --set AMP=1 \
  --set USE_TASK_UNCERTAINTY=0 \
  --set SOIL_PENALTY_ALL_MODES=1 \
  --set SOIL_TOPK_RATIO=0.03 \
  --set SOIL_LAMBDA_MAX=50

FULL
python main.py \
  --subset Papaya_736x552_DPI150 \
  --ablation FULL \
  --run_tag abla_full_fair_s0 \
  --set TRAIN_MODE=SEQUENCE \
  --set TBPTT_CHUNK=4 \
  --set TRAIN_MAX_SEQ_LEN=0 \
  --set AMP=1 \
  --set USE_TASK_UNCERTAINTY=0 \
  --set SOIL_PENALTY_ALL_MODES=1 \
  --set SOIL_TOPK_RATIO=0.03 \
  --set SOIL_LAMBDA_MAX=50

  --set EPOCHS=30 \
  --set BATCH_SIZE=4 \
  --set LR=1e-4 \

2GPU分开跑两个子集：
chmod +x /root/autodl-tmp/RootSAM_Pro/scripts/run_two_subsets_2gpus.sh
nohup ./scripts/run_two_subsets_2gpus.sh > /root/autodl-tmp/logs_rootsam_pro/ablation_master_s0.log 2>&1 &