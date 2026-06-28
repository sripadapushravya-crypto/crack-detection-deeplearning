import torch

ckpt = torch.load(
    r"C:\Users\shrav\Downloads\SDNET\sdnet-enhanced\models\cnn\best_model.pt",
    map_location="cpu", weights_only=False,
)
print("type:", type(ckpt))
sd = ckpt
if isinstance(ckpt, dict):
    print("top-level keys:", list(ckpt.keys())[:20])
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

if isinstance(sd, dict):
    tensors = {k: v for k, v in sd.items() if hasattr(v, "numel")}
    total = sum(v.numel() for v in tensors.values())
    print(f"num tensors: {len(tensors)}   total params: {total:,}  (~{total*4/1e6:.1f} MB)")
    keys = list(tensors.keys())
    print("first layers:", keys[:8])
    print("last layers:", keys[-6:])
    if keys:
        print("final layer shape:", tuple(tensors[keys[-1]].shape))