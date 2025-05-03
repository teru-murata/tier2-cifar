import sys, os, torch, pprint
path = sys.argv[1] if len(sys.argv) > 1 else "save/best_model.pth"

if not os.path.isfile(path):
    raise FileNotFoundError(path)
ckpt = torch.load(path, weights_only=False) 
print("***checkpoint keys:")
pprint.pprint(ckpt.keys())

print("\***nquick stats:")
print("iter_idx     :", ckpt.get("iter_idx"))
print("best_va_acc  :", ckpt.get("best_va_acc"))
print("state_dict # :", len(ckpt.get("model", {})))
print("\nexample parames:")
for i, name in enumerate(ckpt.get("model", {}).keys()):
    if i == 10: break
    print(" ", name)