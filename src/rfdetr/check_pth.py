import torch

BEST_WEIGHTS = "/home/jingmliang/Projects/ObjectDetection/RF-DETR/rfdetr/best_models/checkpoint_bigvision.pth"

print("Loading Checkpoint...")
# weights_only=False: bypass security restrictions and load the full namespace
ckpt = torch.load(BEST_WEIGHTS, map_location="cpu", weights_only=False)

# Print argparse.Namespace (saved in pth)
print("\nOriginal training configuration:")
if 'args' in ckpt:
    print(ckpt['args'])
elif 'config' in ckpt:
    print(ckpt['config'])
else:
    print("Could not find 'args' key directly, printing all keys in ckpt:", ckpt.keys())

state_dict = ckpt['model'] if 'model' in ckpt else ckpt

print("\nLooking for classification head weight dimensions...")
found = False
for key, value in state_dict.items():
    if 'class' in key and 'weight' in key:
        print(f"Found classification layer: {key} | Shape: {value.shape}")
        found = True

if not found:
    print("Could not find class layer directly by name, printing the shapes of the last 10 layers for reference:")
    for key in list(state_dict.keys())[-10:]:
        print(f"Layer: {key} | Shape: {state_dict[key].shape}")