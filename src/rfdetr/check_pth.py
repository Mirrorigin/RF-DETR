import torch
import collections

def inspect_model(file_path):
    print(f"--- Analysis weights file: {file_path} ---")

    is_torchscript = False
    model_obj = None

    try:
        # Try loading as TorchScript very first
        model_obj = torch.jit.load(file_path, map_location="cpu")
        is_torchscript = True
        print("[File format]: TorchScript.\n")
    except Exception:
        try:
            # Retrieve to regular torch.load
            model_obj = torch.load(file_path, map_location="cpu", weights_only=False)
            is_torchscript = False
            print("[File format]: PyTorch Checkpoint.\n")
        except Exception as e:
            print(f"[Error]: Couldn't detect file format! Details: {e}")
            return

    # For regular checkpoint
    if not is_torchscript:
        print(">> Original training configuration:")
        if isinstance(model_obj, dict):
            config_keys = ['args', 'config', 'params']
            found_cfg = False
            for k in config_keys:
                if k in model_obj:
                    print(f"Found config key! '{k}': {model_obj[k]}")
                    found_cfg = True
            if not found_cfg:
                print(f"Could not find 'args' key directly, printing all keys in ckpt: {list(model_obj.keys())}")
        else:
            print("This Checkpoint save model instantiate other than dictionaries.")

    # Extract weights information (layer and dimensions)
    print("\n>> Looking for classification head weight dimensions...")
    found_class_layer = False

    if is_torchscript:
        # TorchScript uses named_parameters()
        param_iterator = model_obj.named_parameters()
    else:
        # Checkpoint usually saves this in 'model' key (or it can be dictionary itself)
        state_dict = model_obj.get('model', model_obj) if isinstance(model_obj, dict) else model_obj
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        param_iterator = state_dict.items()

    all_params = list(param_iterator)
    for name, param in all_params:
        if 'class' in name.lower() and 'weight' in name.lower():
            # Tip: Checkpoint's param is Tensor, and TorchScript's is also Tensor
            print(f"Found classification layer: {name:50} | Dimension: {list(param.shape)}")
            found_class_layer = True

    if not found_class_layer:
        print("Could not find class layer directly by name, printing the shapes of the last 10 layers for reference:")
        for name, param in all_params[-5:]:
            print(f"Layer: {name:50} | Dimension: {list(param.shape)}")


BEST_WEIGHTS = "/home/jingmliang/Projects/RF-DETR/src/rfdetr/best_models/checkpoint_bigvision_latest_v1.pt"
inspect_model(BEST_WEIGHTS)