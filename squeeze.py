import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import squeezenet

def apply_pruning(model, pruning_rate):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.random_unstructured(module, name='weight', amount=pruning_rate)
            prune.remove(module, "weight")

path_to_trained_weights = r"C:\Users\huynh\Downloads\pytorch-cifar100\squeezenet-200-regular.pth"

model = squeezenet.SqueezeNet()

state_dict = torch.load(path_to_trained_weights, map_location='cpu')

# If the loaded state_dict contains 'module.' prefix (for DataParallel models), remove it
# state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# Load the state_dict to the model (ignore missing keys, as the model architecture might be different)
model.load_state_dict(state_dict, strict=False)


pruning_rate = 0.01

apply_pruning(model, pruning_rate)

torch.save(model.state_dict(), "path_to_pruned_model.pth")
