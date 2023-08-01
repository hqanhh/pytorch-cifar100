import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import squeezenet

def apply_pruning(model, pruning_rate):
    prune.l1_unstructured(model.stem[0], name="weight", amount=pruning_rate)
    prune.remove(model.stem[0], "weight")
    # Example: Prune the first 1x1 expansion layer in fire2 with pruning_rate
    prune.l1_unstructured(model.fire2.expand_1x1[0], name="weight", amount=pruning_rate)
    prune.remove(model.fire2.expand_1x1[0], "weight")
    # Example: Prune the 3x3 expansion layer in fire4 with pruning_rate
    prune.l1_unstructured(model.fire4.expand_3x3[0], name="weight", amount=pruning_rate)
    prune.remove(model.fire4.expand_3x3[0], "weight")
    # Continue pruning for other layers...

    # Prune the last convolutional layer (conv10) with pruning_rate
    prune.l1_unstructured(model.conv10, name="weight", amount=pruning_rate)
    prune.remove(model.conv10, "weight")



path_to_trained_weights = r"C:\Users\huynh\Downloads\pytorch-cifar100\squeezenet-10-regular.pth"

model = squeezenet.SqueezeNet()

state_dict = torch.load(path_to_trained_weights, map_location='cpu')

# If the loaded state_dict contains 'module.' prefix (for DataParallel models), remove it
# state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# Load the state_dict to the model (ignore missing keys, as the model architecture might be different)
model.load_state_dict(state_dict, strict=False)


pruning_rate = 0.01

apply_pruning(model, pruning_rate)

torch.save(model.state_dict(), "prun1.pth")
