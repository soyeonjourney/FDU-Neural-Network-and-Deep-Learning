from models import *

model = ResNet18()

# total_params = sum(p.numel() for p in model.parameters()) / 1e6
total_params = (
    sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
)  # Training params

print(total_params)
