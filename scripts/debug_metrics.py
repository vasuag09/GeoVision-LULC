import torch
from src.training.metrics import Evaluator

evaluator = Evaluator(num_classes=7)

# Dummy test matching the diagnostic output from previous tool call
true_mask = torch.tensor([0, 1, 2, 3, 4, 5, 6])
pred_mask = torch.tensor([0, 2, 3, 3, 4, 5, 6])

evaluator.update(pred_mask, true_mask)
print(evaluator.get_metrics())
