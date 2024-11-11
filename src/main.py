import torch
import lightning as L
from .model import AslTranslator
from .data import TRAIN_DATASET


trainer = L.Trainer()
model = AslTranslator(input_dim=(3, 200, 200), output_dim=29)

train_dataset = TRAIN_DATASET.train_test_split(0.2, stratify_by_column="label")
train_dataloader = torch.utils.data.DataLoader(train_dataset["train"])
eval_dataloader = torch.utils.data.DataLoader(train_dataset["test"])

trainer.fit(model, train_dataloader, eval_dataloader)
