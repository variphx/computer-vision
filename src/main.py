import torch
from torch import nn
from schema.model import AslClassifier
from schema.trainer import Trainer
from schema.dataset import AslDataset
from sys import argv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device={device}")

train_dir = "/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
dataset = AslDataset(train_dir, device=device)
dataloader = dataset.dataloader(batch_size=512, shuffle=True)

asl_model = AslClassifier((3, 128, 128), 29, 3, device=device)
optimizer = torch.optim.AdamW(asl_model.parameters())
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

loss_fn = nn.CrossEntropyLoss()

trainer = Trainer(
    model=asl_model,
    dataloader=dataloader,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    loss_fn=loss_fn,
    epochs=5,
    device=device,
)

trainer.train()
model_path = "/kaggle/working/model.pth"
trainer.save_model(model_path)
