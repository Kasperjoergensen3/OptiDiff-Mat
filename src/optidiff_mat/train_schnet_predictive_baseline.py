import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader
from matsciml.datasets import registry as ds_reg
from matsciml.models.pyg.schnet import SchNet # PyG-based SchNet
from matsciml.tasks.scalar import ScalarRegressionTask

# --- Data (MP devset) ---
# devset name as created by the example script:
DATASET_NAME = "materials_project_devset_single_task"

def make_loaders(batch_size=64, num_workers=4):
    train = ds_reg.get_dataset(DATASET_NAME, split="train")
    val   = ds_reg.get_dataset(DATASET_NAME, split="val")
    test  = ds_reg.get_dataset(DATASET_NAME, split="test")
    dl = lambda ds, shuf: DataLoader(ds, batch_size=batch_size, shuffle=shuf,
                                     num_workers=num_workers, pin_memory=True,
                                     collate_fn=ds.collate_fn)
    return dl(train, True), dl(val, False), dl(test, False)

# --- Model ---
def make_model():
    # small SchNet; matsciml wraps it for PyG graphs
    encoder = SchNet(hidden_channels=64, num_filters=64, num_interactions=3)
    # task head: single-target regression (e.g., formation energy)
    task = ScalarRegressionTask(encoder, target_key="y", optimizer="adam", lr=1e-3)
    return task

def main():
    pl.seed_everything(7)
    train_loader, val_loader, test_loader = make_loaders()
    model = make_model()
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main()
