# sitecustomize.py ─ loaded automatically on every `python` start-up
import torch
import pytorch_lightning.callbacks.model_checkpoint as pl_mc

# Allow the Lightning class that lives in the .ckpt/.safetensors
torch.serialization.add_safe_globals([pl_mc.ModelCheckpoint])
