from pathlib import Path
from datasets import load_dataset, Image

DATA_DIR = Path("/kaggle/input/asl-alphabet")
TRAIN_DIR = DATA_DIR.joinpath("asl_alphabet_train", "asl_alphabet_train")

TRAIN_DATASET = load_dataset("imagefolder", data_dir=TRAIN_DIR, split="train")
TRAIN_DATASET = TRAIN_DATASET.cast_column("image", Image(mode="RGB"))
TRAIN_DATASET = TRAIN_DATASET.with_format("torch")
