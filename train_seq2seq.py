from transformers import RobertaTokenizer
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import warnings

import json
from rule import Rule
from model import DEVICE, MODEL_NAME
from model import TextRuleT5DataLoad, TextRuleT5Model
from pathlib import Path
import random

def train(tokenizer, train_rules, val_rules, directory):
    dataload = TextRuleT5DataLoad(train_rules, val_rules, tokenizer)
    dataload.setup()
    device = DEVICE
    model = TextRuleT5Model()
    model.to(device)
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=directory,
        filename="best-model",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        callbacks=[early_stopping, checkpoint_callback],
        max_epochs=EPOCHS,
        accelerator="auto"
    )

    trainer.fit(model, dataload)

pl.seed_everything(100)
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')

# params
# ood_split = out-of-distribution split
# id_split = in-distribution split
SPLIT_TYPE = "ood_split"
EPOCHS = 100 # number of epochs

directory = SPLIT_TYPE
Path(directory).mkdir(parents=True, exist_ok=True)

train_rules_loaded = None
with open(directory + "/train_data.json", "r") as file:
    train_rules_loaded = json.load(file)
train_rules = [Rule(**data) for data in train_rules_loaded]
random.shuffle(train_rules)

val_rules_loaded = None
with open(directory + "/val_data.json", "r") as file:
    val_rules_loaded = json.load(file)
val_rules = [Rule(**data) for data in val_rules_loaded]
random.shuffle(val_rules)

roberta_tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
train(roberta_tokenizer, train_rules, val_rules, directory)
