from transformers import AutoTokenizer
import pytorch_lightning as pl
import torch
import json, random
from model import MODEL_NAME, TextRuleT5DataLoad, TextRuleT5Model
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils import Rule, DATA_DIR
import warnings

EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    checkpoint = ModelCheckpoint(
        dirpath=directory,
        filename="best-model",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        callbacks=[early_stopping, checkpoint],
        max_epochs=EPOCHS,
        accelerator="auto",
    )

    trainer.fit(model, dataload)

pl.seed_everything(100)
warnings.filterwarnings("ignore")
directory = DATA_DIR

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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train(tokenizer, train_rules, val_rules, directory)
