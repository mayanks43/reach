from transformers import AutoTokenizer
import pytorch_lightning as pl
import warnings

import json
from tqdm import tqdm
from utils import Rule, DATA_DIR
from model import TextRuleT5Model, MODEL_NAME, INPUT_MAX_LEN, OUTPUT_MAX_LEN
import random

def generate_rule(text, trained_model, tokenizer):
    inputs_encoding = tokenizer(
        text,
        max_length= INPUT_MAX_LEN,
        add_special_tokens=True,
        padding = 'max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    device = next(trained_model.parameters()).device

    inputs_encoding["input_ids"] = inputs_encoding["input_ids"].to(device)
    inputs_encoding["attention_mask"] = inputs_encoding["attention_mask"].to(device)

    generate_ids = trained_model.model.generate(
        input_ids=inputs_encoding["input_ids"],
        attention_mask=inputs_encoding["attention_mask"],
        max_length=OUTPUT_MAX_LEN,
        num_beams=4,
        num_return_sequences=2,
        early_stopping=False,
    )

    preds = [
        tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for gen_id in generate_ids
    ]

    return preds

pl.seed_everything(100)
warnings.filterwarnings("ignore")
roberta_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
proc_data = []
directory = DATA_DIR
DATA_FILE = "val_data"

print("Loading data")
rules_loaded = None
with open(directory + "/" + DATA_FILE + ".json", "r") as file:
    rules_loaded = json.load(file)
rules = [Rule(**data) for data in rules_loaded]
random.shuffle(rules)

print("Loading model")
trained_model = TextRuleT5Model.load_from_checkpoint(directory + "/best-model.ckpt")
trained_model.freeze()

print("Getting predictions")
results = []
for i in tqdm(range(len(rules))):
    text = rules[i].text

    # Predictions
    preds = generate_rule(text, trained_model, roberta_tokenizer)
    pred0 = preds[0]
    pred1 = preds[1]

    gen_rule = Rule(
        text=text,
        rule=pred1,
    )
    results.append(gen_rule)

results_dict = json.dumps([result.dict() for result in results])
json_file = directory + "/" + DATA_FILE + "_results.json"
with open(json_file, "w") as file:
    file.write(results_dict)
