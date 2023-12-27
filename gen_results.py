from transformers import RobertaTokenizer
import pytorch_lightning as pl
import warnings

import json
from tqdm import tqdm
from rule import Rule, GeneratedRule
from model import TextRuleT5Model, MODEL_NAME, INPUT_MAX_LEN, OUTPUT_MAX_LEN
from pathlib import Path
import random
from utils import add_tabs_and_reconcatenate

def generate_rule(text, trained_model, tokenizer):
    inputs_encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length= INPUT_MAX_LEN,
        padding = 'max_length',
        truncation='only_first',
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

SPLIT_TYPE = "id_split"
DATA_FILE = "train_data"
qualifier = "_500"
row_count = 500
template_file = "events_master_template.yml"

pl.seed_everything(100)
warnings.filterwarnings("ignore")
roberta_tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
proc_data = []
directory = SPLIT_TYPE

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
for i in tqdm(range(min(row_count, len(rules)))):
        text = rules[i].processed_text

        # Predictions
        preds = generate_rule(text, trained_model, roberta_tokenizer)
        pred0 = preds[0]
        pred1 = preds[1]

        gen_rule = GeneratedRule(
            **rules[i].model_dump(),
            pred0=pred0,
            pred1=pred1
        )
        results.append(gen_rule)

Path(directory).mkdir(parents=True, exist_ok=True)
results_dict = json.dumps([result.dict() for result in results])
json_file = directory + "/" + DATA_FILE + qualifier + "_results.json"
with open(json_file, "w") as file:
    file.write(results_dict)

# YAML file writing section
preamble = ""
with open(template_file, "r") as file:
    preamble = file.read()

rule_template = """
- name: {}
  label: {}
  pattern: |
{}
"""

text = preamble
for result in results:
    new_rule = rule_template.format(
        result.rule_name,
        result.base_type,
        add_tabs_and_reconcatenate(result.pred0, 4),
    )
    text += add_tabs_and_reconcatenate(new_rule, 2)

# Sample commands to copy over YAML file
# cp id_split/train_data_500_results_events.yml main/src/main/resources/org/clulab/reach/biogrammar/id_split/
# cp ood_split/train_data_500_results_events.yml main/src/main/resources/org/clulab/reach/biogrammar/ood_split/

yaml_file = directory + "/" + DATA_FILE + qualifier + "_results_events.yml"
with open(yaml_file, "w") as file:
    file.write(text)