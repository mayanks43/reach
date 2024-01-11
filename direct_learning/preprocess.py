import pytorch_lightning as pl
from collections import defaultdict
import itertools
import glob, json
import warnings
from utils import Rule, DATA_DIR
from pathlib import Path

pl.seed_everything(100)
warnings.filterwarnings("ignore")

directory = DATA_DIR

# Read files
rule_files = glob.glob("../rules/*.json")
rule_list = []
for path in rule_files:
  with open(path, 'r') as f:
    rules = json.load(f)
    rule_list.extend(rules)

def add_special_tokens(tokens, tr_range, cd_range, cr_range):
    i = 0
    new_tokens = []
    tr1, tr2 = tr_range[0], tr_range[-1]
    cd1, cd2 = cd_range[0], cd_range[-1]
    cr1, cr2 = cr_range[0], cr_range[-1]

    while i < len(tokens):
        if i == tr1:
            new_tokens.append('<trigger>')
            new_tokens.extend(tokens[tr1:tr2])
            new_tokens.append('</trigger>')
            i += tr2 - tr1
        elif i == cd1:
            new_tokens.append('<controlled>')
            new_tokens.extend(tokens[cd1:cd2])
            new_tokens.append('</controlled>')
            i += cd2 - cd1
        elif i == cr1:
            new_tokens.append('<controller>')
            new_tokens.extend(tokens[cr1:cr2])
            new_tokens.append('</controller>')
            i += cr2 - cr1
        else:
            new_tokens.append(tokens[i])
            i += 1

    return new_tokens

rule_data = defaultdict(list)

for i in range(len(rule_list)):
    rule = rule_list[i]
    if rule['rule'] != 'MISSING VAL':
        tokens = rule['sentence_tokens']
        controller_range = rule['controller_indices']
        controlled_range = rule['controlled_indices']
        trigger_range = rule['trigger_indices']
        rule_name = rule['rule_name']
        new_tokens = add_special_tokens(tokens, trigger_range, controlled_range, controller_range)
        kwargs = {
            "text": ' '.join(new_tokens), 
            "rule": rule['rule'].strip()
        }
        rule_data[rule_name].append(Rule(**kwargs))

train_size = int(0.6 * len(rule_data))
train_rules = dict(itertools.islice(rule_data.items(), train_size))
val_rules = dict(itertools.islice(rule_data.items(), train_size, None))

train_rules_size = len([y for x in train_rules.values() for y in x])
val_rules_size = len([y for x in val_rules.values() for y in x])
print(
    "Doing split for ",
    directory,
    " Training data size:",
    train_rules_size,
    " Validation data size:",
    val_rules_size
)
print("Rules in train:")
for rule_name in train_rules.keys():
    print(rule_name)
print("Rules in val:")
for rule_name in val_rules.keys():
    print(rule_name)

Path(directory).mkdir(parents=True, exist_ok=True)
train_rules_dict = json.dumps(
    [item.dict() for _, value in train_rules.items() for item in value]
)
with open(directory + "/train_data.json", "w") as file:
    file.write(train_rules_dict)

val_rules_dict = json.dumps(
    [item.dict() for _, value in val_rules.items() for item in value]
)
with open(directory + "/val_data.json", "w") as file:
    file.write(val_rules_dict)
