import pytorch_lightning as pl

from collections import defaultdict
from dataclasses import dataclass
import itertools
import random
import os, json
import yaml
from pathlib import Path
from rule import Rule

import warnings
warnings.filterwarnings("ignore")

@dataclass
class SubRule:
    rule: str
    trigger: list[int]
    controlled: list[int]
    controller: list[int]

def read_yaml_files(directory):
    all_yaml_content = {}

    for filename in os.listdir(directory):
        if filename.endswith(".yml"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
                rules = data['rules']
                for rule in rules:
                    all_yaml_content[rule['name']] = rule['pattern']

    return all_yaml_content

def read_json_files(directory):
    all_json_content = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                try:
                    data = json.load(file)
                    all_json_content.extend(data)
                except json.JSONDecodeError:
                    print(
                        f"Error reading {filename}. File might not be a valid JSON.")

    return all_json_content

def mini_parse_event_rule(value):
    sents = value.split("\n")
    sents = [sent.split('#')[0].strip() for sent in sents]
    sents = [sent for sent in sents if sent]

    if len(sents) >= 3:
        args = defaultdict(str)
        possible_args = set(
            ["trigger", "controlled:BioEntity", "controller:PossibleController"]
        )
        current_arg = ""

        for sent in sents:
            found = False
            for arg in possible_args:
                if arg + ' = ' in sent:
                    current_arg = arg
                    args[current_arg] = sent + "\n"
                    found = True
                    break
            if not found:
                args[current_arg] += sent + "\n"

        trigger = args["trigger"]
        controlled = args["controlled:BioEntity"]
        controller = args["controller:PossibleController"]
        return (trigger, controlled, controller)

    return ('','','')

def add_special_tags(tokens, lemmas, tags, outgoing, incoming):
    new_tokens = []

    for i in range(len(tokens)):
        new_token = ""
        new_token += "(word: " + tokens[i] + ", "
        new_token += "lemma: " + lemmas[i] + ", "
        new_token += "tag: " + tags[i] + ", "
        new_token += "outgoing: (" + outgoing[i].strip() + "), "
        new_token += "incoming: (" + incoming[i].strip() + "))"
        new_tokens.append(new_token)

    return new_tokens

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

def reduce_size(proc_data):
    per_base_rule_count = defaultdict(int)
    per_base_rule_data = defaultdict(list)
    for item in proc_data:
        base_rule_name = item.base_rule_name
        per_base_rule_count[base_rule_name] += 1
        if per_base_rule_count[base_rule_name] <= 1000:
            per_base_rule_data[base_rule_name].append(item)
    return per_base_rule_data

def ood_split(proc_data):
    per_base_rule_data = reduce_size(proc_data)

    train_size = int(0.7 * len(per_base_rule_data))
    train_rules = dict(itertools.islice(per_base_rule_data.items(), train_size))
    val_rules = dict(itertools.islice(per_base_rule_data.items(), train_size, None))

    directory = "ood_split"
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

def id_split(proc_data):
    per_base_rule_data = reduce_size(proc_data)
    proc_data = [item for sublist in per_base_rule_data.values() for item in sublist]

    train_size = int(0.7 * len(proc_data))
    train_rules = proc_data[:train_size]
    val_rules = proc_data[train_size:]

    directory = "id_split"
    print(
        "Doing split for ",
        directory,
        " Training data size:",
        len(train_rules),
        " Validation data size:",
        len(val_rules)
    )

    Path(directory).mkdir(parents=True, exist_ok=True)
    train_rules_dict = json.dumps([rule.dict() for rule in train_rules])
    with open(directory + "/train_data.json", "w") as file:
        file.write(train_rules_dict)

    val_rules_dict = json.dumps([rule.dict() for rule in val_rules])
    with open(directory + "/val_data.json", "w") as file:
        file.write(val_rules_dict)

pl.seed_everything(100)

yaml_data = read_yaml_files("main/src/main/resources/org/clulab/reach/biogrammar/rule_explosion")
json_data = read_json_files("training_data/")
proc_data = []

# subrule count per rule type
subrule_per_rule_type = defaultdict(int)
# number of times the exact subrule was added
per_subrule_count = defaultdict(int)

for item in json_data:
    triggers = []
    controlleds = []
    controllers = []

    rule = item['rule']
    rule_name = rule['rule_name']
    sentence_tokens = rule['sentence_tokens']
    trigger_indices = rule['trigger_indices']
    controlled_indices = rule['controlled_indices']
    controller_indices = rule['controller_indices']
    subrules = item['subrules']

    # The same subrule could match many sentences
    for subrule in subrules:
        subrule_name = subrule['rule_name']
        subrule_trigger_match = subrule['trigger']
        subrule_controlled_match = subrule['controlled']
        subrule_controller_match = subrule['controller']
        if subrule_name in yaml_data:
            trigger, controlled, controller = mini_parse_event_rule(yaml_data[subrule_name])
            if 'trigger' in subrule_name:
                triggers.append((trigger, subrule_trigger_match))
            elif 'controlled' in subrule_name:
                controlleds.append((controlled, subrule_controlled_match))
            elif 'controller' in subrule_name:
                controllers.append((controller, subrule_controller_match))

    new_subrules = []
    for trigger in triggers:
        for controlled in controlleds:
            for controller in controllers:
                new_subrules.append(
                    SubRule(
                        (trigger[0] + controlled[0] + controller[0]).strip(),
                        trigger[1],
                        controlled[1],
                        controller[1]
                    )
                )

    for new_subrule in new_subrules:
        # limit same rule to 1 time
        if per_subrule_count[new_subrule.rule] <= 1:
            marked_tokens = add_special_tokens(
                sentence_tokens,
                trigger_indices,
                controlled_indices,
                controller_indices
            )
            tagged_tokens = add_special_tags(
                sentence_tokens,
                rule["lemmas"],
                rule["tags"],
                rule["outgoing"],
                rule["incoming"]
            )
            annotated_tokens = add_special_tokens(
                tagged_tokens,
                new_subrule.trigger,
                new_subrule.controlled,
                new_subrule.controller
            )
            kwargs = {
                "rule_name": rule_name + "_" + str(subrule_per_rule_type[rule_name]),
                "rule": new_subrule.rule,
                "trigger": new_subrule.trigger,
                "controlled": new_subrule.controlled,
                "controller": new_subrule.controller,
                "base_rule_name": rule_name,
                "base_rule": rule["rule"],
                "base_trigger": trigger_indices,
                "base_controlled": controlled_indices,
                "base_controller": controller_indices,
                "base_type": rule["type"],
                "tokens": sentence_tokens,
                "lemmas": rule["lemmas"],
                "tags": rule["tags"],
                "outgoing": rule["outgoing"],
                "incoming": rule["incoming"],
                "marked_tokens": marked_tokens,
                "annotated_tokens": annotated_tokens,
                "base_text": " ".join(sentence_tokens),
                "marked_text": " ".join(marked_tokens),
                "processed_text": " ".join(annotated_tokens).strip(),
            }
            proc_data.append(Rule(**kwargs))
            per_subrule_count[new_subrule.rule] += 1
            subrule_per_rule_type[rule_name] += 1

random.shuffle(proc_data)

ood_split(proc_data)
id_split(proc_data)