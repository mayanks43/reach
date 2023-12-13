import re
import os
import json
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

@dataclass
class Rule:
    name: str
    type: str
    value: str
    trigger: Optional[str] = None
    controlled: Optional[str] = None
    controller: Optional[str] = None

pattern = r"(?:\/([^\/]+)\/)|(?:\(([^)]+)\))"
template_file = 'events_master_template.yml'

def func(matches, i, to_split):
    if i < 0 or i >= len(matches):
        return to_split

    slash, paren = matches[i]
    found = (slash, "/") if not paren else (paren, "(")
    assert found[0] != ""

    recur_matches = re.findall(pattern, found[0])
    split_found = func(recur_matches, 0, [found[0]])

    new_to_split = set([])
    for item in split_found:
        words = item.split("|")
        for split_reg in to_split:
            for word in words:
                find = r"\/" + \
                    re.escape(
                        found[0]) + r"\/" if found[1] == "/" else r"\(" + re.escape(found[0]) + r"\)"
                subst = "/" + word + \
                    "/" if found[1] == "/" else "(" + word + ")"
                new_reg = re.sub(find, subst, split_reg)
                new_to_split.add(new_reg)
    return func(matches, i+1, new_to_split)

def explode(text):
    matches = re.findall(pattern, text)
    return func(matches, 0, set([text]))

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

def mini_parse_event_rule(rule_item):
    name = rule_item["rule_name"]
    type = rule_item["type"]
    value = rule_item["rule"]

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

        return Rule(
            name=name,
            type=type,
            value=value,
            trigger=trigger,
            controlled=controlled,
            controller=controller
        )

    return Rule(name=name, type=type, value=value)

directory_path = "rules"
json_data = read_json_files(directory_path)
rules = {}
for item in json_data:
    value = item["rule"]
    if value == "MISSING VAL":
        continue
    name = item["rule_name"]

    # skip this for now due to explosion in combinatorial space
    if name == "Positive_activation_nested_syntax_2_verb":
        continue

    if name not in rules:
        rules[name] = mini_parse_event_rule(item)

preamble = ""
with open(template_file, "r") as file:
    preamble = file.read()

rule_template = """
- name: {}
  label: {}
  pattern: |
{}
"""

def add_tabs_and_reconcatenate(input_string, number_of_spaces):
    parts = input_string.split("\n")
    space_prefix = " " * number_of_spaces
    modified_parts = [space_prefix + part for part in parts]
    return "\n".join(modified_parts)

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

create_directory_if_not_exists("rule_explosion")
for name, rule in rules.items():
    text = preamble
    if rule.trigger:
        new_triggers = explode(rule.trigger)
        new_controlleds = explode(rule.controlled)
        new_controllers = explode(rule.controller)

        for i, trigger in enumerate(new_triggers):
            new_rule = rule_template.format(
                rule.name + "_trigger_" + str(i),
                rule.type,
                add_tabs_and_reconcatenate(trigger + rule.controlled + rule.controller, 4),
            )
            text += add_tabs_and_reconcatenate(new_rule, 2)

        for i, controlled in enumerate(new_controlleds):
            new_rule = rule_template.format(
                rule.name + "_controlled_" + str(i),
                rule.type,
                add_tabs_and_reconcatenate(rule.trigger + controlled + rule.controller, 4),
            )
            text += add_tabs_and_reconcatenate(new_rule, 2)

        for i, controller in enumerate(new_controllers):
            new_rule = rule_template.format(
                rule.name + "_controller_" + str(i),
                rule.type,
                add_tabs_and_reconcatenate(rule.trigger + rule.controlled + controller, 4),
            )
            text += add_tabs_and_reconcatenate(new_rule, 2)
    else:
        new_rules = explode(rule.value)

        for i, new_rule_pattern in enumerate(new_rules):
            new_rule = rule_template.format(
                rule.name + '_pattern_' + str(i),
                rule.type,
                add_tabs_and_reconcatenate(new_rule_pattern, 4),
            )
            text += add_tabs_and_reconcatenate(new_rule, 2)

    with open("rule_explosion/events_master_" + rule.name + ".yml", "w") as file:
        file.write(text)
