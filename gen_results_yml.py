import json
import re

template_file = "events_master_template.yml"
results_file = "val_results.json"

rules = []
with open(results_file, "r") as file:
    data = json.load(file)
    rules.extend(data)

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

text = preamble
for rule in rules:
    rule_name = rule["rule_name"]
    rule = rule["pred0"]
    match = re.match(r"(.*)_activation", rule_name)
    matched_text = match.group(1)
    if matched_text == "Positive":
        label = "Positive_activation"
    else:
        label = "Negative_activation"
    new_rule = rule_template.format(
        rule_name,
        label,
        add_tabs_and_reconcatenate(rule, 4),
    )
    text += add_tabs_and_reconcatenate(new_rule, 2)

with open("val_results_events.yml", "w") as file:
    file.write(text)
