from transformers import RobertaTokenizer, T5ForConditionalGeneration, AdamW
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from collections import defaultdict
import itertools
import random
import os, json
import yaml

import warnings
warnings.filterwarnings("ignore")

class Rule:
    def __init__(self, text, rule):
        self.text = text
        self.rule = rule

    def __repr__(self):
        return f'{self.text}\n\n{self.rule}\n\n'

class TextRuleT5Dataset:
  def __init__(self, rule_list, tokenizer):   
    self.rule_list = [y for x in rule_list.values() for y in x]
    self.tokenizer = tokenizer
    self.input_max_len = INPUT_MAX_LEN
    self.output_max_len = OUTPUT_MAX_LEN
  
  def __len__(self):
    return len(self.rule_list)

  def __getitem__(self, idx):
    text = self.rule_list[idx].text
    rule = self.rule_list[idx].rule

    input_tokenize = self.tokenizer(      
        text,
        add_special_tokens=True,
        max_length=self.input_max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    output_tokenize = self.tokenizer(
        rule,
        add_special_tokens=True,
        max_length=self.output_max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    
    input_ids = input_tokenize["input_ids"].flatten()
    attention_mask = input_tokenize["attention_mask"].flatten()
    labels = output_tokenize['input_ids'].flatten()

    out = {
        'text':text,      
        'rule':rule,
        'input_ids': input_ids,
        'attention_mask':attention_mask,
        'target':labels
    }
        
    return out

class TextRuleT5DataLoad(pl.LightningDataModule):
    def __init__(self, train_data, val_data, tokenizer):
        super().__init__()
        self.train_data_raw = train_data
        self.val_data_raw = val_data
        self.tokenizer = tokenizer
    
    def setup(self, stage=None):
        self.train_data = TextRuleT5Dataset(self.train_data_raw, self.tokenizer)
        self.val_data = TextRuleT5Dataset(self.val_data_raw, self.tokenizer)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
          self.train_data,
          batch_size=TRAIN_BATCH_SIZE,
          shuffle=True, 
          num_workers=2
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
          self.val_data,
          batch_size=VAL_BATCH_SIZE,
          num_workers=2
        )

class TextRuleT5Model(pl.LightningModule):  
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
          input_ids=input_ids, 
          attention_mask=attention_mask, 
          labels=labels
        )
        return output.loss, output.logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["target"]
        loss, logits = self(input_ids , attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["target"]
        loss, logits = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)

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
        new_token += "incoming: (" + incoming[i].strip() + ")) "
        new_tokens.append(new_token)

    return new_tokens

def add_special_tokens(tokens, cr_range, cd_range, tr_range):
    i = 0
    new_tokens = []
    cr1, cr2 = cr_range[0], cr_range[-1]
    cd1, cd2 = cd_range[0], cd_range[-1]
    tr1, tr2 = tr_range[0], tr_range[-1]

    while i < len(tokens):
        if i == cr1:
            new_tokens.append('<controller>')
            new_tokens.extend(tokens[cr1:cr2])
            new_tokens.append('</controller>')
            i += cr2 - cr1
        elif i == cd1:
            new_tokens.append('<controlled>')
            new_tokens.extend(tokens[cd1:cd2])
            new_tokens.append('</controlled>')
            i += cd2 - cd1
        elif i == tr1:
            new_tokens.append('<trigger>')
            new_tokens.extend(tokens[tr1:tr2])
            new_tokens.append('</trigger>')
            i += tr2 - tr1
        else:
            new_tokens.append(tokens[i])
            i += 1

    return new_tokens

def run(tokenizer):
    dataload = TextRuleT5DataLoad(train_rules, val_rules, tokenizer)
    dataload.setup()
    device = DEVICE
    model = TextRuleT5Model()
    model.to(device)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="text_rule_t5_model",
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

yaml_data = read_yaml_files("main/src/main/resources/org/clulab/reach/biogrammar/rule_explosion")
json_data = read_json_files("training_data/")
proc_data = []

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
    for subrule in subrules:
        subrule_name = subrule['rule_name']
        if 'trigger' in subrule_name:
            if subrule_name in yaml_data:
                trigger, controlled, controller = mini_parse_event_rule(yaml_data[subrule_name])
                triggers.append(trigger)
        elif 'controlled' in subrule_name:
            if subrule_name in yaml_data:
                trigger, controlled, controller = mini_parse_event_rule(yaml_data[subrule_name])
                controlleds.append(controlled)
        elif 'controller' in subrule_name:
            if subrule_name in yaml_data:
                trigger, controlled, controller = mini_parse_event_rule(yaml_data[subrule_name])
                controllers.append(controller)
    item['triggers'] = triggers
    item['controlleds'] = controlleds
    item['controllers'] = controllers
    new_subrule_count = 0
    new_subrules = []
    stop_making_rules = False
    for trigger in triggers:
        for controlled in controlleds:
            for controller in controllers:
                new_subrules.append(trigger + controlled + controller)
                if False: # new_subrule_count >= 1000:
                    stop_making_rules = True
                    break
            if stop_making_rules:
                break
        if stop_making_rules:
            break
    item['new_subrules'] = new_subrules
    for new_subrule in new_subrules:
        proc_data_entry = {}
        proc_data_entry['sentence_tokens'] = sentence_tokens
        proc_data_entry['base_rule_name'] = rule_name
        proc_data_entry['trigger_indices'] = trigger_indices
        proc_data_entry['controlled_indices'] = controlled_indices
        proc_data_entry['controller_indices'] = controller_indices
        proc_data_entry['lemmas'] = rule['lemmas']
        proc_data_entry['tags'] = rule['tags']
        proc_data_entry['outgoing'] = rule['outgoing']
        proc_data_entry['incoming'] = rule['incoming']
        proc_data_entry['rule'] = new_subrule
        proc_data.append(proc_data_entry)

random.shuffle(proc_data)

per_rule = defaultdict(int)
rule_list = []
for item in proc_data:
    rule_name = item['base_rule_name']
    per_rule[rule_name] += 1
    if per_rule[rule_name] <= 1000:
        rule_list.append(item)

for i in range(len(rule_list)):
    rule = rule_list[i]
    tokens = rule['sentence_tokens']
    lemmas = rule['lemmas']
    tags = rule['tags']
    incoming = rule['incoming']
    outgoing = rule['outgoing']
    new_tokens = add_special_tags(tokens, lemmas, tags, outgoing, incoming)
    rule_list[i]['new_tokens'] = new_tokens

rule_data = defaultdict(list)

for i in range(len(rule_list)):
    rule = rule_list[i]
    tokens = rule['new_tokens']
    controller_range = rule['controller_indices']
    controlled_range = rule['controlled_indices']
    trigger_range = rule['trigger_indices']
    rule_name = rule['base_rule_name']
    new_tokens = add_special_tokens(tokens, controller_range, controlled_range, trigger_range)
    rule_data[rule_name].append(Rule(' '.join(new_tokens), rule['rule'].strip()))

train_size = int(0.6 * len(rule_data))
train_rules = dict(itertools.islice(rule_data.items(), train_size))
val_rules = dict(itertools.islice(rule_data.items(), train_size, None))

train_rules_size = len([y for x in train_rules.values() for y in x])
val_rules_size = len([y for x in val_rules.values() for y in x])
print(train_rules_size, val_rules_size)

# params
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_MAX_LEN = 3328 #input length
OUTPUT_MAX_LEN = 256 # output length
TRAIN_BATCH_SIZE = 4 # batch size of training
VAL_BATCH_SIZE = 2 # batch size for validation
EPOCHS = 100 # number of epoch
MODEL_NAME = 'Salesforce/codet5-small'

roberta_tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
run(roberta_tokenizer)
