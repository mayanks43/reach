import pytorch_lightning as pl
import torch
from transformers import T5ForConditionalGeneration, AdamW

# params
INPUT_MAX_LEN = 512 #input length
OUTPUT_MAX_LEN = 512 # output length
TRAIN_BATCH_SIZE = 4 # batch size of training
VAL_BATCH_SIZE = 4 # batch size for validation
MODEL_NAME = "Salesforce/codet5-small"

class TextRuleT5Dataset:
  def __init__(self, rule_list, tokenizer):   
    self.rule_list = rule_list
    self.tokenizer = tokenizer
    self.input_max_len = INPUT_MAX_LEN
    self.output_max_len = OUTPUT_MAX_LEN
  
  def __len__(self):                      # This method retrieves the number of items from the dataset
    return len(self.rule_list)

  def __getitem__(self, idx):             # This method retrieves the item at the specified index item. 
    text = self.rule_list[idx].text
    rule = self.rule_list[idx].rule

    input_tokenize = self.tokenizer(      
        text,
        max_length=self.input_max_len,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    output_tokenize = self.tokenizer(
        rule,
        max_length=self.output_max_len,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    
    input_ids = input_tokenize["input_ids"].flatten()
    attention_mask = input_tokenize["attention_mask"].flatten()
    labels = output_tokenize["input_ids"].flatten()

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
