from pydantic import BaseModel
DATA_DIR = "data"

class Rule(BaseModel):
    text: str
    rule: str
