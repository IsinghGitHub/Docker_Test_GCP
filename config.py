import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10

BERT_PATH = "bert-base-uncased"
MODEL_PATH = "model.bin"
TRAIN_FILE = "train.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)
DEVICE = "cpu"
