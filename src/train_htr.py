import os
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG = os.path.join(_HERE, "..", "Project", "config.yaml")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from src.dataset import IAMDataset
from src.htr_models import CRNN, build_trocr
from src.metrics import calculate_cer_wer

class CTCLabelConverter(object):
    def __init__(self, character):
        dict_character = list(character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1
        self.character = ['[blank]'] + dict_character

    def encode(self, text, batch_max_length=25):
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text_index, length):
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]
            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.character[t[i]])
            texts.append(''.join(char_list))
            index += l
        return texts

def train_crnn(config_path=None):
    if config_path is None:
        config_path = _DEFAULT_CONFIG
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    
    vocab = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
    converter = CTCLabelConverter(vocab)
    nclass = len(converter.character)

    dataset = IAMDataset(csv_file=config['dataset']['csv_path'], root_dir="")
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, drop_last=True)
    
    model = CRNN(imgH=128, nc=1, nclass=nclass, nh=config['models']['crnn']['hidden_size']).to(device)
    
    criterion = nn.CTCLoss(zero_infinity=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    model.train()
    for epoch in range(config['training']['epochs']):
        for i, batch in enumerate(dataloader):
            if batch is None or batch['image'] is None: continue
            
            image = batch['image'].to(device)
            text = batch['text']
            
            text_encoded, length = converter.encode(text)
            text_encoded = text_encoded.to(device)
            
            preds = model(image)
            preds_size = torch.IntTensor([preds.size(0)] * config['training']['batch_size'])
            
            cost = criterion(preds, text_encoded, preds_size, length)
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch}/{config['training']['epochs']}] Batch {i} Loss: {cost.item():.4f}")

    # --- Evaluation after training ---
    print("\n=== CRNN Evaluation Metrics ===")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            if batch is None or batch['image'] is None:
                continue
            image = batch['image'].to(device)
            text = batch['text']
            preds = model(image)
            preds_size = torch.IntTensor([preds.size(0)] * image.size(0))
            _, preds_index = preds.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            pred_str = converter.decode(preds_index.data, preds_size.data)
            all_preds.extend(pred_str)
            all_labels.extend(text)
    cer, wer = calculate_cer_wer(all_preds, all_labels)
    print(f"  Character Error Rate (CER): {cer:.4f}")
    print(f"  Word Error Rate     (WER): {wer:.4f}")
    print("================================\n")

def train_trocr(config_path=None):
    if config_path is None:
        config_path = _DEFAULT_CONFIG
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    processor, model = build_trocr(config['models']['trocr']['pretrained_model'])
    
    train_dataset = IAMDataset(csv_file=config['dataset']['csv_path'], root_dir="", processor=processor, is_trocr=True)
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=config['training']['batch_size'],
        fp16=True if torch.cuda.is_available() else False, 
        output_dir="./trocr_model",
        logging_steps=10,
        save_steps=1000,
        eval_steps=1000,
        num_train_epochs=config['training']['epochs']
    )
    
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
        
        cer, wer = calculate_cer_wer(pred_str, label_str)
        print(f"\n=== TrOCR Evaluation Metrics ===")
        print(f"  Character Error Rate (CER): {cer:.4f}")
        print(f"  Word Error Rate     (WER): {wer:.4f}")
        print("================================\n")
        return {"cer": cer, "wer": wer}

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.image_processor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=train_dataset, 
        data_collator=lambda x: {k: torch.stack([d[k] for d in x if d is not None]) for k in x[0].keys()} 
    )
    
    trainer.train()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "trocr":
        train_trocr()
    else:
        train_crnn()
