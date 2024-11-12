from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
import json
import argparse
from torch.utils.data import Dataset, DataLoader
import deepspeed
import os
import socket
import torch.distributed as dist
from transformers import AdamW

BOX_TOKEN = "<unk>"
BOX_TOKEN_ID = 0
INPUT_PROMPT_TEMPLATE = "given document <document>{ocr}</document>, answer following question: {question}\n## answer:"



import random
import torch
from torch.utils.data import Dataset

class LayTextDataset(Dataset):
    def __init__(
        self, 
        data, 
        tokenizer, 
        img_size_key='img_size', 
        poly_key='poly', 
        text_key='ocr', 
        question_key='question', 
        answer_key='answer',
        randomize_prob=0.0):  # New parameter for randomization probability

        self.data = data
        self.tokenizer = tokenizer
        self.img_size_key = img_size_key
        self.poly_key = poly_key
        self.text_key = text_key
        self.question_key = question_key
        self.answer_key = answer_key
        self.randomize_prob = randomize_prob

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Extract data
        texts = example[self.text_key]
        polys = example[self.poly_key]
        w, h = example[self.img_size_key]['w'], example[self.img_size_key]['h']
        question = example[self.question_key]
        answer = example[self.answer_key]
        
        # Skip empty OCR texts
        if len(texts) == 0:
            return None
        
        # Prepare input tokens and normalized layout polygons
        input_ids, input_polys = [], []
        
        # Pair texts and polys and apply randomization with 20% probability
        paired_texts_polys = list(zip(texts, polys))
        if random.random() < self.randomize_prob:
            random.shuffle(paired_texts_polys)
        
        for text, poly in paired_texts_polys:
            input_ids.append(BOX_TOKEN_ID)
            text_ids = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids.extend(text_ids)
            normalized_poly = [poly[0] / w, poly[1] / h, poly[4] / w, poly[5] / h]
            input_polys.append(normalized_poly)

        # Format question with OCR text
        input_data = {"ocr": self.tokenizer.decode(input_ids), "question": question}
        input_texts = INPUT_PROMPT_TEMPLATE.format(**input_data)
        question_ids = self.tokenizer.encode(input_texts, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        target_ids = [-100] * len(question_ids) + answer_ids + [self.tokenizer.eos_token_id]
        input_ids = question_ids + answer_ids + [self.tokenizer.eos_token_id]
        attention_mask = torch.ones(len(input_ids))
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": attention_mask,
            "layout_input": torch.tensor(input_polys),
            "labels": torch.tensor(target_ids)
        }


def collate_fn(batch):
    # Filter out any None entries from the batch
    batch = [b for b in batch if b is not None]

    # Left-pad input_ids and labels
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"].flip(0) for b in batch], batch_first=True, padding_value=-1
    ).flip(1)
    
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"].flip(0) for b in batch], batch_first=True, padding_value=-100
    ).flip(1)

    # Left-pad attention_mask in the same way as input_ids
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"].flip(0) for b in batch], batch_first=True, padding_value=0
    ).flip(1)

    # Pad layout_input by finding the maximum number of coordinates in this batch
    max_layout_len = max(len(b["layout_input"]) for b in batch)
    
    # Initialize a padded tensor with zeros, based on the batch size and max layout length
    padded_layout_input = torch.zeros((len(batch), max_layout_len, 4))
    
    # Fill the padded_layout_input with actual values
    for i, b in enumerate(batch):
        layout_len = len(b["layout_input"])
        padded_layout_input[i, :layout_len, :] = b["layout_input"]

    return {
        "input_ids": input_ids,
        "laytout_input": padded_layout_input,
        "labels": labels,
        "attention_mask": attention_mask
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--identifier",
        type=str,
        default='funsd',
        help="model identifier",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="LayTextLLM/LayTextLLM-Zero",
        help="Model path",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="./datasets/funsd_train.json",
        help="Training data path",
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="./ds_config/ds_z2_offload_config.json",
        help="Path to DeepSpeed configuration file",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank passed by DeepSpeed for distributed training",
    )
    args = parser.parse_args()

    # Set up distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Load dataset
    with open(args.train_data, "r") as f:
        train_data = json.load(f)

    dataset = LayTextDataset(train_data, tokenizer)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"./trained_models/{args.identifier}",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        deepspeed=args.deepspeed_config,
        bf16=True,
        report_to="none",  # Disable reporting for cleaner logs, modify if needed
        remove_unused_columns=False,
        local_rank=args.local_rank,
        learning_rate=1e-5,     
        weight_decay=0.01,
        warmup_steps=0.005,
    )

    # Define custom data collator
    def data_collator(features):
        return collate_fn(features)

    # Initialize Trainer with DeepSpeed support
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()