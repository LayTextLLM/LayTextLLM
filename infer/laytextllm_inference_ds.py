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
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler

BOX_TOKEN = "<unk>"
BOX_TOKEN_ID = 0
INPUT_PROMPT_TEMPLATE = "given document <document>{ocr}</document>, answer following question: {question}\n## answer:"



class LayTextDataset(Dataset):
    def __init__(
        self, 
        data, 
        tokenizer, 
        img_size_key='img_size', 
        poly_key='poly', 
        text_key='ocr', 
        question_key='question', 
        meta_key='metadata',
        answer_key='answer'):

        self.data = data
        self.tokenizer = tokenizer
        self.img_size_key = img_size_key
        self.poly_key = poly_key
        self.text_key = text_key
        self.question_key = question_key
        self.answer_key = answer_key
        self.meta_key = meta_key

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
        metadata = example[self.meta_key]
        
        # Skip empty OCR texts
        if len(texts) == 0:
            return None
        
        # Prepare input tokens and normalized layout polygons
        input_ids, input_polys = [], []
        
        for text, poly in zip(texts, polys):
            input_ids.append(BOX_TOKEN_ID)
            text_ids = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids.extend(text_ids)
            normalized_poly = [poly[0] / w, poly[1] / h, poly[4] / w, poly[5] / h]
            input_polys.append(normalized_poly)

        # Format question with OCR text
        input_data = {"ocr": self.tokenizer.decode(input_ids), "question": question}
        input_texts = INPUT_PROMPT_TEMPLATE.format(**input_data)
        question_ids = self.tokenizer.encode(input_texts, add_special_tokens=False)
        input_ids = question_ids
        attention_mask = torch.ones(len(input_ids))
        

        return {
            "prompt_text": input_texts,
            "answer_text": answer,
            "input_ids": torch.tensor(input_ids),
            "attention_mask": attention_mask,
            "layout_input": torch.tensor(input_polys),
            "metadata": metadata
        }

def collate_fn(batch):
    batch = [b for b in batch if b is not None]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=0
    )
    answers = [b["answer_text"] for b in batch]
    prompts = [b["prompt_text"] for b in batch]
    metadata = [b["metadata"] for b in batch]
    layout_input = torch.stack([b["layout_input"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
     

    return {
        "input_ids": input_ids, 
        "layout_input": layout_input, 
        "attention_mask": attention_mask,
        "answer": answers,
        "prompt": prompts,
        "metadata": metadata
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='funsd', help="Dataset like funsd")
    parser.add_argument("--model_path", type=str, default="trained_models/funsd/checkpoint-594", help="Model path")
    parser.add_argument("--test_data", type=str, default="./datasets/funsd_test.json", help="Training data path")
    parser.add_argument("--identifier", default="funsd_ds", type=str,help="model identifier")
    parser.add_argument("--deepspeed_config", type=str, default="./ds_config/ds_z2_offload_config.json", help="Path to DeepSpeed configuration file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by DeepSpeed for distributed training")
    args = parser.parse_args()

    ## model generation setup
    generate_params = {
        "use_cache": True,
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 4096,
        "min_new_tokens": None,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "num_return_sequences": 1,
        "temperature": 1.0,
        "keyword": None
    }

    # Initialize distributed training
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True).to(device)

    # Initialize DeepSpeed for inference
    model = deepspeed.init_inference(
        model, 
        mp_size=dist.get_world_size() if args.local_rank != -1 else 1, 
        dtype=torch.float, 
        replace_method='auto', 
        replace_with_kernel_inject=True,
        max_out_tokens=4096
        )

    # Load dataset
    with open(args.test_data, "r") as f:
        test_data = json.load(f)

    dataset = LayTextDataset(test_data, tokenizer)

    # Create a DistributedSampler to split the dataset across GPUs
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=args.local_rank)

    # Set up DataLoader with collate function
    data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, sampler=sampler)

    # Run inference
    model.eval()
    outputs = []

    if args.local_rank == 0:
        pbar = tqdm(total=len(data_loader), desc="Processing", unit="batch")

    for batch_id, batch in enumerate(data_loader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        layout_input = batch["layout_input"].to(device)
        answer = batch["answer"]
        prompt = batch["prompt"]
        meta = batch["metadata"]
        
        with torch.no_grad():
            model_output = model.generate(
                input_ids=input_ids,
                laytout_input=layout_input,
                attention_mask=attention_mask,
                **generate_params)

            output_ids = model_output[0][len(input_ids[0]):]
            output_str = tokenizer.decode(output_ids, skip_special_tokens=True)
            answer = answer[0]

            output_line = {
                "gt": [answer] if not isinstance(answer, list) else answer,
                "pred": output_str,
                "prompt": tokenizer.decode(input_ids[0]),
                "metadata": meta
            }
            outputs.append(output_line)

        if args.local_rank == 0:
            pbar.update(1)
                
        # print("Generated Output:", tokenizer.decode(model_output[0], skip_special_tokens=True))

    # Aggregate results across ranks
    all_outputs = [None] * dist.get_world_size()
    dist.all_gather_object(all_outputs, outputs)

    # Only save on primary rank
    if dist.get_rank() == 0:
        # Flatten the list of lists from all ranks
        aggregated_outputs = [item for sublist in all_outputs for item in sublist]
        
        # Save to JSON
        output_path = f'./results/{args.dataset}_{args.identifier}.json'
        print(f"Save prediction results to =======> {output_path}")
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(aggregated_outputs, f, indent=4, ensure_ascii=False)