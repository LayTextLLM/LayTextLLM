from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
import argparse


BOX_TOKEN = "<unk>"
BOX_TOKEN_ID = 0
INPUT_PROMPT_TEMPLATE = "given document <document>{ocr}</document>, answer following question: {question}\n## answer:"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default='funsd_layoutllm',
        help="dataset like funsd",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="LayTextLLM/LayTextLLM-Zero",
        help="model path",
    )
    parser.add_argument(
        "--cuda_num",
        default="0",
        type=str,
        help="index of gpu used",
    )
    parser.add_argument(
        "--identifier",
        default="all",
        type=str,
        help="model identifier",
    )
    parser.add_argument(
        "--max_new_token",
        default=512,
        type=int,
        help="max output length",
    )
    parser.add_argument(
        "--test_data",
        default="./datasets/funsd_layoutllm_test.json",
        type=str,
        help="test data path",
    )

    
    args = parser.parse_args()

    # Load tokenizer and model
    device = f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu"
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code = True, padding_side = 'left')
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code = True).to(device)
    
    print('====num model parameters====', count_parameters(model))

    ## model generation setup
    generate_params = {
        "use_cache": True,
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 512,
        "min_new_tokens": None,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "num_return_sequences": 1,
        "temperature": 1.0,
        "keyword": None
    }

    with open(args.test_data, "r") as fin:
        test_data = json.load(fin)

    print('==========num examples', len(test_data))
 
    outputs = []
    
    with torch.no_grad():
        for idx,example in enumerate(tqdm(test_data)):

            input_ids, input_polys = [], []
            img_size = {}

            texts = example['ocr']
            polys = example['poly']
            w, h = example['img_size']['w'], example['img_size']['h']
            question = example['question']
            answer = example['answer']
            meta = example['metadata']

            ## if ocr is empty, skip this example
            if len(texts) == 0:
                continue

            ## prepare input text ids, and layout polys
            for text, poly in zip(texts, polys):
                input_ids += [BOX_TOKEN_ID]
                text_ids = tokenizer.encode(text, add_special_tokens=False)
                input_ids += text_ids
                text_poly = [poly[0]/w,poly[1]/h,poly[4]/w,poly[5]/h]
                input_polys.append(text_poly)

            # extract layout embeddings
            input_polys = torch.as_tensor(input_polys).unsqueeze(0).to(device)

            # extract text embeddings
            # assign template to input texts
            input_data = {"ocr": tokenizer.decode(input_ids), "question": question}
            input_texts = INPUT_PROMPT_TEMPLATE.format(**input_data)

            # extract text ids
            input_ids = tokenizer.encode(input_texts, add_special_tokens=False)
            input_ids = torch.as_tensor(input_ids).unsqueeze(0).to(device)
            attention_mask = torch.ones_like(input_ids).to(device)

            model_output = model.generate(
                input_ids=input_ids,
                laytout_input=input_polys,
                attention_mask=attention_mask,
                **generate_params
            )

            output_ids = model_output[0][len(input_ids[0]):]
            output_str = tokenizer.decode(output_ids, skip_special_tokens=True)

            output_line = {
                "id": idx, 
                "gt": [answer] if not isinstance(answer, list) else answer, 
                "pred": output_str, 
                "prompt": tokenizer.decode(input_ids[0]),
                "metadata": meta
            }
            
            outputs.append(output_line)

        json.dump(
            outputs, 
            open(f'./results/{args.dataset}_{args.identifier}.json', "w", encoding='utf-8') , 
            indent=4, 
            ensure_ascii=False
            )
