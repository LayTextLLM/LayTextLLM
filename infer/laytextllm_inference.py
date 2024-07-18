from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
import argparse
import torch.nn as nn
from layout_projector import LinearProjector


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default='docvqa',
        help="dataset like new_lp",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/laytextllm_all_e1",
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
        default="./datasets/docvqa_test.json",
        type=str,
        help="test data path",
    )

    
    args = parser.parse_args()

    # Load tokenizer and model
    device = f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu"
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code = True, padding_side = 'left')
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code = True).to(device)
    loaded_dict = torch.load(f"{model_path}/layout_projector.pt")

    params_to_load = {
        'linear.weight': loaded_dict['layout_projector.weight'], 
        'linear.bias': loaded_dict['layout_projector.bias']
        }
 
 
    input_dim = 4  # bounding box size
    output_dim = 4096  # model hidden size
    projector = LinearProjector(input_dim, output_dim)
    projector.load_state_dict(params_to_load)
    projector.to(device)
    print('====num model parameters====', count_parameters(model))


    ## model generation setup
    default_generate_params = {
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
    
    extra_generate_params = dict(
            eos_token_id=tokenizer.eos_token_id,
            **default_generate_params
        )

    with open(args.test_data, "r") as fin:
        test_data = json.load(fin)

    print('==========num examples', len(test_data))

    BOX_TOKEN = "<unk>"
    BOX_TOKEN_ID = tokenizer.encode(BOX_TOKEN, add_special_tokens=False)[0]

    input_prompt_template="given document <document>{ocr}</document>, answer following question: {question}\n## answer:"
 
    outputs = []
    
    # model.eval()
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
            input_polys_tensor = torch.as_tensor(input_polys).unsqueeze(0)
            layout_embeds = projector(input_polys_tensor.to(device))

            # extract text embeddings
            # assign template to input texts
            input_data = {"ocr": tokenizer.decode(input_ids), "question": question}
            input_texts = input_prompt_template.format(**input_data)

            # extract text ids
            prompt_input_ids = tokenizer.encode(input_texts, add_special_tokens=False)
            prompt_input_ids_tensor = torch.as_tensor(prompt_input_ids).to(device)
            # hacking using abs to avoid negative id problem to extract text embedding
            text_embeds = model.get_input_embeddings()(prompt_input_ids_tensor.abs()).unsqueeze(0)
            
            # interleave layout and text embeddings
            # Replace positions where mask is True with corresponding layout embedding
            layout_mask = (prompt_input_ids_tensor == BOX_TOKEN_ID).unsqueeze(0)
            layout_mask.to(device)
    
            for i in range(len(text_embeds)):
                true_indices = layout_mask[i].nonzero().squeeze(1)
                text_embeds[i][true_indices] = layout_embeds[i][:len(true_indices)]

            model_output = model.generate(
                inputs_embeds=text_embeds,
                layout_mask=layout_mask, ## comment out if using llama2 original model
                **extra_generate_params
            )

            output_str = tokenizer.decode(model_output[0].tolist(), skip_special_tokens=True)            

            output_line = {
                "id": idx, 
                "gt": [answer] if not isinstance(answer, list) else answer, 
                "pred": output_str, 
                "prompt": tokenizer.decode(prompt_input_ids),
                "metadata": meta
                }
            
            outputs.append(output_line)

            json.dump(
                outputs, 
                open(f'./results/{args.dataset}_{args.identifier}.json', "w", encoding='utf-8') , 
                indent=4, 
                ensure_ascii=False
                )