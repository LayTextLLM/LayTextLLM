import json
from tqdm import tqdm
from utils import evaluateANLS, evaluate_exact_match_accuracy, evalFscore




model_name ="laytextllm" 
identifier="vqa"
dataset="funsd"
pred_result_path = f"./results/{dataset}_{identifier}.json"


with open(pred_result_path, "r") as fin:
    preds_data = json.load(fin)

gts = {}
preds = {}

for line in preds_data:
    filename = json.loads(line["metadata"])["sample_name"]

    if dataset == "sroie":
        key = line["prompt"].split("What is the")[-1].split("in this receipt?")[0]
    elif dataset == "cord":
        key = line["prompt"].split("What is the ")[-1].split("?\n## answer:")[0].strip('''\"''')
    elif dataset == "funsd":
        key = line["prompt"].split("content in the ")[-1].split(" field?\n## answer:")[0].strip('''\"''')

    gt = [line["gt"]] if isinstance(line['gt'], str) else line["gt"]
    pred = [line["pred"]] if isinstance(line["pred"], str) else line["pred"]

    if filename not in gts:
        gts[filename] = {key:gt}
    else:
        if key not in gts[filename]:
            gts[filename][key] = gt
        else:
            gts[filename][key].extend(gt)

    if filename not in preds:
        preds[filename] = {key:pred}
    else:
        if key not in preds[filename]:
            preds[filename][key] = pred
        else:
            preds[filename][key].extend(pred)

print('dataset and model: ', f"{model_name}_{identifier}_{dataset}")
evalFscore(gts, preds)
