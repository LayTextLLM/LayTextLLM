import json
from tqdm import tqdm
from utils import evaluateANLS, evaluate_exact_match_accuracy




model_name ="laytextllm" 
identifier="vqa"
dataset="docvqa"
pred_result_path = f"./results/{dataset}_{identifier}.json"


with open(pred_result_path, "r") as fin:
    preds_data = json.load(fin)

human_part = []

idx = 0
for item in tqdm(preds_data):

   
    human_part.append({
        'answer': item['pred'],
        'annotation': item['gt'] 
    }) 


acc = evaluate_exact_match_accuracy(human_part)
anls = evaluateANLS(human_part)

print('dataset and model: ', f"{model_name}_{identifier}_{dataset}")
print('acc: ', acc)
print('anls: ', anls)
