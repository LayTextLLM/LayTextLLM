from pycocoevalcap.cider.cider import Cider
import json
from tqdm import tqdm


model_name ="laytextllm" 
identifier="vqa"
dataset="visualmrc"
pred_result_path = f"./results/{dataset}_{identifier}.json"


with open(pred_result_path, "r") as fin:
    preds_data = json.load(fin)

human_part = []

idx = 0
res_dict = {}
gt_dict = {}

for idx, item in enumerate(tqdm(preds_data)):

    pred = [item['pred']] if isinstance(item['pred'], str) else item['pred']
    gt = [item['gt']] if isinstance(item['gt'], str) else item['gt']

    res_dict[str(idx)] = pred
    gt_dict[str(idx)] = gt

   
cider = Cider()
print('dataset and model: ', f"{model_name}_{identifier}_{dataset}")
score, _  = cider.compute_score(gt_dict, res_dict)
print('cider: ', score)