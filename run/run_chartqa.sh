python3 infer/laytextllm_inference.py \
--dataset chartqa \
--model_path LayTextLLM/LayTextLLM-VQA \
--cuda_num 4 \
--test_data datasets/chartqa_test.json \
--identifier vqa