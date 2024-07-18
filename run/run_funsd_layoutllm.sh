python3 infer/laytextllm_inference.py \
--dataset funsd_layoutllm \
--model_path LayTextLLM/LayTextLLM-VQA \
--cuda_num 4 \
--test_data datasets/funsd_layoutllm_test.json \
--identifier vqa