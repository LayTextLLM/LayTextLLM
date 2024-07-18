python3 infer/laytextllm_inference.py \
--dataset cord_layoutllm \
--model_path LayTextLLM/LayTextLLM-VQA \
--cuda_num 1 \
--test_data datasets/cord_layoutllm_test.json \
--identifier vqa