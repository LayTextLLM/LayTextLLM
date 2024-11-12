deepspeed --num_gpus=8 --master_port=12355 train/laytextllm_train.py \
--train_data datasets/docvqa_train.json \
--identifier docvqa_ds_bs2 \
--deepspeed_config ds_config/ds_z2_offload_config.json \
--model_path LayTextLLM/LayTextLLM-Zero \
--batch_size 2