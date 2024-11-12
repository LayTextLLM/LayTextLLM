deepspeed --num_gpus=8 --master_port=12355 infer/laytextllm_inference_ds.py \
--test_data datasets/funsd_test.json \
--dataset funsd \
--identifier funsd_ds_bs2 \
--deepspeed_config ds_config/ds_z2_offload_config.json \
--model_path trained_models/funsd_ds_bs2/checkpoint-298

