python -u ../evaluate_by_rouge.py \
    --device 3,4 \
    --log_path logs/evaluating.log \
    --summary_model_path /home/zhy2018/projects/abstract_gpt/LAW_Summary_models/min_ppl_model \
    --model_config /home/zhy2018/projects/abstract_gpt/GPT2_NLPCC_Summary/config.json \
    --test_data_path /home/zhy2018/projects/abstract_gpt/data/test.json

