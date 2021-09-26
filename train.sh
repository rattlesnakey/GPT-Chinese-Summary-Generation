set -e
model_config_path=config/model_config_NLPCC.json
vocab_path=vocabulary/vocab_NLPCC.txt
train_raw_path=data/train.json
dev_raw_path=data/dev.json
test_raw_path=data/test.json
train_tokenized_path=data/train_tokenized.txt
dev_tokenized_path=data/dev_tokenized.txt
test_tokenized_path=data/test_tokenized.txt
log_path=logs/training.log
raw=True
epochs=50
batch_size=4
lr=1.5e-4
warmup_steps=1000
log_step=1
gradient_accumulation=2
max_grad_norm=1.0
summary_model_output_path=LAW_Summary_models/
pretrained_model_path=GPT2_NLPCC_Summary/
seed=42
num_workers=4
patience=4
# mkdir -p

python -u train.py \
    --device 0,1 \
    --model_config $model_config_path \
    --vocab_path $vocab_path \
    --train_raw_path $train_raw_path \
    --train_tokenized_path $train_tokenized_path \
    --dev_raw_path $dev_raw_path \
    --dev_tokenized_path $dev_tokenized_path \
    --test_raw_path $test_raw_path \
    --test_tokenized_path $test_tokenized_path \
    --log_path $log_path \
    --raw $raw \
    --epochs $epochs \
    --batch_size $batch_size \
    --lr $lr \
    --warmup_steps $warmup_steps \
    --log_step $log_step \
    --gradient_accumulation $gradient_accumulation \
    --max_grad_norm $max_grad_norm \
    --summary_model_output_path $summary_model_output_path \
    --pretrained_model_path $pretrained_model_path \
    --seed $seed \
    --num_workers $num_workers \
    --patience $patience
    --model_config 


