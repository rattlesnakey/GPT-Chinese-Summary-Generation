# -*- encoding:utf-8 -*-
from rouge import Rouge
from dataset import MyDataset
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
import argparse
from tqdm import tqdm 
import json
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES']='0'
PAD = '[PAD]'
pad_id = 0

def calculate_loss_and_accuracy(outputs, labels, device):
    logits = outputs[0]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    _, preds = shift_logits.max(dim=-1)  

    return loss, accuracy

def collate_fn(batch):
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param batch:
    :return:
    """
    global pad_id
    input_ids = []
    btc_size = len(batch)
    max_input_len = 0  # 该batch中最长的input，用于该batch的数据对齐
    # 计算该batch中input的最大长度
    for btc_idx in range(btc_size):
        if max_input_len < len(batch[btc_idx]):
            max_input_len = len(batch[btc_idx])
    # 使用pad_id对小于max_input_len的input_id进行补全
    for btc_idx in range(btc_size):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend([pad_id] * (max_input_len - input_len))
    return torch.tensor(input_ids, dtype=torch.long)

def create_model(args, vocab_size):
    """
    :param args:
    :param vocab_size:字典大小
    :return: model, n_ctx
    """
    if args.pretrained_model_path:  # 如果指定了预训练的GPT2模型
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model_path)
    else:  # 若没有指定预训练模型，则初始化模型
        model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
        model = GPT2LMHeadModel(config=model_config)
    # 根据tokenizer的vocabulary调整GPT2模型的voca的大小
    model.resize_token_embeddings(vocab_size)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    return model, model.config.to_dict().get("n_ctx")

def evaluate(model, device, test_list, multi_gpu, args):
    model.eval()
    test_dataset = MyDataset(test_list)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                collate_fn=collate_fn,drop_last=True)
    with torch.no_grad():
        for batch_idx, input_ids in enumerate(test_dataloader):
            input_ids.to(device)
            outputs = model.forward(input_ids=input_ids)
            loss, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids, device=device)

            if multi_gpu:
                loss = loss.mean()
                accuracy = accuracy.mean()
            if args.gradient_accumulation > 1:
                loss = loss / args.gradient_accumulation
                accuracy = accuracy / args.gradient_accumulation
            logger.info("evaluate batch {} ,loss {} ,accuracy {}".format(batch_idx, loss, accuracy))
            # tb_writer.add_scalar('loss', loss.item(), overall_step)
        logger.info("finishing evaluating")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True) # gpt那里也可以用tokenizer的batch_decode来试一试，他其实就是把id转化成了token 
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [' '.join(list(pred.strip().replace('<extra_id_0>', ''))) for pred in decoded_preds]
    decoded_labels = [' '.join(list(label.strip().replace('<extra_id_0>', ''))) for label in decoded_labels]
    result = metric.get_scores(decoded_preds, decoded_labels, avg=True)
    # Extract a few results，这边是乘以100了，而且取的是mid的fmeasure值
    result = {key: value['f'] * 100 for key, value in result.items()}
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


def main():
    tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    vocab_size = len(tokenizer)
    global pad_id
    pad_id = tokenizer.convert_tokens_to_ids(PAD)
    model, n_ctx = create_model(args, vocab_size)
    model.to(device)
# **********************above is old version, which is not acceptable ******************************
from interact import top_k_top_p_filtering
from interact import create_logger
from interact import get_summary

def set_evaluate_args():
    """
    Sets up the evaluate arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='GPT2_NLPCC_Summary/config.json', type=str, required=False,
                        help='模型参数')
    parser.add_argument('--log_path', default='logs/interacting_2.log', type=str, required=False, help='interact日志存放位置')
    parser.add_argument('--voca_path', default='vocabulary/vocab_NLPCC.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--summary_model_path', default='/home/zhy2018/projects/abstract_gpt/GPT2_NLPCC_Summary', type=str, required=False, help='对话模型路径')
    parser.add_argument('--save_samples_path', default="sample/", type=str, required=False, help="保存聊天记录的文件路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--seed', type=int, default=42, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--max_len', type=int, default=512, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=1, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', default=False, help='不使用GPU进行预测')
    parser.add_argument('--test_data_path', help='the path of test data path')
    return parser.parse_args()

def process_summay(pred_summary, label_summary):
    return [' '.join(list(pred_summary)), ' '.join(list(label_summary))]

def get_rouge(test_data_path, model, tokenizer, device, args):
    total_rouge_1, total_rouge_2, total_rouge_l, count = 0, 0, 0, 0
    for line in tqdm(open(test_data_path, 'r')):
        try:
            cur_json = json.loads(line)
            pred_sumary = get_summary(cur_json['document'], model, tokenizer, device, args)
            label_summary = cur_json['summary']
            cur_result = metric.get_scores(*process_summay(pred_sumary, label_summary))[0]
            total_rouge_1 += cur_result['rouge-1']['f'] * 100
            total_rouge_2 += cur_result['rouge-2']['f'] * 100
            total_rouge_l += cur_result['rouge-l']['f'] * 100
            count += 1
        except Exception:
            continue

    logger.info(f'valid count:{count}')
    logger.info(f'rouge1:{total_rouge_1 / (count)}, rouge2:{total_rouge_2 / (count)}, rougel:{total_rouge_l / (count)}')
        


    

if __name__ == '__main__':
    args = set_evaluate_args()
    logger = create_logger(args)
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    # device = 'cpu'
    logger.info('using device:{}'.format(device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    tokenizer = BertTokenizer(vocab_file=args.voca_path)
    model = GPT2LMHeadModel.from_pretrained(args.summary_model_path)
    model.to(device)
    model.eval()
    logger.info(args)
    print('***********************evaluate start************************')
    metric = Rouge()
    get_rouge(args.test_data_path, model, tokenizer, device, args)