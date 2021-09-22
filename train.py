import transformers
import torch
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# from Pytorchtool import EarlyStopping
import json
import random
import numpy as np
import argparse
import wandb
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
# from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup

PAD = '[PAD]'
pad_id = 0
logger = None




def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1', type=str, required=False, help='the cuda device you use')
    parser.add_argument('--no_cuda', default=False, help='where or not use cuda')
    parser.add_argument('--model_config', default='config/model_config_NLPCC.json', type=str, required=False,
                        help='the path of you model config ')
    parser.add_argument('--vocab_path', default='vocabulary/vocab_NLPCC.txt', type=str, required=False, help='the path of tokenizer vocab file')
    parser.add_argument('--train_raw_path', default='data/train.json', type=str, required=False, help='the path of raw data')
    parser.add_argument('--train_tokenized_path', default='data/train_tokenized.txt', type=str,
                        required=False,
                        help='the path of tokenized data saved')
    parser.add_argument('--dev_raw_path', default='data/dev.json', type=str, required=False, help='the path of raw data')
    parser.add_argument('--dev_tokenized_path', default='data/dev_tokenized.txt', type=str,
                        required=False,
                        help='the path of tokenized data saved')
    parser.add_argument('--test_raw_path', default='data/test.json', type=str, required=False, help='the path of raw data')
    parser.add_argument('--test_tokenized_path', default='data/test_tokenized.txt', type=str,
                        required=False,
                        help='the path of tokenized data svaed')
    parser.add_argument('--log_path', default='logs/training.log', type=str, required=False, help='the path of training log saved')
    parser.add_argument('--raw', default=True, help='where or not to tokenize the raw data, if tokenized data saved, use False')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='iteration times')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='learning rate')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up steps')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='after log_step to info loss')
    parser.add_argument('--gradient_accumulation', default=2, type=int, required=False, help='gradient accumulate step')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--summary_model_output_path', default='LAW_Summary_models/', type=str, required=False,
                        help='model output path')
    parser.add_argument('--pretrained_model_path', default='GPT2_NLPCC_Summary', type=str, required=False, help='pretrained GPT2 model path')
    # parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', type=int, default=1, help="dataloader works num")
    parser.add_argument('--patience', type=int, default=4, help="patience for early stopping")
    # parser.add_argument('--train_mmi', default=False, help="若指定该参数，则训练DialoGPT的MMI模型")
    # parser.add_argument('--train_mmi_tokenized_path', default='data/train_mmi_tokenized.txt', type=str,
                        # required=False,
                        # help='将原始训练语料的每段对话翻转，然后进行tokenize之后的数据的存放位置，用于训练MMI模型')
    # parser.add_argument('--mmi_model_output_path', default='mmi_model', type=str, required=False, help='MMI模型保存路径')
    # parser.add_argument('--max_len', type=int, default=60, help='每个utterance的最大长度,超过指定长度则进行截断')
    # parser.add_argument('--max_history_len', type=int, default=4, help="dialogue history的最大长度")
    return parser.parse_args()


def set_random_seed(args):
    """
    设置训练的随机种子
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # logger里面有两个handler
    return logger


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


def preprocess_raw_data(raw_path, tokenized_path, tokenizer, n_ctx):
    """
    对原始语料进行处理，将原始语料转换为用于train的token id，对于每个sample，将其处于成如下形式"[CLS]document[SEP]summary[SEP]"
    :param args:
    :param tokenizer:
    :param n_ctx:GPT2模型的上下文窗口大小,对于超过n_ctx(n_ctx包括了特殊字符)的数据进行截断
    :return: tokenized data
    """
    logger.info("tokenizing raw data,raw data path:{}, tokenized data output path:{}".format(raw_path,
                                                                                    tokenized_path))

    with open(tokenized_path,"w+",encoding="utf-8") as f:
        with open(raw_path, 'r',encoding="utf-8") as file:
            for line in tqdm(file.readlines()):
                try:
                    file_line = json.loads(line)
                except:
                    print ("line",line)

                else:
                    initial_ids = [tokenizer.cls_token_id]
                    initial_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in file_line['document']])
                    initial_ids.append(tokenizer.sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
                    initial_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in file_line['summary']])
                    initial_ids.append(tokenizer.sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
                    # 对超过n_ctx的长度进行截断,否则GPT2模型会报错
                    initial_ids = initial_ids[:n_ctx]
                    # 输出到文件当中去，每个id之间隔着一个空格
                    for initial_id in initial_ids:
                        f.write(str(initial_id) + ' ')
                    f.write("\n")

def calculate_loss_and_accuracy(outputs, labels, device):
    """
    计算非pad_id的平均loss和准确率
    :param outputs:
    :param labels:
    :param device:
    :return:
    """
    logits = outputs[0]  # 每个token用来预测下一个token的prediction_score,维度:[batch_size,token_len,voca_size]
    # 用前n-1个token，预测出第n个token
    # 用第i个token的prediction_score用来预测第i+1个token。
    # 假定有input有n个token，则shift_logits表示model中第[0,n-2]个token的prediction_score，shift_labels表示第[1，n-1]的label
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')  # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]

    # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
    not_ignore = shift_labels.ne(pad_id)  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
    num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量

    correct = (shift_labels == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets
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

def train_fn(model, device, epoch, train_dataloader, optimizer, scheduler, multi_gpu, args):
    model.train()
    for batch_idx, input_ids in enumerate(train_dataloader):
            # 注意：GPT2模型的forward()函数，是对于给定的context，生成一个token，而不是生成一串token
            # GPT2Model的输入为n个token_id时，输出也是n个hidden_state，使用第n个hidden_state预测第n+1个token
            input_ids.to(device)
            # 解决在运行过程中，由于显存不足产生的cuda out of memory的问题
            outputs = model.forward(input_ids=input_ids)
            loss, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids, device=device)
            if multi_gpu:
                loss = loss.mean()
                accuracy = accuracy.mean()
            if args.gradient_accumulation > 1:
                loss = loss / args.gradient_accumulation
                accuracy = accuracy / args.gradient_accumulation
            loss.backward()
            # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 进行一定step的梯度累计之后，更新参数
            if (batch_idx + 1) % args.gradient_accumulation == 0:
                # running_loss += loss.item()
                # 更新参数
                optimizer.step()
                # 清空梯度信息
                optimizer.zero_grad()
                # 进行warm up
                scheduler.step()
                logger.info(
                    "train batch {} of epoch {}, loss {}, accuracy {}".format(batch_idx + 1, epoch + 1, loss,
                                                                        accuracy))
    return loss, accuracy

def eval_fn(model, device, epoch, dev_dataloader, multi_gpu, args):
    model.eval()
    with torch.no_grad():
        for batch_idx, input_ids in enumerate(dev_dataloader):
            input_ids.to(device)
            outputs = model.forward(input_ids=input_ids)
            loss, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids, device=device)
            if multi_gpu:
                loss = loss.mean()
                accuracy = accuracy.mean()
            if args.gradient_accumulation > 1:
                loss = loss / args.gradient_accumulation
                accuracy = accuracy / args.gradient_accumulation
            logger.info("evaluate batch {} of {}, loss {} ,accuracy {}".format(batch_idx, epoch + 1, loss, accuracy))
            # tb_writer.add_scalar('loss', loss.item(), overall_step)
    return loss, accuracy


# 得增加dev_loss才可以
def train(model, device, train_list, dev_list, multi_gpu, args):
    wandb.init(project='summary-generation-gpt', name=f'batch_size:{args.batch_size},lr:{args.lr}', entity='hengyuan')
    train_dataset = MyDataset(train_list)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                collate_fn=collate_fn, drop_last=True)
    dev_dataset = MyDataset(dev_list)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                collate_fn=collate_fn, drop_last=True)
    # model.train()
    # 计算所有epoch进行参数优化的总步数total_steps
    total_steps = int(train_dataset.__len__() * args.epochs / args.batch_size / args.gradient_accumulation)
    logger.info('total training steps = {}'.format(total_steps))

    # 设置优化器，并且在初始训练时，使用warmup策略
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    # early_stopping = EarlyStopping(args.patience, verbose=True, save_path=args.save_model_path)
    logger.info('starting training')
    # 用于统计每次梯度累计的loss
    # running_loss = 0
    best_dev_acc = -1
    dev_acc_list = []
    # 开始训练
    for epoch in range(args.epochs):
        epoch_start_time = datetime.now()
        train_loss, train_accuracy = train_fn(model, device, epoch, train_dataloader, optimizer, scheduler, multi_gpu, args)
        dev_loss, dev_accuracy = eval_fn(model, device, epoch, dev_dataloader, multi_gpu, args)
        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        wandb.log({'train_loss':train_loss.item(), 'train_accuracy':train_accuracy, 'dev_loss':dev_loss, 'dev_accuracy':dev_accuracy, 'lr':cur_lr})
        logger.info({'train_loss':train_loss.item(), 'train_accuracy':train_accuracy, 'dev_loss':dev_loss, 'dev_accuracy':dev_accuracy, 'lr':cur_lr})
        logger.info('saving model for epoch {}'.format(epoch + 1))
        if dev_accuracy > best_dev_acc:
            best_dev_acc = dev_accuracy
            logger.info('saving current best model for epoch {}'.format(epoch + 1))
            model_path = join(args.summary_model_output_path, 'min_ppl_model'.format(epoch + 1))
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(model_path)
        logger.info('epoch {} finished'.format(epoch + 1))
        epoch_finish_time = datetime.now()
        logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))
        dev_acc_list.append(dev_accuracy)

        if dev_accuracy > best_dev_acc:
            best_dev_acc = dev_accuracy
            es = 0
        else:
            es += 1
            logging.info(f"Counter {es} of {args.patience}")
            if es > args.patience:
                logging.info(f"Early stopping with best_acc: {best_dev_acc}, and val_acc for this epoch:{dev_accuracy}")
                break
        logger.info(f"best performance on dev set appears in epoch {dev_acc_list.index(best_dev_acc)+1}! ")
        wandb.log({'best_acc_epoch':dev_acc_list.index(best_dev_acc)+1})
    logger.info('training finished')


def evaluate(model, device, test_list, multi_gpu, args):
    logger.info("start evaluating model")
    model.eval()
    logger.info('starting evaluating')
    test_dataset = MyDataset(test_list)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
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

def read_data(tokenized_path):
    with open(tokenized_path, "r", encoding="utf8") as f:
        data = f.read()
    data_list = data.split("\n")
    return data_list

def main():
    args = setup_train_args()
    # 日志同时输出到文件和console
    # 声明全局变量logger, 然后赋值给这个全局变量
    global logger
    logger = create_logger(args)
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    # 为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    # 当得到比较好的结果时我们通常希望这个结果是可以复现
    if args.seed:
        set_random_seed(args)

    # 设置使用哪些显卡进行训练

    # 初始化tokenizer
    tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    # tokenizer的字典大小
    vocab_size = len(tokenizer)

    global pad_id
    pad_id = tokenizer.convert_tokens_to_ids(PAD)

    if not os.path.exists(args.summary_model_output_path):
        os.mkdir(args.summary_model_output_path)

    # 加载GPT2模型
    model, n_ctx = create_model(args, vocab_size)
    model.to(device)
    # 对原始数据进行预处理,将原始语料转换成对应的token_id

    if args.raw:
        preprocess_raw_data(args.train_raw_path, args.train_tokenized_path, tokenizer, n_ctx)
        preprocess_raw_data(args.dev_raw_path, args.dev_tokenized_path, tokenizer, n_ctx)
        preprocess_raw_data(args.test_raw_path, args.test_tokenized_path, tokenizer, n_ctx)

    # 是否使用多块GPU进行并行运算
    multi_gpu = False
    if args.cuda and torch.cuda.device_count() > 1:
        logger.info("Let's use GPUs to train")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
        multi_gpu = True
    # 记录模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))

    # 加载数据
    logger.info("loading data")

    train_list = read_data(args.train_tokenized_path)
    dev_list = read_data(args.dev_tokenized_path)
    test_list = read_data(args.test_tokenized_path)

    # train_list, test_list = train_test_split(data_list, test_size=0.2, random_state=1)
    # 开始训练
    train(model, device, train_list, dev_list, multi_gpu, args)
    # 测试模型
    evaluate(model, device, test_list, multi_gpu, args)


if __name__ == '__main__':
    main()
