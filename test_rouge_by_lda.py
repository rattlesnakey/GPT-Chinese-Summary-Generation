# -*- encoding:utf-8 -*-
from rouge import Rouge
import jiagu
import json
from tqdm import tqdm 

def process_for_rouge(pred_summary, summary):
    return [' '.join(list(pred_summary)), ' '.join(list(summary))]

if __name__ == '__main__':
    metric = Rouge()
    total_rouge_1, total_rouge_2, total_rouge_l = 0, 0, 0
    fin = open('./data/test.json', 'r')
    count = 0
    for line in tqdm(fin):
        line = json.loads(line)
        document, summary = line['document'], line['summary']
        pred_summary = ''.join(jiagu.summarize(document, 2))
        result = metric.get_scores(*process_for_rouge(pred_summary, summary))[0]
        total_rouge_1 += result['rouge-1']['f']
        total_rouge_2 += result['rouge-2']['f']
        total_rouge_l += result['rouge-l']['f']
        count += 1
    print(f'rouge1:{total_rouge_1 / count}, rouge2:{total_rouge_2 / count}, rougeL:{total_rouge_l / count}' ) 



