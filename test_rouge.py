# import datasets
# rouge = datasets.load_metric('rouge')
from rouge import Rouge
import jieba
rouge = Rouge()
print(jieba.cut("我很快乐"))
predictions = [jieba.cut("我很快乐")]
references = [jieba.cut("你很快乐吗")]
# results = rouge.compute(predictions=predictions, references=references)
results = rouge.get_scores(predictions, references)
print(results)