# import datasets
# rouge = datasets.load_metric('rouge')
from rouge import Rouge
rouge = Rouge()
predictions = ' '.join(list("我很快乐"))
references = ' '.join(list("你很快乐吗"))
# results = rouge.compute(predictions=predictions, references=references)
results = rouge.get_scores(predictions, references)[0]
final_result = {}
for k, v in results.items():
    final_result[k] = v['f']
print(final_result)