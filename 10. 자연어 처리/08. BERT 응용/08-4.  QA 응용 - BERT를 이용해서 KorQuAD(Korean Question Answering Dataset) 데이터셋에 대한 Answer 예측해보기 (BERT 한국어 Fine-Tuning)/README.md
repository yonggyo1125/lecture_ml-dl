# BERT를 이용한 KorQuAD 예측

- Reference :
    - https://github.com/ThilinaRajapakse/simpletransformers
    - https://towardsdatascience.com/question-answering-with-bert-xlnet-xlm-and-distilbert-using-simple-transformers-4d8785ee762a

## Simple Transformer 설치

```python
!pip install simpletransformers
```

## KorQuAD v1.0 데이터 다운로드(train, eval)

```
!wget https://raw.githubusercontent.com/korquad/korquad.github.io/master/dataset/KorQuAD_v1.0_train.json -O KorQuAD_v1.0_train.json
```

```
!wget https://raw.githubusercontent.com/korquad/korquad.github.io/master/dataset/KorQuAD_v1.0_dev.json -O KorQuAD_v1.0_dev.json
```

```python
import json

with open('KorQuAD_v1.0_train.json', 'r') as f:
    train_data = json.load(f)

train_data = [item for topic in train_data['data'] for item in topic['paragraphs'] ]
```

```python
train_data
```

```python
len(train_data)
```

```
9681
```

## 빠른 학습을 위해 일부 샘플만을 training data로 설정

```python
using_num_sample = 1000
train_data = train_data[:using_num_sample]
len(train_data)
```

```
1000
```

```python
import logging

from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
```

```python
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
```

## Config 설정

```python
# Configure the model
model_args = QuestionAnsweringArgs()
model_args.train_batch_size = 64
```


## MultiLang BERT 불러오기

```python
model = QuestionAnsweringModel(
    "bert", "bert-base-multilingual-cased", args=model_args
)
```

```
Downloading: 100%
625/625 [00:00<00:00, 14.6kB/s]
Downloading: 100%
681M/681M [00:25<00:00, 30.7MB/s]
Some weights of the model checkpoint at bert-base-multilingual-cased were not used when initializing BertForQuestionAnswering: ['cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight', 'cls.predictions.decoder.weight', 'cls.predictions.bias']
- This IS expected if you are initializing BertForQuestionAnswering from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForQuestionAnswering from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of BertForQuestionAnswering were not initialized from the model checkpoint at bert-base-multilingual-cased and are newly initialized: ['qa_outputs.bias', 'qa_outputs.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Downloading: 100%
972k/972k [00:00<00:00, 1.44MB/s]
Downloading: 100%
29.0/29.0 [00:00<00:00, 742B/s]
Downloading: 100%
1.87M/1.87M [00:00<00:00, 10.3MB/s]
```

```python
# Train the model
model.train_model(train_data)
```

```
INFO:simpletransformers.question_answering.question_answering_model: Converting to features started.
convert squad examples to features: 100%|██████████| 6167/6167 [00:48<00:00, 126.33it/s]
add example index and unique id: 100%|██████████| 6167/6167 [00:00<00:00, 702429.74it/s]
Epoch 1 of 1: 100%
1/1 [11:38<00:00, 698.64s/it]
Epochs 0/1. Running Loss: 1.0443: 100%
115/115 [11:22<00:00, 4.47s/it]
/usr/local/lib/python3.7/dist-packages/torch/optim/lr_scheduler.py:134: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
INFO:simpletransformers.question_answering.question_answering_model: Training of bert model complete. Saved to outputs/.
(115, 1.2860765229100766)
```

## dev 데이터 불러오기

```python
with open('KorQuAD_v1.0_dev.json', 'r') as f:
    dev_data = json.load(f)

dev_data = [item for topic in dev_data['data'] for item in topic['paragraphs'] ]
```

```python
dev_data
```

```python
len(dev_data)
```

```
964
```

## Prediction 수행

```python
preds = model.predict(dev_data)
```

```
INFO:simpletransformers.question_answering.question_answering_model: Converting to features started.
convert squad examples to features: 100%|██████████| 5774/5774 [00:49<00:00, 117.02it/s]
add example index and unique id: 100%|██████████| 5774/5774 [00:00<00:00, 720171.03it/s]
Running Prediction: 100%
889/889 [04:03<00:00, 4.43it/s]
```

## Prediction 결과 확인

```python
with open('KorQuAD_v1.0_dev.json', 'r') as f:
    gt_data = json.load(f)

gt_data = [item for topic in gt_data['data'] for item in topic['paragraphs'] ]
```

```python
gt_data[0]['context']
```

```
1989년 2월 15일 여의도 농민 폭력 시위를 주도한 혐의(폭력행위등처벌에관한법률위반)으로 지명수배되었다. 1989년 3월 12일 서울지방검찰청 공안부는 임종석의 사전구속영장을 발부받았다. 같은 해 6월 30일 평양축전에 임수경을 대표로 파견하여 국가보안법위반 혐의가 추가되었다. 경찰은 12월 18일~20일 사이 서울 경희대학교에서 임종석이 성명 발표를 추진하고 있다는 첩보를 입수했고, 12월 18일 오전 7시 40분 경 가스총과 전자봉으로 무장한 특공조 및 대공과 직원 12명 등 22명의 사복 경찰을 승용차 8대에 나누어 경희대학교에 투입했다. 1989년 12월 18일 오전 8시 15분 경 서울청량리경찰서는 호위 학생 5명과 함께 경희대학교 학생회관 건물 계단을 내려오는 임종석을 발견, 검거해 구속을 집행했다. 임종석은 청량리경찰서에서 약 1시간 동안 조사를 받은 뒤 오전 9시 50분 경 서울 장안동의 서울지방경찰청 공안분실로 인계되었다.
```

```python
gt_data[0]['qas'][0]['question']
```

```
임종석이 여의도 농민 폭력 시위를 주도한 혐의로 지명수배 된 날은?
```

```python
# 첫번째 데이터
print('정답 answer :', gt_data[0]['qas'][0]['answers'][0]['text'])
print('예측한 answer :', preds[0][0]['answer'][0])
```

```
정답 answer : 1989년 2월 15일
예측한 answer : 1989년 2월 15일
```

```python
# 두번째 데이터
print('정답 answer :', gt_data[0]['qas'][1]['answers'][0]['text'])
print('예측한 answer :', preds[0][1]['answer'][0])
```

```
정답 answer : 임수경
예측한 answer : 임수경
```
