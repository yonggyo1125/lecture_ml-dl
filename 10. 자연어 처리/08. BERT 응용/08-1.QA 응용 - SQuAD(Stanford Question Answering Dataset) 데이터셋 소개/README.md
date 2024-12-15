# BERT Fine-Tuning SQuAD 데이터셋

## 전이 학습(Transfer Learning)

- 전이 학습(Transfer Learning) 또는 Fine-Tuning이라고 부르는 기법은 이미 학습된 Neural Networks의 파라미터를 새로운 Task에 맞게 다시 미세조정(Fine-Tuning)하는 것을 의미합니다.
- 컴퓨터 비전 문제 영역에서는 ImageNet 등의 데이터셋에 미리 Pre-Training 시키고 이 파라미터들을 내가 풀고자하는 문제에 맞게 Fine-Tuning하는 과정이 광범위하게 사용되고 있었습니다.
- 최근에는 BERT, GPT 같은 대규모 자연어 처리 모델이 등장하면서 자연어 처리 문제 영역에서도 전이 학습의 개념이 광범위하게 사용되고 있습니다.

![스크린샷 2024-12-15 오전 9 58 38](https://github.com/user-attachments/assets/4cc03795-ed59-4c76-8529-51952990557c)

## BERT Overview

- 대량의 corpus 데이터셋으로 Pre-training -> 목적에 맞게 Fine-Tuning

![스크린샷 2024-12-15 오전 10 00 55](https://github.com/user-attachments/assets/e88c8b89-6288-46b7-9fca-46310c5881ec)

## SQuAD v1.1 Dataset

- The Stanford Question Answering Dataset (SQuAD v1.1) 는 100K(10만개)의 Question & Answer 쌍으로 구성된 데이터셋입니다.
- Question과 Answer를 위한 문맥 Paragraph가 주어지고 해당 Paragraph에 있는 정보를 통대로 Answer를 맞추는 형태의 데이터셋입니다.
- https://rajpurkar.github.io/SQuAD-explorer/

![스크린샷 2024-12-15 오전 11 13 02](https://github.com/user-attachments/assets/932def71-3eb3-4b91-a2b0-50929021141b)

## SQuAD Dataset BERT Input 및 학습

- Question + Paragraph를 Sentence A & Sentence B로 하나로 묶음
- 적절한 Answer를 Prediction
- Fine-Tuning 학습 세팅
  - 3epoch
  - learning rate : 5e-5(0.00005)
  - batch size : 32

## SQUAD 데이터 셋의 성능 평가 방식

![스크린샷 2024-12-15 오전 11 17 40](https://github.com/user-attachments/assets/579847a5-48cb-4f39-822e-dfd0564019de)


## SQuAD v1.1 BERT Result

- 기존 모델 대비 더 높은 stsate-of-the-art 성능의 EM(Exact Match), F1 Score 성능을 보여줌

![스크린샷 2024-12-15 오전 11 23 07](https://github.com/user-attachments/assets/4c62213a-fc5e-400a-8349-f7e13c8ffb92)


## SQuAD v2.0 Dataset

- <b>SQuAD v2.0</b>은 기존의 SQuAD v1.1 모델의 100K(10만개)의 Question & Answer 쌍에 <b>답을 구할수 없는(un-answerable) 문제 50K</b>(5만개)의 데이터를 추가한 데이터셋입니다.


