# BERT(Bidirectional Encoder Representations from Transformers)

## 전이 학습(Transfer Learning)

- <b>전이 학습(Transfer Learning)</b> 또는 Fine-Tuning이라고 부르는 기법은 이미 학습된 Neural Networks의 파라미터를 <b>새로운 Task에 맞게 다시 미세조정(Fine-Tuning)</b>하는 것을 의미합니다.
- **컴퓨터 비전** 문제 영역에서는 **ImageNet 등의 데이터셋에 미리 Pre-Training** 시키고 이 파라미터들을 **내가 풀고자하는 문제에 맞게 Fine-Tuning**하는 과정이 광범위하게 사용되고 있었습니다.
- 최근에는 **BERT, GPT 같은 대규모 자연어 처리 모델**이 등장하면서 **자연어 처리 문제 영역에서도 전이 학습의 개념이 광범위하게 사용**되고 있습니다.

![스크린샷 2024-12-13 오후 10 16 21](https://github.com/user-attachments/assets/b9193574-28b2-401e-8e4a-78413f4d732d)

## BERT의 핵심 아이디어

- **BERT의 핵심 아이디어** : 대량의 단어 Corpus로 <b>양방향으로(Bidirectional)</b> 학습시킨 Pre-Trained 자연어 처리 모델을 제공하고, 마지막 레이어에 간단한 ANN 등의 추가만을 통한 Fine-Tuning을 이용해서 **다양한 자연어처리 Task에 대해서 state-of-the-art 성능**을 보여줄 수 있음

## 𝑩𝑬𝑹𝑻<sub>𝑩𝑨𝑺𝑬</sub>와 𝑩𝑬𝑹𝑻<sub>𝑳𝑨𝑹𝑮𝑬</sub>

- **L** : number of layers(i.e., Transformer blocks)
- **H** : the hidden size
- **A** : the number of self-attention heads

- 2가지 사이즈의 BERT 모델을 공개
  - 𝑩𝑬𝑹𝑻<sub>𝑩𝑨𝑺𝑬</sub> (L=12, H=768, A=12, Total Parameters=**110M**)
  - 𝑩𝑬𝑹𝑻<sub>𝑳𝑨𝑹𝑮𝑬</sub> (L=24, H=1024, A=16, Total Parameters=**340M**)

## BERT Overview

- 대량의 corpus 데이터셋으로 **Pre-training** -> 목적에 맞게 **Fine-Tuning**

![스크린샷 2024-12-13 오후 10 25 26](https://github.com/user-attachments/assets/5fea2877-8523-441c-a546-b1eadf02d171)

## BERT Input

- We use WordPiece embeddings (Wu et al., 2016) with a 30,000 tokenvocabulary.

![스크린샷 2024-12-13 오후 10 27 07](https://github.com/user-attachments/assets/ddf4ef96-cff5-493c-b21f-e09e97425a6d)

## Pre-training BERT ‒ Task 1 ‒ Masked LM(MLM)

- 2가지 비지도 학습 문제(Unsupervised Task)에 대해 BERT를 Pre-Training 함
- <b>Task 1 - Masked LM(MLM)</b> : 인풋 데이터의 일부를 Mask(\[MASK\] 토큰)로 가리고 가린 Mask에 대한 Prediction을 수행하도록 학습시킴
- 전체 실험에서 WordPiece 토큰에서 랜덤하게 **15%**를 마스크 처리할 대상으로 선택함
- 선택된 대상에서
  - **80%**는 \[MASK\] 토큰으로 Masking 처리 함
  - **10%**는 랜덤한 토큰으로 변경함
  - **10%**는 원래 단어를 유지함

![스크린샷 2024-12-13 오후 10 35 19](https://github.com/user-attachments/assets/c6aa9b0c-da8f-43f3-8eff-bb3dc3159a43)

## Pre-training BERT ‒ Task 2 - Next Sentence Prediction (NSP)

- <b>Task 2 ‒ Next Sentence Prediction(NSP)</b> : 2개의 문장이 이어지는 문장인지 아닌지를 이진 분류(binary prediction)하도록 학습시킴
- 데이터셋 구성 과정에서 50%는 실제로 A와 B가 이어지는 문장으로 구성함 (IsNext라는 레이블로 설정)
- 50%는 랜덤한 문장 묶음으로 구성함 (NotNext라는 레이블로 설정)

![스크린샷 2024-12-13 오후 10 37 53](https://github.com/user-attachments/assets/05911c38-49a3-4330-a0cb-8641d47d97ab)

- 아래 그림에서 C 부분이 Next Sentence Prediction 예측에 대응되는 부분
- 학습이 완료된 모델은 NSP 태스크에 대해서 **97%-98%**의 정확도를 보여줌

![스크린샷 2024-12-13 오후 10 39 58](https://github.com/user-attachments/assets/5567f1b0-41d0-4264-912c-8393c1d39f7b)

## Pre-training BERT ‒ Pre-Training Dataset

- Pre-Training을 위해 다음의 데이터 셋을 사용
- BooksCorpus (**800M words**)
- English Wikipedia (**2,500M words**)
- Wikipedia 데이터셋에서는 텍스트만 사용하고 리스트, 테이블, 헤더 등은 무시함

## Pre-training Procedure

- 학습 데이터의 최대 길이는 512 토큰으로 지정
- Batch size 256으로 학습 (256 sequences \* 512 tokens = 128,000 tokens/batch)
- 1,000,000 steps 학습, 이는 약 3.3 billion(33억) word corpus에 대한 40 epoch 정도의 학습
- Adam Optimizer 사용 (lr=1e-4, 𝛽! = 0.9, 𝛽" = 0.99, L2 weight decay of 0.01,
- learning rate warmup over the first 10,000 steps, and linear decay of the learning rate.)
- We use a dropout probability of 0.1 on all layers. We use a gelu activation(Hendrycks and Gimpel, 2016) rather than the standard relu, following OpenAI GPT.
- The training loss is the sum of the mean masked LM likelihood and the mean next sentence prediction likelihood.

## Pre-training Procedure

Training of 𝐵𝐸𝑅𝑇 ()\*+ was performed on <b>4 Cloud TPUs in Pod configuration (16 TPU chips total).</b>
• Training of 𝐵𝐸𝑅𝑇<sub>LARGE</sub> was performed on <b>16 Cloud TPUs (64 TPU chips total).</b>
• Each pre-training took <b>4 days to complete.</b>
• Longer sequences are disproportionately expensive because attention is quadratic to the sequence length.
• To speed up pre-traing in our experiments, we pre-train the model with sequence length of 128 for 90% of the steps.
• Then, we train the rest 10% of the steps of sequence of 512 to learn the positional embeddings.

## Fine-Tuning BERT

- Compared to pre-training, **fine-tuning is relatively inexpensive.**
  • All of the results in the paper <b>can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU</b>, starting from the exact same pre-trained model.

## Experiment Result

![스크린샷 2024-12-14 오후 11 13 30](https://github.com/user-attachments/assets/a583c0a1-27fd-4c07-9ac9-72bc0ee26a7a)



## BERT Input

- **Token Id** : 토큰화된 인풋 데이터
- **Mask Id** : 실제 단어로 구성된 부분을 Binary로 표현
- **Segment Id** : 인풋 문장이 여러개인지 구분 (인풋 문장이 하나면 모두 0로, 2개면 첫문장은 0, 뒷문장은 1로 구분)

## 오픈소스로 공개된 BERT

- 구글에서는 대용량 데이터 셋에 대해 학습시킨 BERT 모델의 파라미터를 모두 **오픈소스로 공개**하였습니다.
- 따라서 우리가 특정 자연어 처리 문제 영역을 해결하고자 할 때 구글에서 공개한 BERT 모델을 토대로 Fine-Tuning을 진행하면 **적은 노력으로도 고성능의 자연어처리 모델**을 만들어 낼 수 있습니다

## BERT 공식 구현체

- https://github.com/google-research/bert
