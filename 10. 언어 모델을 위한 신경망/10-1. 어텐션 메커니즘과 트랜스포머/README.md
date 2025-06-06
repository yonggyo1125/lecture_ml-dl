# 어텐션 매커니즘과 트랜스포머

## 키워드 정리
- **시퀀스-투-시퀀스 작업**은 시퀀스 데이터를 입력받아 다시 시퀀스 데이터를 출력하는 작업입니다. 자연어 처리 분야에서는 텍스트를 입력받아 또 다른 텍스트를 출력해야 하는 요약이나 번역 등의 작업이 이에 해당됩니다. 전통적으로 시퀀스-투-시퀀스 작업에는 두 개의 신경망을 사용한 인코더-디코더 구조가 널리 사용됩니다. 일반적으로 순환 신경망이 인코더와 디코더에 각각 사용되었지만, 긴 텍스트에서 문맥을 감지하는데 어려움이 있습니다. 
- **어텐션 매커니즘**은 인코더-디코더 구조에 사용된 순환 신경망의 성능을 향상시키기 위해 고안되었습니다. 기존에는 인코더의 마지막 타임스텝에서 출력한 은닉 상태만을 사용해 디코더가 새로운 텍스트를 생성했습니다. 어텐션 매커니즘은 이를 해결하기 위해 모든 타임스텝에서 인코더가 출력한 은닉 상태를 참조합니다. 이를 통해 디코더가 새로운 토큰을 생성할 때 인코더에서 처리한 토큰 중 어떤 토큰에 주의를 기울일지 결정할 수 있습니다.
- **트랜스포머** 모델은 어텐션 매커니즘을 기반으로 한 인코더-디코더 구조에서 순환층을 완전히 제거했습니다. 이를 통해 인코더에서 한 번에 하나의 토큰을 처리하지 않고 입력 텍스트 전체를 한 번에 처리할 수 있습니다. 트랜스포머의 인코더와 디코더는 비슷한 구조를 가지고 있으며 핵심 구성 요소는 멀티 헤드 어텐션, 층 정규화, 잔차 연결, 피드포워드 네트워크입니다. 인코더와 디코더는 각각 동일한 블록을 반복적으로 여러 개 쌓아서 구성됩니다. 인코더와 디코더는 각각 동일한 블록을 반복적으로 여러 개 쌓아서 구성됩니다. 인코더에서 최종적으로 출력한 토큰의 은닉 벡터는 모든 디코더의 두 번째 멀티 헤드 어텐션 층에 키워 값으로 전달됩니다. 이 층을 크로스 어텐션이라고도 부릅니다.
- **멀티 헤드 어텐션**은 트랜스포머 모델의 핵심 구성 요소입니다. 어텐션 메커니즘을 계산하는 헤드를 여러 개 병렬로 구성하고 마지막에 밀집층을 두어 원래 임베딩 차원으로 복원하는 구조를 가집니다. 이 어텐션 메커니즘은 입력 텍스트에 있는 토큰 간의 어텐션 점수를 계산하기 때문에 셀프 어텐션 메커니즘이라고도 부릅니다. 디코더에서 사용하는 첫 번째 멀티헤드 어텐션층은 훈련 시에 미래의 토큰을 사용해 어텐션 점수를 계산 할 수 없도록 마스킹하는 기법을 사용합니다. 그래서 이 층을 마스크드 멀티 헤드 어텐션 층이라고도 부릅니다.

## 순환 신경망을 사용한 인코더-디코더 네트워크
- 9장에서 배운 순환 신경망은 텍스트와 같은 시퀀스 데이터를 효과적으로 처리할 수 있지만 여러 가지 한계를 가지고 있습니다. 대표적으로 시퀀스가 길어질수록 이전에 처리한 데이터를 기억하기 어렵다는 것이죠. 이 문제를 해결하기 위해 `LSTM`과 `GRU` 같은 구조가 개발되었지만, 완벽한 해결책이 되지는 못했습니다. 
- 이러한 한계는 **기계 번역**<sup>machine translation</sup> 애플리케이션에서도 두드러졌습니다. 번역할 문장이 길어질 수록 기존 **RNN** 기반 모델의 번역 품질을 유지하기 어렵습니다. 기계 번역에 사용되는 신경망 구조는 전형적으로 **시퀀스-투-시퀀스**<sup>sequence-to-sequence</sup> 구조를 가지고 있습니다.
- 시퀀스-투-시퀀스 작업은 텍스트를 입력받아 텍스트를 출력하는 작업입니다. 대표적인 예로는 기계 번역과 문서 요약이 있습니다. 이런 작업을 수행하기 위해 보통 인코더-디코더 구조를 사용하며, 다음 그림과 같이 인코더와 디코더에 각각 순환 신경망을 적용합니다.

![스크린샷 2025-04-12 오후 9 28 40](https://github.com/user-attachments/assets/acc8ed58-65fa-48ef-9e09-6239ec1e71d1)

> 그림에서는 이해를 돕기 위해 인코더에 입력되는 텍스트와 디코더에서 출력되는 텍스트의 길이를 동일하게 표현했습니다. 하지만 일반적으로 이 두 텍스트의 길이는 다를 수 있습니다. 

- 위 그림은 기계 번역을 수행하는 인코더-디코더 구조를 보여줍니다. 그림을 보면 인코더 신경망은 입력된 문장을 단어(토큰) 단위로 하나씩 처리하면서 전체 정보를 하나의 은닉 상태에 압축합니다. 그런 다음, 디코더 신경망이 이 은닉 상태를 받아 마찬가지로 한 단어씩 번역된 문장을 생성합니다. 

> 위 그림은 인코더와 디코더가 타임스텝에 따라 순차적으로 동작하는 과정을 펼쳐서 나타낸 것입니다. 세 개의 셀(층)을 가진 신경망을 나타내는 것이 아니니 오해하지 마세요.

- 이런 구조에서는 번역할 문장이 길어질수록 초기에 입력된 내용을 기억하기 어려워집니다. 특히, 디코더 신경망은 인코더의 마지막 은닉 상태만 참고하여 번역을 수행하기 때문에 이런 문제가 더 심해집니다.
- 또한 인코더와 디코더는 텍스트를 한 토큰씩 처리합니다. 입력할 때도 한 토큰씩 받고, 출력할 때도 한 토큰씩 생성해야 하므로 속도가 느립니다. 앞의 그림의 디코더는 "I"라는 토큰을 생성한 후, 인코더의 마지막 은닉 상태와 자신의 은닉 상태를 활용해 "love"를 만듭니다. "I love"를 생성한 후 같은 방식으로 인코더의 마지막 은닉상태와 자신의 은닉 상태를 활용해 "you"를 출력합니다.
- 이처럼 디코더는 이전에 생성한 토큰을 참고하면서 다음 토큰을 생성하는데 이를 **자기회귀 모델**<sup>autoregressive model</sup>이라고 합니다. 이 개념은 순환 신경망을 사용하는 인코더-디코더 구조뿐만 아니라 앞으로 배울 다른 구조에서도 동일하게 적용됩니다. 
- 그런데 2014년, 어텐션 메커니즘이 등장하면서 순환 신경망 기반 기계 번역 애플리케이션의 성능이 크게 개선되었습니다.

## 어텐션 메커니즘

- **어텐션 매커니즘**<sup>attention mechanism</sup>은 순환 신경망 기반 인코더-디코더 모델의 성능을 크게 향상시킨 기술입니다. 기존에는 디코더가 인코더의 마지막 은닉 상태만 참고하여 번역을 수행했지만, 어텐션 메커니즘을 사용하면 인코더의 모든 타임스텝에서 계산된 은닉 상태를 활용할 수 있습니다. 
- 아래 그림을 보면서 어텐션 메커니즘이 어떻게 동작하는지 자세히 살펴보겠습니다.

![스크린샷 2025-04-12 오후 9 52 20](https://github.com/user-attachments/assets/50813d4c-a575-418d-bb71-a6608744bab9)

- 디코더가 두 번째 타임스텝에서 "love"라는 토큰을 생성할 때, 인코더의 모든 타임스텝에서 출력된 은닉 상태를 참고합니다. 즉, 디코더는 단순히 인코더의 마지막 은닉 상태만 활용하는 것이 아니라 각각의 타임스텝에서 생성된 모든 은닉 상태를 참조하여 출력을 만듭니다.

### 어텐션 가중치
- 디코더가 인코더의 은닉 상태를 활용하는 방식은 가중치를 곱하는 형태로 이루어집니다. 디코더는 인코더의 모든 타임스텝에서 생성된 은닉 상태를 동일하게 참고하는 것이 아니라, 각 은닉 상태마다 가중치를 다르게 적용하여 더 중요한 정보를 강조합니다. 이러한 가중치는 다른 모델 파라미터와 마찬가지로 신경망을 훈련하면서 함께 학습됩니다.
- 위 그림에서는 어텐션 가중치를 **a**<sub>1</sub>, **a**<sub>2</sub>, **a**<sub>3</sub>으로 나타냈습니다. 이 값들은 각각 다른 값을 가지며, 디코더의 타임스텝마다 달라질 수 있습니다. 이를 디코더가 타임스텝에서 인코더의 각기 다른 은닉 상태에 주의를 기울인다고 이해할 수 있습니다. 디코더가 입력 토큰마다 중요도를 다르게 부여하는 이런 방식을 어텐션 메커니즘이라고 합니다.

### 어텐션 매커니즘의 장점과 단점
- 어텐션 메커니즘은 긴 텍스트를 처리할 때 정보 손실을 줄이는 데 매우 효과적입니다. 인코더의 모든 타임스텝에서 생성된 은닉 상태를 참고하여 더 정확한 출력을 생성할 수 있기 때문입니다.
- 하지만 이 방식에도 단점이 있습니다. 어텐션 가중치를 계산하기 위해 인코더의 모든 타임스텝에서 생성된 은닉 상태를 저장해야 하므로 연산량이 증가합니다. 따라서 인코더가 처리할 수 있는 타임스텝의 최대 개수를 정해야 하며, 이로 인해 입력 텍스트의 길이가 제한될 수 있습니다. 또한, 어텐션을 사용해도 여전히 한 번에 한 토큰씩 처리해야 하는 한계는 남아 있습니다. 이러한 문제에도 불구하고, 어텐션 매커니즘을 사용한 획기적인 모델이 등장하면서 기계 번역은 물론, 자연어 처리 전 분야에 혁명을 일으켰습니다.

## 트랜스포머
- 어텐션 메커니즘의 효과를 극대화하기 위해 2017년 구글 연구팀은 **트랜스포머**<sup>Transformer</sup>라는 새로운 신경망 구조를 발표했습니다. 이 연구는 "Attention Is All You Need"라는 제목의 논문에서 소개되었으며, 트랜스포머는 논문의 제목처럼 어텐션 메커니즘을 적극적으로 활용합니다. 하지만 어텐션 메커니즘만 사용하는 것이 아니며 다양한 기술을 조합하여 구성됩니다.
- 트랜스포머는 기존의 인코더-디코더 구조를 유지하면서도 순환 신경망을 완전히 제거했다는 특징을 가지고 있습니다. 따라서 입력 텍스트를 한 토큰씩 처리할 필요 없이 한 번에 모두 처리할 수 있습니다.
- 다음 그림은 트랜스포머의 전체적인 구조를 단순화하여 나타낸 것입니다.

![스크린샷 2025-04-12 오후 10 04 36](https://github.com/user-attachments/assets/d18a08d0-1209-4a83-a807-e591a83a4972)

> 설명의 편의를 위해 인코더를 디코더보다 크게 그렸습니다. 앞으로 보게 되겠지만 실제로는 디코더의 구성 요소가 조금 더 많습니다.

- 그림에서 볼 수 있듯이 **인코더**와 **디코더**는 기존의 순환 신경망과 구분하기 위해 사각형으로 표시했습니다. 트랜스포머의 기본 작동 방식을 먼저 살펴보고, 이후에 내부 구조를 더 자세히 알아보겠습니다.

### 트랜스포머의 작동 방식

- 트랜스포머의 인코더는 입력된 텍스트를 한 번에 모두 처리합니다. 기존의 순환 신경망과 달리, 타입 스텝의 개념이 필요하지 않습니다. 어텐션 메커니즘으로 인해 사실상 입력 텍스트의 길이에 제한이 있는 점은 아쉽지만, 한 번에 모두 처리할 수 있어 모델의 처리 속도가 크게 향상됩니다. 

> 앞으로 알아보겠지만, 최근에는 트랜스포머 기반 모델들이 발전하면서 더 긴 문장이나 문서 전체를 처리할 수 있는 방식이 개발되고 있습니다. 

- 인코더에서 처리된 결과는 디코더에 전달되며, 디코더는 이를 바탕으로 번역된 문장을 생성합니다. 기존의 순환 신경망을 사용한 인코더-디코더 구조에서는 디코더가 인코더의 마지막 은닉 상태를 받아 번역을 수행했습니다. 하지만 트랜스포머는 순환 신경망을 사용하지 않으므로, 더 이상 은닉 상태라는 개념을 사용하지 않습니다. 대신, **은닉 벡터**<sup>hidden vector</sup> 또는 **단어 벡터**<sup>word vector</sup>, **임베딩 벡터**<sup>embedding vector</sup>라는 표현을 사용합니다.
- 디코더는 인코더에서 전달받은 은닉 벡터를 활용해 각 타임스텝에서 출력할 토큰을 생성합니다. 기존의 인코더-디코더 모델처럼, 디코더는 이전에 생성된 토큰을 참고하면서 새로운 토큰을 만듭니다. 하지만, 순환 신경망 없이도 이전 출력값을 반영할 수 있는 구조를 갖추고 있습니다.
- 예를 들어, 디코더가 두 번째 타임스텝에서 "love"를 출력하려면, 앞서 생성한 텍스트 "I"를 입력으로 받아야 합니다. 세 번째 타임스텝에서는 "I love"를 입력으로 받아 "you"를 출력하난 식입니다.
- 그림에서 볼 수 있듯이 디코더가 자기회귀 방식으로 작동하는 것은 기존의 순환 신경망을 사용한 인코더-디코더 구조와 동일합니다. 하지만 디코더가 이전 출력값을 활용하는 방식이 다릅니다. 
- 트랜스포머의 전체적인 구조를 알아보았으니, 이제 인코더 내부에서 어떻게 입력 텍스트를 한 번에 처리하는지 알아보겠습니다.


## 셀프 어텐션 메커니즘
- 기존 어텐션 매커니즘은 인코더의 은닉 상태와 디코더의 은닉 상태를 비교해 디코더가 특정 타임스텝에서 어떤 입력 토큰에 집중해야 하는지를 학습합니다. 하지만 트랜스포머에서는 이와 다르게 인코더에 입력되는 토큰만으로 어텐션 가중치를 학습하도록 만들었습니다. 이를 **셀프 어텐션**<sup>self-attention</sup>이라고 합니다.

### 셀프 어텐션의 계산 과정

- 그림으로 차근차근 알아보죠. 먼저 입력 텍스트의 각 토큰을 밀집층에 통과시킵니다. 아래 그림에 있는 밀집층은 모두 같은 층입니다. 7장에서 배웠듯이 밀집층은 한 번에 여러 개의 샘플을 처리할 수 있습니다. 여기서는 이해를 돕기 위해 각각의 토큰이 밀집층을 통과하는 것처럼 그려졌지만, 사실 전체 토큰이 한 번에 밀집층을 통과합니다.

![스크린샷 2025-04-12 오후 10 19 18](https://github.com/user-attachments/assets/3abf0eb7-42be-4073-b556-08f4a4fddec8)

> 사실상 토큰이 바로 입력되는 것이 아니라, 9장에서 배운 단어 임베딩 과정을 거친 후 어텐션 매커니즘에 전달됩니다. 이에 대해서는 뒤에서 더 자세히 알아보겠습니다.

- 밀집층을 통과한 벡터를 **쿼리**<sup>Query</sup> 벡터라고 합니다. 같은 입력 텍스트를 두 번 밀집층에 통과시켜 **키**<sup>key</sup> 벡터를 만듭니다. 여기서 중요한 점은 쿼리를 생성하는 밀집층과 키를 생성하는 밀집층이 서로 다른 층이라는 점입니다. 이를 강조하기 위해 그림에서도 다른 색으로 구분했습니다.

![스크린샷 2025-04-12 오후 10 22 34](https://github.com/user-attachments/assets/d566cfdf-49f0-4f39-9031-778043a88180)


### 어텐션 점수 계산
- 쿼리 벡터와 키 벡터가 생성되면 두 벡터를 서로 곱해서 **어텐션 점수**<sup>attention score</sup>를 계산합니다. 예를 들어, 입력된 토큰이 3개라면 쿼리 벡터도 3개, 키 벡터도 3개가 생성됩니다. 각 쿼리 벡터와 키 벡터를 곱하면 총 9개의 어텐션 점수가 만들어집니다. 이를 행렬 형태로 정리한 것이 **어텐션 행렬**<sup>attention matrix</sup>입니다. 

![스크린샷 2025-04-12 오후 10 25 09](https://github.com/user-attachments/assets/6ae7cca7-3100-401a-a6c9-64e2483e5055)

- 그 다음, 입력 텍스트를 또 다른 밀집층에 통과시켜 **값**<sup>value</sup> 벡터를 계산합니다.

![스크린샷 2025-04-12 오후 10 25 39](https://github.com/user-attachments/assets/b770c4bd-0cc5-4408-a9e2-491fe6176bcf)

- 이제 계산된 어텐션 점수를 값 벡터에 곱해서 최종적인 셀프 어텐션 출력을 생성합니다. 이 출력 벡터는 각 입력 토큰이 다른 토큰들과 얼마나 관련이 있는지를 반영한 은닉 벡터라고 할 수 있습니다. 이를 통해 모델은 문맥을 더 정확하게 이해하고, 중요한 정보를 효과적으로 강조할 수 있습니다.
- 셀프 어텐션에서 중요한 점은 각 토큰의 벡터 표현이 주어진 문제를 효과적으로 해결할 수 있도록 학습된다는 것입니다. 이를 위해 쿼리, 키, 값 벡터를 생성하는 밀집층 세 개의 가중치도 함께 학습됩니다. 셀프 어텐션 메커니즘을 간단히 그림으로 나타내면 다음과 같습니다.

![스크린샷 2025-04-12 오후 10 30 57](https://github.com/user-attachments/assets/20dfa4c2-423e-40ed-bb19-544bcfc8d036)

### 멀티 헤드 어텐션

- 셀프 어텐션 연산을 수행하는 하나의 단위를 **어텐션 헤드**<sup>attention head</sup>라고 합니다. 트랜스포머는 여러개의 어텐션 헤드를 사용하는데, 이를 **멀티 헤드 어텐션**<sup>multi-head attention</sup>이라고 합니다. 이를 그림으로 나타내면 다름과 같습니다.


![스크린샷 2025-04-13 오후 9 16 47](https://github.com/user-attachments/assets/356bcfd2-0bed-40fa-afd5-77dbcb480aad)

- 각 어텐션 헤드에서는 쿼리, 키, 값 벡터를 생성하는 밀집층이 서로 다르게 사용됩니다. 어텐션 헤드들의 출력은 하나로 합쳐진 후, 밀집층을 통과하여 어텐션 층의 최종 출력이 됩니다. 헤드의 개수는 모델마다 다르며, 보통 몇 개에서 많게는 수십 개까지 사용됩니다. 
- 트랜스포머의 핵심 요소 중 하나인 멀티 헤드 어텐션을 알아보았으니 이제 정규화 층을 알아보겠습니다. 

## 층 정규화
- 3장에서 모델을 훈련하기 전에 사이킷런의 `StandardScaler` 클래스를 사용해 입력 데이터를 정규화하는 방법을 알아보았습니다. 하지만 딥러닝에서는 여러 개 층을 거치면서 특성의 스케일이 변할 수 있기 때문에, 단순한 입력 정규화면으로는 충분하지 않습니다. 이를 해결하기 위해 고안된 것이 **배치 정규화**<sup>batch normalization</sup>입니다.
- 배치 정규화는 주로 합성곱 신경망에 널리 활용되며 층과 층 사이에 놓입니다. 이전 층의 출력을 배치 단위로 평균과 분산을 계산하여 평균이 0, 분산이 1이 되도록 조정한 후, 다음 층으로 전달합니다. 

> 모든 특성을 동일한 분포(평균 0, 분산 1)로 정규화하면 신경망이 학습한 유용한 정보가 손실될 수 있습니다. 이를 방지하기 위해 배치 정규화 층은 평균과 분산의 양을 조정하는 두 개의 파라미터를 학습하여 정규화를 수행합니다. 

- 예를 들어 합성곱 신경망의 경우 일반적으로 합성곱 층이 출력한 특성 맵의 크기는 (샘플 개수, 높이, 너비, 채널)이 됩니다. 이런 특성 맵에 배치 정규화가 적용되는 범위를 그림으로 나타내면 다음과 같습니다.

![스크린샷 2025-04-13 오후 9 24 12](https://github.com/user-attachments/assets/b0827094-cc2c-419a-9f4e-42bdbab21734)

- 위 그림에서 파란 색으로 표시된 부분이 정규화가 적용되는 단위입니다. 즉 모든 샘플에서 특정 채널의 데이터를 모아 평균과 분산을 계산한 후, 정규화를 적용합니다. 이처럼 배치 단위로 정규화를 수행하기 때문에 배치 정규화라고 부릅니다.
- 배치 정규화를 적용하면 훈련 속도가 빨라지고, 학습 과정이 안정화됩니다. 이로 인해 모델의 성능이 향상될 수 있어, 많은 신경망에서 널리 사용됩니다. 하지만 이 방식을 텍스트 데이터에 적용하기는 어려웠습니다. 텍스트 데이터는 샘플마다 길이가 다르기 때문입니다. 이를 위해 고안된 것이 **층 정규화**<sup>layer normalization</sup>입니다.
- 층 정규화는 각 샘플의 토큰마다 개별적으로 정규화를 수행하는 방식입니다. 이렇게 하면 샘플마다 길이가 달라도 독립적으로 정규화할 수 있어 샘플의 길이에 영향을 받지 않습니다. 이를 그림으로 나타내면 다음과 같습니다. 

![스크린샷 2025-04-13 오후 9 30 04](https://github.com/user-attachments/assets/5e8aa5b9-bb29-4a15-abd0-f75d064bbb86)

- 트랜스포머에서도 멀티 헤드 어텐션 층 다음에 드롭아웃과 층 정규화가 사용됩니다. 일부 모델에서는 층 정규화를 멀티 헤드 어텐션 층 앞에 배치하기도 하지만, 여기서는 원본 트랜스포머 모델의 구조를 따라 배치해 보겠습니다. 지금까지 배운 멀티 헤드 어텐션과 드롭아웃, 층 정규화를 그림으로 나타내면 다음과 같습니다.

![스크린샷 2025-04-13 오후 9 32 17](https://github.com/user-attachments/assets/01e5f1d4-29f2-4af5-a97f-d74b15a524aa)

- 멀티 헤드 어텐션과 층 정규화 사이에는 **잔차 연결**<sup>residual connection</sup>이 추가됩니다. 잔차 연결 또는 **스킵 연결**<sup>skip connection</sup>은 **ResNet**이라는 합성곱 신경망에서 처음 도입된 이후, 많은 신경망에서 널리 사용되고 있는 기술입니다. 
- 신경망은 층이 많을수록 훈련이 어려워집니다. 이를 해결하기 위해 잔차 연결이 도입되었으며, 그림에서 보듯이 멀티 헤드 어텐션 층을 거친 출력에 입력값을 그대로 더하는 방식입니다. 
- 신경망이 훈련될 때는 뒤에서부터 거꾸로 모델의 파라미터 업데이트 신호가 전파됩니다. 잔차 연결이 추가되면 이 신호가 멀티 헤드 어텐션 층을 거치지 않고 직접 앞쪽 층으로 전달될 수 있습니다. 이렇게 하면 신경망의 층을 많이 쌓아도 효과적으로 훈련할 수 있게 됩니다. 잔차 연결의 효과는 여러 신경망에서 검증되었으며, 이후 다양한 모델에서 적용되었습니다. 
- 트랜스포머의 인코더 역시 두 개의 잔차 연결을 사용합니다. 두 번째 잔차 연결은 다음에 배울 피드 포워드 네트워크의 앞뒤를 연결하는 역할을 합니다.

## 피드포워드 네트워크와 인코더 블록
- 9장에서 언급했듯이 피드포워드 신경망에는 합성곱 신경망과 완전 연결 신경망이 포함됩니다. 기술적으로 보면 트랜스포머 역시 피드포워드 신경망의 한 종류라고 볼 수 있습니다. 하지만 기존의 신경망과 구조가 크게 다르고 많은 파생 모델을 만들어내고 있기 때문에 독립적인 범주로 구분하는 것이 더 적절합니다. 트랜스포머를 추가해서 인공 신경망의 종류를 다시 그려보면 다음과 같습니다.

![스크린샷 2025-04-13 오후 9 39 30](https://github.com/user-attachments/assets/90d79666-b40b-4e2b-b6f4-1a95b7512747)

- 여기서 다루려는 **피드포워드 네트워크**<sup>feedforward network</sup>는 일반적인 피드포워드 신경망을 의미하는 것이 아닙니다. 트랜스포머 인코더에서 멀티 헤드 어텐션과 층 정규화 다음에 나오는 밀집층을 종종 피드포워드 네트워크라고 부릅니다.
- 피드포워드 네트워크는 보통 두 개의 밀집층으로 구성됩니다. 첫 번째 밀집층은 `ReLU` 활성화 함수를 사용하고, 두 번째 밀집층은 활성화 함수를 사용하지 않습니다. 그다음 다시 드롭아웃 층이 추가되며, 이 세 개의 층을 또 다른 잔차 연결이 감싸게 됩니다. 이 구조를 그림으로 나타내면 다음과 같습니다.

![스크린샷 2025-04-13 오후 9 42 51](https://github.com/user-attachments/assets/af5fa7c7-4c4b-4163-a157-3f259172239f)

- 이제 멀티 헤드 어텐션과 피드포워드 네트워크를 연결해 보겠습니다. 두 부분을 합친 후 마지막에 층 정규화를 다시 배치하면 트랜스포머 인코더 블록이 완성됩니다.

![스크린샷 2025-04-13 오후 9 42 57](https://github.com/user-attachments/assets/588cc9e8-da7c-461b-ab68-df9c5d34c01f)

- 인코더 블록이 출력하는 값은 여전히 각 토큰의 은닉 벡터입니다. 앞서 언급했듯이 입력 토큰은 단어 임베딩과 같은 벡터 표현으로 변환되어 입력됩니다. 이 벡터의 차원과 인코더 블록이 출력하는 은닉 벡터의 차원은 동일합니다. 이런 특징 덕분에, 동일한 인코더 블록을 여러 개 반복해서 배치할 수 있습니다. 모델마다 다르지만 적게는 몇 개에서 많게는 수십 개의 인코더 블록을 순차적으로 쌓아 인코더 모델을 구성합니다.

![스크린샷 2025-04-13 오후 9 43 04](https://github.com/user-attachments/assets/9d04a4e7-b22a-496a-a035-74c87adde1f7)

- 이제 인코더 모델의 구조를 거의 다 설명했습니다. 마지막으로 입력 토큰의 임베딩 벡터를 만드는 방법에 대해 알아보겠습니다.

## 토큰 임베딩과 위치 인코딩

- 9장에서 배웠듯이 자연어 처리에서는 모델이 입력된 문자를 이해할 수 있도록 토큰을 숫자로 변환하는 과정이 필요합니다. 이때 사용되는 대표적인 단어 임베딩입니다. 단어 임베딩을 생성하는 방법은 다양하지만 9장에서 했던 것처럼 임베딩 층을 추가하여 특정 작업에 맞도록 임베딩 벡터를 학습할 수 있습니다. 

### 토큰 임베딩

- 트랜스포머에서도 토큰을 고정된 크기의 실수 벡터로 변환하기 위해 임베딩 층을 사용합니다. 하지만 트랜스포머는 기존 모델과 다르게 모든 토큰을 동시에 처리하는 방식을 사용하기 때문에 토큰의 위치를 고려하지 않는다는 문제가 발생합니다. 앞서 소개한 어텐션 행렬을 다시 생각해 보세요. 이 행렬의 계산과정에서 토큰 간의 관계는 반영되지만 위치는 따로 고려되지 않았습니다. 

![스크린샷 2025-04-13 오후 10 09 42](https://github.com/user-attachments/assets/ef3bb975-7f16-49c1-9ea0-43020389d54e)

- 하지만 단어는 그 위치에 따라 의미가 달라질 수 있습니다. 예를 들어. "I love you"와 "You love me"라는 두 문장에서 사용된 단어(토큰)는 같지만, 위치가 달라 의미가 완전히 달라집니다. 따라서 트랜스포머가 문장의 의미를 정확히 이해하려면 위치 정보가 추가적으로 제공되어야 합니다. 

### 위치 임베딩

- 이 문제를 해결하기 위해 트랜스포머는 **위치 인코딩**<sup>positional encoding</sup>을 사용합니다. 위치 인코딩은 **사인**<sup>sine</sup>**함수**와 **코사인**<sup>cosine</sup>**함수**를 사용해 토큰의 위치에 따라 변하는 벡터를 생성하고, 이를 단어 임베딩에 더하는 방식입니다. 
- 예를 들어 임베딩 벡터의 차원이 5이고, 모델에 입력되는 문장의 10번째 토큰이 다음과 같이 표현된다고 가정해 보겠습니다.

![스크린샷 2025-04-14 오후 9 17 12](https://github.com/user-attachments/assets/cf187d08-8fe1-4a90-bac1-cd988664ce5d)

- 이제 벡터의 각 원소 인덱스를 임베딩 벡터의 전체 길이로 나눈 후, 이 값을 10,000의 거듭제곱한 값의 약수로 변환합니다. 그런 다음, 해당 토큰의 순서인 10을 곱합니다. 이 과정은 다음 그림을 통해 쉽게 이해할 수 있습니다. 

![스크린샷 2025-04-14 오후 9 19 32](https://github.com/user-attachments/assets/ef9fd917-a1d5-44e7-b982-686a9761eef6)

- 마지막으로 임베딩 벡터에서 짝수 번째 원소의 경우는 사인 함수를 적용하고, 홀수 번째 원소의 경우는 코사인 함수를 적용합니다. 이렇게 구한 값을 원본 임베딩 벡터에 더하면 됩니다.

![스크린샷 2025-04-14 오후 9 19 42](https://github.com/user-attachments/assets/accf4318-128e-4be7-bea3-949c0a9b32ce)

- 결국 원본 단어 임베딩의 값은 문장에서의 위치에 따라 조금씩 달라집니다. 또한 임베딩 벡터의 차원에 따라서도 값이 변합니다. 하지만 삼각 함수를 사용했기 때문에 주기성을 가질 것이라 예상할 수 있습니다. 

> 사인과 코사인 함수는 일정한 주기로 값을 반복하기 때문에, 위치 인코딩도 자연스럽게 추가적인 패턴을 갖게 됩니다.

- 위치 인코딩은 토큰의 위치와 임베딩 벡터의 차원에 따라 일정한 값으로 계산되므로, 이를 절대 위치 인코딩이라고도 부릅니다. 이어서 최근 많이 사용하는 상대 위치 인코딩(또는 상대 위치 임베딩)에 대해 알아보겠습니다.
- 그 전에 지금까지 배운 내용을 하나의 그림으로 정리하면 다음과 같습니다. 트랜스포머 인코더가 출력하는 것은 결국 각 토큰에 대한 임베딩 벡터라는 것을 기억하고, 인코더 블록 여러 개를 순서대로 쌓아 구성한다는 것도 잊지 마세요.

![스크린샷 2025-04-14 오후 9 26 28](https://github.com/user-attachments/assets/cdf411f9-6803-43a4-9ff4-8e0ff5be3283)

- 그럼 이제 디코더 블록의 구성에 대해 알아보겠습니다.


## 디코더 블록

- 트랜스포머 디코더는 몇 가지 차이점을 제외하면 인코더와 매우 비슷합니다. 그중 하나가 인코더가 출력한 임베딩 벡터를 입력으로 받는 멀티 헤드 어텐션 층입니다. 이 층은 디코더에서 받은 벡터를 쿼리로 사용하고, 인코더의 출력을 키와 값으로 사용합니다. 그래서 **크로스 어텐션**<sup>cross attention</sup>이라고 부르기도 합니다.

![스크린샷 2025-04-14 오후 9 26 43](https://github.com/user-attachments/assets/3cee17ce-41cd-446d-a649-05b5a0f1be5d)

- 디코더 블록에서는 크로스 어텐션 층이 인코더의 멀티 헤드 어텐션 층과 피드포워드 네트워크 사이에 배치됩니다. 크로스 어텐션 층의 전후에도 잔차 연결이 있습니다. 크로스 어텐션을 추가한 디코더 블록을 그림으로 나타내면 다음과 같습니다.

![스크린샷 2025-04-14 오후 9 26 59](https://github.com/user-attachments/assets/89588085-adde-4cb8-973f-6f7564ea235d)

- 디코더 블록 역시 하나만 존재하는 것이 아니라 여러 개가 반복적으로 쌓여 전체 디코더 모델을 구성합니다. 따라서 인코더 블록의 출력은 첫 번째 디코더 블록뿐만 아니라 반복되는 모든 디코더 블록에 전달됩니다. 이를 그림으로 간단히 나타내면 다음과 같습니다. 

![스크린샷 2025-04-14 오후 9 35 25](https://github.com/user-attachments/assets/97eeeea2-cb7a-48a2-99b6-8f1c842c9e11)

- 디코더 블록이 출력하는 값도 인코더 블록과 마찬가지로 동일한 크기의 은닉 벡터입니다. 따라서 여러 개의 디코더 블록을 반복해서 쌓을 수 있습니다. 그다음 해결하고자 하는 작업에 층을 마지막 디코더 블록 다음에 놓습니다. 예를 들어 텍스트 분류 작업에서는 마지막 디코더 블록 뒤에 밀집층을 추가해 여러 클래스 중 하나를 예측하도록 설계할 수 있습니다. 
- 디코더 블록과 인코더 블록의 또 다른 차이점은 맨 왼쪽의 멀티 헤드 어텐션에서 디코더 입력을 처리하는 방식입니다. 이를 설명하기 위해 한-영 번역 모델을 예로 들어 보겠습니다. 인코더의 입력이 "I love you"와 같은 영어 문장이라면, 이 문장 전체가 인코더에 전달됩니다. 인코더는 각 토큰을 분석하여 각각의 의미를 포함한 은닉 벡터를 만들고, 이를 디코더 블록에게 전달합니다.
- 하지만 디코더는 자기회귀 모델의 방식을 따라 한 번에 하나의 토큰만 생성합니다. 예를 들어, 모델이 올바르게 번역한다고 가정하면, 첫 번째 타임스텝에서 디코더는 "나는"을 출력하고, 두 번째 타임스텝에서는 "나는"이 입력되어 "너를"을 출력합니다. 세 번째 타임스텝에서는 "나는 너를"이 입력되어 "사랑한다"가 출력될 것입니다. 이 과정을 그림으로 나타내면 다음과 같습니다.

![스크린샷 2025-04-14 오후 9 47 46](https://github.com/user-attachments/assets/a25a5d52-68c0-422a-ab71-327835e98a46)

- 모델을 훈련할 때의 상황은 실제 번역을 수행할 때와 다릅니다. "I love you"의 올바른 번역(타깃값)인 "나는 너를 사랑한다"를 디코더에 한 번에 전달하고, 각 토큰에서 다른 토큰을 예측하도록 모델을 훈련합니다. 디코더는 "나는"에서 "너를"을 예측하고, "나는 너를"을 사용해 "사랑한다"를 예측하도록 훈련되는 것입니다. 이 훈련 과정은 모델의 출력을 정답(타깃)과 비교하여 오차를 줄이는 기존의 방식과 동일합니다.
- 하지만 디코더가 다음에 출력할 정답을 이미 알게 되면 올바른 학습이 이루어질 수 없습니다. "나는" 이라는 토큰을 처리할 때 "너를 사랑한다"라는 정답을 미리 볼 수 있다면, 모델이 단순히 정답을 복사하는 방식으로 학습될 위험이 있습니다. 이를 방지하기 위해 디코더의 첫 번째 멀티 헤드 어텐션 층에서는 **마스킹**<sup>masking</sup> 처리를 합니다. 즉, 디코더가 한 타임스텝에서 어텐션 점수를 계산할 때 현재 토큰 까지만 참고하고, 이후의 토큰을 볼 수 없도록 제한하는 것입니다. 이런 이유 때문에 디코더 블록의 첫 번째 멀티 헤드 어텐션 층을 **마스크드 멀티 헤드 어텐션**<sup>masked multi-head attention</sup> **층**이라고 부릅니다.
- 마지막으로 인코더와 디코더를 합쳐 트랜스포머 전체 모델을 그림으로 나타내 보겠습니다.

![스크린샷 2025-04-14 오후 9 57 58](https://github.com/user-attachments/assets/755d0839-03a1-4b6a-86cf-eebbae3422cc)

- 지금까지 트랜스포머 모델의 핵심 요소와 전체 구조를 하나씩 살펴보았습니다. 확실히 이전에 배웠던 합성곱 신경망이나 순환 신경망과는 크게 다르군요. 처음 접하는 개념이 많아 다소 생소하게 느껴질 수도 있습니다. 만약 이해가 잘 되지 않는다면, 이 절을 다시 처음부터 차근차근 읽어 보세요.
- 트랜스포머는 어텐션, 층 정규화, 잔차 연결, 드롭아웃 같은 기술을 효과적으로 조합하여 새로운 구조를 만들었고, 이를 통해 인코더-디코더 모델에서 순환층을 완전히 제거했습니다. 이제 트랜스포머 모델은 순환 신경망 기반의 인코더-디코더 모델보다 훨씬 높은 성능을 보여주며, 텍스트 처리 분야의 표준 모델로 자리 잡았습니다.
- 트랜스포머 모델은 대규모 텍스트 데이터셋을 학습하며, 매우 많은 모델 파라미터를 가지고 있습니다. 이런 모델을 **대규모 언어 모델**<sup>large language model, LLM</sup>이라 부릅니다. 예상할 수 있듯이, 이런 모델을 훈련하는 데는 많은 자원과 비용이 필요합니다. 하지만 다행히 이미 훈련된 모델을 가져다 사용할 수 있는 방법이 있습니다. 다음 절에서는 미리 훈련된 트랜스포머 모델을 활용해 텍스트 요약 작업을 직접 수행해 보도록 하겠습니다.