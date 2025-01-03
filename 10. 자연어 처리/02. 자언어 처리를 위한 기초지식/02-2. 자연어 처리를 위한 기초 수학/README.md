## 자연어 처리를 위한 기초 수학 ‒ 랜덤 변수(Random Variable)
- <b>랜덤 변수(Random Variable)</b>은 확률론에서 특정 사건이 발생할 확률을 나타냅니다.
- 예를 들어 주사위를 던졌을때 1이 나올 확률을 다음과 같이 표현할 수 있습니다.
- 𝑃(𝑥=1)=1/6

## 자연어 처리를 위한 기초 수학 ‒ 결합 확률(joint probability)

- <b>결합 확률(joint probability)</b>은 여러 개의 사건이 동시에 일어날 확률을 의미합니다.
- 예를 들어 두 개의 주사위 A와 B를 던졌을때 A 주사위는 1, B 주사위는 2가 나올 결합 확률을 다음과 같이 표현할 수 있습니다.
- 𝑃(𝐴=1, 𝐵=2)
- 이때 주사위 던지기는 각각의 주사위 던지기가 다른 주사위 던지기 결과에 영향을 끼치지 않습니다. 이를 두 사건이 <b>독립(independent)</b>이라고 표현합니다.
- 두 사건이 독립일 경우 다음의 조건을 만족합니다.
- 𝑃(𝐴, 𝐵)=P(A)P(B)

## 자연어 처리를 위한 기초 수학 ‒ 조건부 확률(conditional probability)

- 조건부 확률(conditional probability)은 특정 사건이 발생했을 때 다른 사건이 발생할 확률을 의미합니다.

![스크린샷 2024-12-08 오후 3 10 52](https://github.com/user-attachments/assets/199b463a-8e21-42c1-85a1-08d166837de8)

- 예를 들어 아래 수식은, A 주사위가 1이라는 결과가 나왔을 때, B라는 주사위가 2라는 결과가 나올 확률을 의미합니다.

![스크린샷 2024-12-08 오후 3 11 01](https://github.com/user-attachments/assets/67149e7e-5e22-443a-b81f-a5e0cabc9bfe)

- 조건부 확률은 자연어 처리에서 광범위하게 사용되는 개념입니다.


## 자연어 처리를 위한 기초 수학 ‒ MLE(Maximum Likelihood Estimation)

- <b>MLE(Maximum Likelihood Estimation)-최대가능도추정</b>-은 어떤 현상이 발생했을 때 그 현상이 발생할 확
률이 가장 높은 우도(Likelihood)를 추정하는 방법론입니다.

![스크린샷 2024-12-08 오후 3 12 47](https://github.com/user-attachments/assets/f8c70ceb-20a2-4b70-93ec-7f7e3c7825f7)

- 예를 들어, 어떤 주머니에서 3개의 공을 꺼냈을때 빨간공 2개, 초록색공 1개가 나왔다면 이공을 꺼낸 주머니에 빨간공과 초록색공 몇개가 있어야만 **이런 현상이 발생할 확률이 가장 높은지를 추정**하는 가는 것이 Maximum Likelihood Estimation(MLE)입니다.
- MLE는 머신러닝과 자연어처리 분야에 광범위하게 사용됩니다.

![스크린샷 2024-12-08 오후 3 13 23](https://github.com/user-attachments/assets/6f8ec427-f6c6-4f4d-896e-a55668cd3d51)

