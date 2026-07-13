# Neural Blocks Lab

블록을 추가, 삭제, 정렬하면서 신경망 구조와 학습 결과를 실시간으로 확인하는 브라우저 기반 AI 교육 플랫폼 MVP입니다.

## 실행

의존성을 설치한 뒤 로컬 서버로 실행합니다.

```bash
cd ai-learning-platform
npm install
python3 -m http.server 8770 --bind 127.0.0.1
```

브라우저에서 다음 주소를 엽니다.

```text
http://127.0.0.1:8770
```

## 현재 기능

- Dense + ReLU, Tanh, Sigmoid 블록 추가
- 은닉층 삭제 및 순서 변경
- 뉴런 수 1-32 조절
- XOR, 원형, 선형 분류 데이터
- 데이터 개수와 노이즈 조절
- SGD, Momentum, RMSProp, Adam
- 실제 forward pass / backpropagation
- 실시간 결정경계
- 선택 입력점의 뉴런 활성화와 연결 가중치
- Train / validation loss 곡선
- Train / validation accuracy
- MLP, CNN, RNN, GAN 모델 패밀리 템플릿
- CNN: Conv2D, Conv2DTranspose, pooling, flatten, global average pooling
- RNN: SimpleRNN, LSTM, GRU, Embedding
- 공통: Dense, Dropout, Batch Normalization, Reshape, Activation
- 13개 activation 함수
- 각 레이어의 tensor shape 실시간 검증
- CNN/RNN/GAN TensorFlow.js 모델 compile 및 파라미터 계산
- GAN Generator / Discriminator branch 개별 편집
- CNN 8x8 선 방향 이미지 분류 실시간 학습
- RNN 12-step 상승/하락 시계열 분류 실시간 학습
- GAN 2차원 원형 데이터 adversarial training
- GAN 목표 분포와 생성 샘플 실시간 비교

## 모델 패밀리 실행 범위

### MLP

현재 실제 학습 엔진까지 연결되어 있습니다.

- 블록 수정
- 역전파
- optimizer
- 결정경계
- 활성화
- train / validation 지표

### CNN / RNN / GAN

레이어 편집 결과를 실제 TensorFlow.js 모델과 학습 세션에 연결합니다.

- 레이어 카탈로그
- 레이어 추가, 삭제, 순서 변경
- 레이어 파라미터 수정
- tensor shape propagation
- 잘못된 shape 조합 차단
- TensorFlow.js 모델 생성
- 실제 파라미터 수 계산
- GAN branch별 모델 생성
- CNN/RNN `model.fit()` 기반 분류 학습
- GAN Generator/Discriminator 개별 optimizer 기반 적대적 학습
- 실제 loss 및 accuracy 실시간 표시
- 레이어 구조를 유지한 가중치 초기화

## Activation 함수

- ReLU
- Leaky ReLU
- ELU
- SELU
- Tanh
- Sigmoid
- Softmax
- Linear
- Swish
- GELU
- Hard Sigmoid
- Softplus
- Softsign

## 시각화 원칙

은닉층이 있는 신경망의 손실 함수는 수십에서 수천 차원의 파라미터 공간에 존재합니다. 이를 하나의 고정된 3D 표면이라고 표시하면 실제 학습과 다른 그림이 됩니다.

이 MVP에서는 다음 정보를 직접 시각화합니다.

- 입력 공간의 실제 결정경계
- 각 레이어의 실제 활성화
- 실제 연결 가중치의 부호와 크기
- 실제 train / validation loss

추후 3D 기능은 `두 파라미터 고정 slice`, `PCA trajectory`, `loss landscape projection`처럼 표현 방식을 명시한 별도 모듈로 추가하는 것이 맞습니다.

## 검증

```bash
npm run test:all
```

`test.mjs`는 선형, XOR, 원형 데이터에서 학습 전후 손실 감소와 validation accuracy를 확인합니다.

`catalog-test.mjs`는 다음을 검증합니다.

- MLP, CNN, RNN 모델의 실제 TensorFlow.js 생성
- 모든 activation 함수 compile
- GAN Generator / Discriminator 생성
- 입력과 출력 tensor shape
- 파라미터 수

`tf-training-test.mjs`는 CNN/RNN의 실제 손실 감소와 validation accuracy, GAN의 유한한 Generator/Discriminator loss, 학습 중 모델 교체 시 지연된 메모리 해제를 검증합니다.
