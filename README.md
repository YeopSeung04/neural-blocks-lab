# Neural Blocks Lab

블록을 추가, 삭제, 정렬하면서 신경망 구조와 학습 결과를 실시간으로 확인하는 브라우저 기반 AI 교육 플랫폼 MVP입니다.

## 실행

의존성을 설치한 뒤 로컬 서버로 실행합니다.

```bash
cd ai-learning-platform
npm install
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
npm run serve
```

브라우저에서 다음 주소를 엽니다.

```text
http://127.0.0.1:8770
```

전체 Python 의존성이 필요한 PostgreSQL·OIDC·LTI 검증 서버는 다음처럼 실행할 수 있습니다.

```bash
.venv/bin/python server.py --bind 127.0.0.1 --port 8770
```

### Docker + PostgreSQL

```bash
docker compose up --build
```

Compose는 PostgreSQL 16과 애플리케이션을 함께 실행합니다. 운영 환경에서는 `docker-compose.yml`의 개발용 비밀번호를 사용하지 말고 `.env.example`을 기준으로 Secret Manager나 배포 플랫폼의 비밀 환경변수를 연결해야 합니다.

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
- 5단계 Dataset Setup Wizard: Source, Schema, Process, Split, Inspect
- CSV 표 데이터 업로드와 다중 feature / target 컬럼 매핑
- 이미지 폴더 업로드와 resize / grayscale / pixel normalization
- 시계열 CSV 업로드와 다중 signal, timestamp, sequence window 설정
- Train / validation / test 분할과 고정 random seed
- Train 데이터로만 결측치 대체 및 scaling 통계 계산
- 표/이미지 stratified split과 시계열 chronological split
- 행 수, 클래스 수, 결측치, Tensor shape, 데이터 미리보기 검사
- 업로드 데이터를 실제 MLP/CNN/RNN 학습 세션에 연결
- 가중치 hover 시 `activation x weight`, bias, `z`, activation output 표시
- CPU, RAM, GPU, VRAM 또는 unified memory 실시간 모니터
- TensorFlow.js backend, tensor 개수, tensor memory, JS heap 표시
- 실제 계정 로그인과 대학별 tenant 격리
- HttpOnly 세션 쿠키, CSRF 검증, 인증 요청 rate limit
- 대학 관리자·교수·학생 역할 기반 접근 제어
- 대학 가입 코드와 강좌 가입 코드
- 강좌와 과제 생성, 현재 실험을 과제 시작 템플릿으로 저장
- 학생 실험 프로젝트 버전 저장과 과제 제출
- 모델 패밀리와 목표 accuracy 자동 조건 검사
- 교수 점수·피드백 입력과 학생 제출 이력
- SQLite 기반 강좌·과제·프로젝트·제출 영구 저장
- PostgreSQL `DATABASE_URL` 전환
- 교수·학생 이메일 초대와 강좌 자동 배정
- 강좌 학생 명단, 제출 수, 이메일 인증·로그인 방식 확인
- 이메일 인증, 인증 메일 재발송, 비밀번호 재설정
- SMTP 발송과 로컬 JSONL 개발 메일함
- tenant 감사 로그
- 대학별 OIDC 공급자 설정과 Authorization Code callback
- LTI 1.3 OIDC login initiation, signed launch, context-to-course 매핑

## 사용자 데이터

업로드 파일은 서버로 전송하지 않고 현재 브라우저 메모리에서 전처리합니다.

### CSV 표 데이터

- 숫자 feature 열 2개 이상
- binary target 열 1개
- 모든 선택 feature를 MLP 학습에 사용
- 결정경계에 표시할 X/Y feature를 별도로 선택
- 표시하지 않는 feature는 Train 중앙값으로 고정
- Stratified random, random, chronological 분할 선택
- 샘플: `samples/tabular_binary.csv`

### 이미지 폴더

다음 구조로 두 개 class 폴더를 준비합니다.

```text
dataset/
  class_a/
    image_001.png
  class_b/
    image_002.png
```

- 8, 16, 28, 32 크기 선택
- grayscale 1채널 변환
- `0~1` 또는 `-1~1` pixel normalization
- 클래스 비율을 유지하는 stratified train / validation / test 분할
- CNN 입력 shape와 자동 연결

### 시계열 CSV

- 숫자 signal 열 1개 이상
- binary target 열 1개
- 선택적 timestamp 또는 step 열
- 8, 12, 16, 24 step window
- stride 1, 2, 4, 8
- 시간순 chronological train / validation / test 분할
- 시계열 행은 무작위로 섞지 않음
- RNN/LSTM/GRU 입력 shape와 자동 연결
- 샘플: `samples/time_series_binary.csv`

현재 사용자 데이터 학습은 이진 분류를 지원합니다. Wizard의 다중 분류와 회귀 항목은 지원 범위를 오해하지 않도록 비활성 상태로 표시합니다.

## 교육 운영 MVP

상단 `수업 운영` 버튼에서 대학 수업 흐름을 시험할 수 있습니다.

### 대학 관리자·교수

- 새 대학 워크스페이스 개설
- 대학 가입 코드 확인
- 강좌 생성과 강좌별 학생 가입 코드 발급
- 강좌명, 강좌 코드, 학기 설정
- 과제 설명, 마감일, 필수 모델 패밀리, 목표 validation accuracy 설정
- 현재 신경망 구조를 과제 시작 템플릿으로 포함
- 학생 제출 결과와 자동 조건 검사 확인
- 0-100점 점수와 교수 피드백 저장

### 학생

- 대학 가입 코드로 계정 생성
- 강좌 가입 코드로 수업 참여
- 현재 신경망 구조, optimizer, 데이터 설정, 학습 지표를 프로젝트로 저장
- 같은 프로젝트의 여러 버전 저장
- 과제 시작 템플릿 불러오기
- 현재 실험을 과제에 제출
- 제출 시도, 점수와 교수 피드백 확인

교육 운영 데이터는 `.data/neural_blocks.db` SQLite 데이터베이스에 저장됩니다. 모든 강좌·과제·프로젝트·제출 쿼리는 로그인 사용자의 `tenant_id`로 제한되며, 학생에게는 대학·강좌 가입 코드가 다시 노출되지 않습니다.

비밀번호는 PBKDF2-HMAC-SHA256으로 해시하고 세션 원문은 데이터베이스에 저장하지 않습니다. 브라우저 세션은 JavaScript가 읽을 수 없는 HttpOnly 쿠키를 사용하며, 데이터 변경 API는 별도 CSRF 토큰을 확인합니다.

신규 비밀번호 계정은 이메일 인증 전까지 강좌 데이터 변경이 차단됩니다. SMTP 환경변수가 없으면 `.data/mail-outbox.jsonl`에 메일이 기록됩니다. 로컬 개발에서는 응답과 UI에 검증 토큰이 표시되지만 운영에서는 반드시 `NBL_EXPOSE_DEV_TOKENS=0`으로 설정해야 합니다.

### PostgreSQL

`NBL_DATABASE_URL`이 있으면 PostgreSQL을 사용하고, 없으면 `NBL_DATABASE_PATH` 또는 `.data/neural_blocks.db` SQLite를 사용합니다.

```bash
export NBL_DATABASE_URL='postgresql://user:password@host:5432/neural_blocks'
export NBL_BASE_URL='https://ai-lab.example.edu'
export NBL_SECURE_COOKIES=1
export NBL_EXPOSE_DEV_TOKENS=0
```

### 이메일

```bash
export NBL_SMTP_HOST='smtp.example.edu'
export NBL_SMTP_PORT=587
export NBL_SMTP_USER='neural-blocks'
export NBL_SMTP_PASSWORD='secret'
export NBL_SMTP_FROM='no-reply@example.edu'
export NBL_SMTP_TLS=1
```

### OIDC

관리자 화면에서 issuer, client ID, authorization endpoint, token endpoint, JWKS URI와 client secret 환경변수 이름을 등록합니다. Client secret 값 자체는 DB에 저장하지 않습니다.

```text
Callback URL: https://서비스주소/api/auth/oidc/callback
Login start:  https://서비스주소/api/auth/oidc/start
```

OIDC ID Token은 공급자의 JWKS로 서명, issuer, audience, 만료, nonce를 검증합니다.

### LTI 1.3

LMS 플랫폼에는 다음 Tool URL을 등록합니다.

```text
OIDC login initiation: https://서비스주소/api/auth/lti/login
Launch redirect URI:   https://서비스주소/api/auth/lti/launch
```

LTI launch는 서명된 ID Token, nonce, deployment ID, LTI version과 message type을 확인하고 LTI context를 내부 강좌에 매핑합니다. 현재 범위는 Resource Link·Deep Linking launch 기반이며 NRPS 학생 명단 동기화와 AGS 성적 역전송은 다음 단계입니다.

참조 규격:

- [OpenID Connect Core 1.0](https://openid.net/specs/openid-connect-core-1_0.html)
- [1EdTech LTI 1.3 Core](https://www.imsglobal.org/spec/lti/v1p3/)
- [1EdTech Security Framework](https://www.imsglobal.org/spec/security/v1p0/)

### 운영 전 추가 보강

현재 HTTP 서버는 파일럿과 단일 인스턴스 검증용입니다. 대학 서비스 운영 전에는 HTTPS reverse proxy, 관리형 PostgreSQL 백업·복구, 메일 전송 큐, 감사 로그 보존 정책, Secret Manager, provider 수정·폐기, 관리자 MFA, NRPS·AGS 연동, 부하 테스트가 필요합니다.

사용자가 업로드한 CSV와 이미지 원본은 실험 Snapshot에 포함하지 않습니다. Snapshot에는 모델 구조, 학습 설정, 데이터 요약과 지표만 저장되며 원본 데이터는 복원 시 다시 선택해야 합니다.

## 디바이스 성능 측정

`server.py`가 OS 값과 브라우저 값을 결합합니다.

- macOS: `top`, `IOAccelerator`
- NVIDIA GPU: `nvidia-smi`
- Windows/Linux: `psutil` 사용 권장
- 브라우저: TensorFlow.js `tf.memory()`, backend, JS heap

Windows/Linux에서 더 정확한 CPU/RAM 측정을 사용하려면 설치합니다.

```bash
python3 -m pip install -r requirements.txt
```

표준 브라우저는 시스템 전체 GPU 사용률과 실제 VRAM을 직접 제공하지 않습니다. OS bridge가 지원되지 않으면 UI thread 부하, JS heap, TensorFlow.js tensor memory만 표시하며 해당 값을 시스템 전체 사용률로 표시하지 않습니다.

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
- hover한 연결의 현재 forward 계산식
- 실제 train / validation loss
- 학습 중 디바이스 및 TensorFlow.js 메모리 변화

추후 3D 기능은 `두 파라미터 고정 slice`, `PCA trajectory`, `loss landscape projection`처럼 표현 방식을 명시한 별도 모듈로 추가하는 것이 맞습니다.

## 검증

```bash
npm run test:all
```

`test.mjs`는 선형, XOR, 원형 데이터에서 학습 전후 손실 감소와 validation accuracy를 확인합니다.

`data-pipeline-test.mjs`는 CSV parsing, 다중 feature, Train-only scaling, 명시적 train/validation/test 분할, binary label mapping, timestamp 정렬, 시계열 window, 이미지 폴더 label 구조를 검증합니다.

`education-api-test.mjs`는 API 클라이언트의 same-origin 요청과 CSRF 헤더, 서버 오류 변환을 검증합니다.

`backend_test.py`는 계정 생성과 로그인, 두 대학 tenant 격리, 학생의 대학·강좌 가입, 프로젝트 버전, 제출 자동 검사와 채점을 검증합니다.

`server_api_test.py`는 실제 HTTP 서버에서 HttpOnly·SameSite 세션 쿠키, CSRF 차단, 인증 세션 종료를 검증합니다.

`federation_test.py`는 OIDC/LTI authorization URL, LTI claim, RSA 서명 ID Token과 nonce 검증을 확인합니다. PyJWT crypto 의존성이 없으면 서명 검증 테스트만 건너뜁니다.

`postgres_test.py`는 `NBL_TEST_POSTGRES_URL`이 있을 때 PostgreSQL 스키마, 이메일 인증, 강좌, 교수 초대와 명단을 실제 DB에서 검증합니다.

```bash
NBL_TEST_POSTGRES_URL='postgresql://postgres:password@127.0.0.1:5432/testdb' \
NBL_TEST_POSTGRES_RESET=1 \
.venv/bin/python postgres_test.py
```

`catalog-test.mjs`는 다음을 검증합니다.

- MLP, CNN, RNN 모델의 실제 TensorFlow.js 생성
- 모든 activation 함수 compile
- GAN Generator / Discriminator 생성
- 입력과 출력 tensor shape
- 파라미터 수

`tf-training-test.mjs`는 CNN/RNN의 실제 손실 감소와 validation accuracy, 업로드 RNN의 명시적 chronological tensor split, GAN의 유한한 Generator/Discriminator loss, 학습 중 모델 교체 시 지연된 메모리 해제를 검증합니다.

`server_test.py`는 CPU/RAM/GPU 시스템 metric 응답 구조와 값 범위를 검증합니다.
