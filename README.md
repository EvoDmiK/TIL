# TIL
Today I Learned — 공부한 내용을 주제별로 정리하는 저장소

대부분의 내용은 Jupyter 노트북(`.ipynb`)으로 정리한다. 이미지·데이터셋·모델 가중치는 `.gitignore`로 제외했기 때문에 일부 노트북은 데이터를 따로 받아야 다시 실행할 수 있다.

## 목차

- [agent-study](#agent-study) · [ai-study](#ai-study) · [kaggle](#kaggle) · [data-science](#data-science) · [db](#db)
- [python-study](#python-study) · [coding-test](#coding-test) · [bioinformatics](#bioinformatics) · [quant](#quant)
- [julia-study · mojo-study · rust-study](#다른-언어) · [assets](#assets)
- [개발 환경](#개발-환경) · [작성 규칙](#작성-규칙)

---

## agent-study
LLM 에이전트 프레임워크 공부. Gemini·Claude·GPT 모델로 실습하고, 모든 실행을 MLflow에 트레이스로 남긴다.

| 폴더 | 내용 |
|---|---|
| [langchain](agent-study/langchain) | LangChain 기초 — 모델 호출, 스트리밍, LCEL, 구조화 출력, 도구 호출 |
| [langgraph](agent-study/langgraph) | LangGraph 기초 — StateGraph, 조건 분기, `create_agent` + 체크포인터 |
| [common](agent-study/common) | 노트북 공통 코드 — `.env` 로드, MLflow 연결, 모델 생성 |

세팅 방법은 [agent-study/README.md](agent-study/README.md)를 참고한다.

## ai-study
인공지능 이론과 모델 구현 공부.

| 폴더 | 내용 |
|---|---|
| [CV](ai-study/CV) | 이상 탐지(PatchCore), 깊이 추정(DPT, GLPN), 의료 영상(MedMNIST, MONAI), OCR, 세그멘테이션(SAM, MaskFormer), 스타일 트랜스퍼(AdaIN), Few-shot(Siamese Network) |
| [Generative Model](ai-study/Generative%20Model) | GAN(CycleGAN — TF/PyTorch), Diffusion(Text-to-Video: NUWA, Make-A-Video) |
| [NLP](ai-study/NLP) | 한국어 방언 사전(SentencePiece), Seq2Seq + Attention, 문자열 유사도(코사인, 자카드, 레벤슈타인) |
| [ML](ai-study/ML) | 앙상블, HMM, 불균형 학습, 최적화, 추천 시스템, 도서 실습(파이썬 라이브러리를 활용한 머신러닝, 핸즈온 머신러닝 2판) |
| [Frameworks](ai-study/Frameworks) | Numpy로 신경망 밑바닥 구현, PyTorch(DNN·CNN·GNN), TensorFlow(RNN), JAX 기초 |
| [Metrics](ai-study/Metrics) | IoU·NMS, SSIM·PSNR |
| [Paper Review](ai-study/Paper%20Review) / [Paper Implementation](ai-study/Paper%20Implementation) | YOLO v1 리뷰와 구현 |
| [MultiModal](ai-study/MultiModal) | HydraNet |
| [Experiment](ai-study/Experiment) | 활성화 함수 비교(Mish vs Swish) |
| [js](ai-study/js) | Transformers.js로 웹 브라우저에서 AI 모델 실행해보기 (Django) |
| [utils](ai-study/utils) | 데이터 증강 등 공용 코드 |

## kaggle
Kaggle 데이터셋으로 한 실습.

| 폴더 | 내용 |
|---|---|
| [CV](kaggle/CV) | 이미지 분류(고양이 품종, 새, 음표 등), 의료 영상(폐렴, 뇌종양, 배아), 객체 탐지(거북이 얼굴) |
| [ML](kaggle/ML) | 정형 데이터 분류(효소, 신장결석, 단백질 계열 등 15개), 추천 시스템(Epic Store, Metacritic) |
| [NLP](kaggle/NLP) | 재난 트윗 분류, 닉네임 생성 |
| [AudioAI](kaggle/AudioAI) | 작곡가 분류, 음악 장르 분류(CNN) |
| [Visualization](kaggle/Visualization) | 영화, 세계 인구, NetCDF, 총기 사건 데이터 시각화 |
| [DOODLE](kaggle/DOODLE) | 넷플릭스, 자살 통계 데이터 탐색 |

## data-science
데이터 분석 도구와 도서 실습.

| 폴더 | 내용 |
|---|---|
| [파이썬 데이터 사이언스 핸드북](data-science/%5B파이썬%20데이터%20사이언스%20핸드북%5D) | Numpy 배열 생성과 속성 |
| [김도형의 데이터 사이언스 스쿨](data-science/%5B소문난%20명강의%20-%20김도형의%20데이터%20사이언스%20스쿨%5D%20) | LaTeX 수식 표현 |
| [Visualize](data-science/Visualize) | Plotly, HiPlot |
| [EDA](data-science/EDA) | 공공·개인 데이터 탐색(서울 지하철 승객, 출생 연도별 띠, 한국 지명 글자 빈도, 카톡방 대화 분석) |
| [Pyspark](data-science/Pyspark) | PySpark DataFrame |
| [Audio Data](data-science/Audio%20Data) | librosa로 오디오를 이미지로 변환 |

## db
데이터베이스 공부.

| 폴더 | 내용 |
|---|---|
| [SQL](db/SQL) | MySQL 연결, 김상형의 SQL 정복 실습, SQL 연습 문제, DB 데이터 분석, 네이버 웨일 방문 기록(SQLite) 분석 |
| [REDIS](db/REDIS) | Redis CRUD |
| [misc](db/misc) | DB 연결 설정과 유틸 코드 |

## python-study
파이썬 문법과 파이썬으로 하는 여러 가지 공부.

| 폴더 | 내용 |
|---|---|
| [python](python-study/python) | 내장 모듈(collections, itertools, 데코레이터) |
| [algorithm](python-study/algorithm) | 그리디, DFS·BFS, 누적 합, 투 포인터, 세그먼트 트리 (개념 + 연습 문제), 유전 알고리즘 |
| [Mathematics](python-study/Mathematics) | 선형대수(개발자를 위한 실전 선형대수학), 정수론, 집합론, 암호학(RSA), 푸리에 변환, 콜라츠 수열, 원과 접선 |
| [API](python-study/API) | ChatGPT API, 카카오 API, 마인크래프트 RCON(mcrcon) 접속 |
| [CV](python-study/CV) | 이미지 스티칭 |
| [Gradio](python-study/Gradio) | Gradio 앱 |
| [experiment](python-study/experiment) | 원소 검색 성능 비교, numpy 없이 argmin·argmax 구현, 셀룰러 오토마타(라이프 게임), 보로노이 + 노이즈 맵 생성, 랜덤으로 격자 채우기 시뮬레이션 |

## coding-test
알고리즘 문제 풀이 (Python).

| 폴더 | 내용 |
|---|---|
| [백준](coding-test/Python/백준%20) | Bronze·Silver·Gold 단계별 풀이 약 40문제 |
| [프로그래머스](coding-test/Python/프로그래머스) | 코딩테스트 연습(Lv.1~2), 2022 카카오 공채·인턴십 기출 |
| [LeetCode](coding-test/Python/leetcode) | 로마 숫자 변환, 전화번호 문자 조합 |
| [Project Euler](coding-test/Python/Project%20Euler) | 소인수, 삼각수, 대칭수 등 수학 문제 |

## bioinformatics
생명정보학 공부. 참고한 도서와 사이트는 [bioinformatics/README.md](bioinformatics/README.md)에 정리했다.

| 폴더 | 내용 |
|---|---|
| [Python](bioinformatics/Python) | Biopython으로 만나는 생물정보학 실습 — 서열 처리, SeqRecord·파일 포맷, 다중 서열 정렬, BLAST |
| [Weblogo](bioinformatics/Weblogo) | 서열 로고 시각화 |
| [utils](bioinformatics/utils) | MUSCLE 정렬 도구 |

## quant
금융 데이터 공부. 지금은 증권사 리포트 스크래핑 노트북 하나가 있다.

## 다른 언어

| 폴더 | 내용 |
|---|---|
| [julia-study](julia-study) | 변수·표현식, 함수 |
| [mojo-study](mojo-study) | Mojo 튜토리얼 |
| [rust-study](rust-study) | 변수, 함수, 제어 흐름. 설치 방법과 Jupyter Rust 커널 설정은 [rust-study/README.md](rust-study/README.md) 참고 |

## assets
노트북과 README에서 쓰는 이미지 모음 (AI 논문 리뷰, Numpy 신경망, OrganMNIST, Plotly, SQL).

---

## 개발 환경

- **agent-study**: [uv](https://docs.astral.sh/uv/) 프로젝트 하나로 관리한다. 의존성은 [agent-study/pyproject.toml](agent-study/pyproject.toml)에 있고, 하위 폴더가 모두 `agent-study/.venv`를 같이 쓴다.

  ```bash
  cd agent-study
  uv sync
  ```

- **그 외 파이썬 폴더**: `ai-study`, `data-science`, `db`, `python-study`, `bioinformatics/Python`에 있는 `requirements.txt`를 참고한다. 예전에 `pip freeze`로 뽑은 목록이라 최신 Python에서는 버전을 풀어서 설치해야 할 수 있다.

## 작성 규칙

- **노트북 이름**: 시리즈로 이어지는 공부는 `[주제 #번호] 제목.ipynb` 형식으로 짓는다. 예: `[Algorithm #001] Greedy Algorithm.ipynb`, `[Biopython #004]. BLAST.ipynb`
- **`[Doodle]` 접두어**: 예전 Doodle-Box 저장소에서 옮겨 온 가벼운 실험 노트북이다. 주제에 맞는 폴더로 나눠 넣었다.
- **커밋 메시지**: `타입. 내용` 형식으로 쓴다.

  | 타입 | 용도 |
  |---|---|
  | `study` | 공부 내용 추가·수정 |
  | `docs` | README 등 문서 |
  | `build` | 개발 환경, 의존성 |
  | `rename` | 파일·폴더 이동, 이름 변경 |
  | `chore` | 그 외 설정 (`.gitignore` 등) |
