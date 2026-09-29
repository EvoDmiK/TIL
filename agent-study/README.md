# Agent Study

LangChain·LangGraph 등 LLM 에이전트 프레임워크를 공부하고, 모든 실행을 MLflow(https://mlflow.dove-nest.com)에 트레이스로 남긴다.

| 경로 | 내용 |
|---|---|
| [langchain/01_langchain_basics.ipynb](langchain/01_langchain_basics.ipynb) | 모델 호출, 스트리밍, 프롬프트 템플릿(LCEL), 구조화 출력, 도구 호출 |
| [langchain/02_langgraph_basics.ipynb](langchain/02_langgraph_basics.ipynb) | StateGraph 기초, 조건 분기 라우팅, `create_agent` + 체크포인터 메모리 |
| [langchain/src/common.py](langchain/src/common.py) | `.env` 로드, `setup_mlflow()`, `get_model()` |
| `langgraph/` | LangGraph 심화 (준비 중) |

## 세팅

`agent-study` 폴더 전체가 uv 프로젝트 하나이고, 가상환경은 `agent-study/.venv` 를 같이 쓴다.

```bash
cd agent-study
uv sync                 # agent-study/.venv 생성 + 의존성 설치
cp .env.example .env    # API 키 입력
```

VS Code 에서 노트북을 열고 커널로 `agent-study/.venv/bin/python` 을 선택한다.

## 모델

`get_model(provider, model_name)` 으로 모델을 만든다. 기본 provider 는 `Google` 이고, 기본 모델은 `common.py` 의 `DEFAULT_MODEL` 에서 바꾼다.

| provider | 기본 모델 | 인증 |
|---|---|---|
| `Google` | `gemini-3.1-flash-lite` | `.env` 의 `GOOGLE_API_KEY` |
| `Anthropic` | `claude-sonnet-5` | `.env` 의 `ANTHROPIC_API_KEY` |
| `Codex` | `gpt-5.5` | ChatGPT 로그인 (최초 1회 `login_chatgpt()`) |

## MLflow 트레이싱

`setup_mlflow()` 가 `mlflow.langchain.autolog()` 를 켜므로 LangChain·LangGraph 호출은 전부 자동 기록된다.

- UI: https://mlflow.dove-nest.com → Experiments → `langchain-study` → **Traces**
- 서버 3.14.0 / 클라이언트 `mlflow>=3.14,<3.15`

## 다음에 공부할 것

- [ ] RAG: 문서 로더 → 벡터스토어 → 리트리버 도구
- [ ] LangGraph human-in-the-loop (`interrupt`)
- [ ] `mlflow.genai.evaluate()` 로 프롬프트/그래프 변경 전후 비교
- [ ] MLflow 프롬프트 레지스트리
