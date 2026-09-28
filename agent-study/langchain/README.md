# LangChain / LangGraph Study

LangChain·LangGraph 를 Claude 모델로 공부하고, 모든 실행을 MLflow(https://mlflow.dove-nest.com)에 트레이스로 남긴다.

| 파일 | 내용 |
|---|---|
| [01_langchain_basics.ipynb](01_langchain_basics.ipynb) | 모델 호출, 스트리밍, 프롬프트 템플릿(LCEL), 구조화 출력, 도구 호출 |
| [02_langgraph_basics.ipynb](02_langgraph_basics.ipynb) | StateGraph 기초, 조건 분기 라우팅, `create_agent` + 체크포인터 메모리 |
| [common.py](common.py) | `.env` 로드, `setup_mlflow()`, `get_model()` |

## 세팅

```bash
cd "Agent Study/Langchain"
uv sync                 # .venv 생성 + 의존성 설치
cp .env.example .env    # ANTHROPIC_API_KEY 입력
```

VS Code 에서 노트북을 열고 커널로 `.venv` 를 선택한다.

## 모델

기본값은 `claude-opus-5`. `.env` 의 `CLAUDE_MODEL` 로 바꾼다 (`claude-sonnet-5`, `claude-haiku-4-5` 등).

> Opus 5 는 `temperature`, `top_p` 같은 샘플링 파라미터를 받지 않는다(400 에러). `get_model()` 에 넘기지 말 것.

## MLflow 트레이싱

`setup_mlflow()` 가 `mlflow.langchain.autolog()` 를 켜므로 LangChain·LangGraph 호출은 전부 자동 기록된다.

- UI: https://mlflow.dove-nest.com → Experiments → `langchain-study` → **Traces**
- 서버 3.14.0 / 클라이언트 `mlflow>=3.14,<3.15`

## 다음에 공부할 것

- [ ] RAG: 문서 로더 → 벡터스토어 → 리트리버 도구
- [ ] LangGraph human-in-the-loop (`interrupt`)
- [ ] `mlflow.genai.evaluate()` 로 프롬프트/그래프 변경 전후 비교
- [ ] MLflow 프롬프트 레지스트리
