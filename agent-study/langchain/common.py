"""노트북 공통 설정: .env 로드, MLflow 트레이싱 연결, Claude 모델 생성."""

import os

import mlflow
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

DEFAULT_MODEL = "claude-opus-5"


def setup_mlflow(experiment: str | None = None) -> None:
    """MLflow 서버에 연결하고 LangChain/LangGraph 자동 트레이싱을 켠다."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.dove-nest.com"))
    mlflow.set_experiment(experiment or os.getenv("MLFLOW_EXPERIMENT_NAME", "langchain-study"))
    mlflow.langchain.autolog()


def get_model(**kwargs) -> ChatAnthropic:
    """ChatAnthropic 인스턴스. 모델은 CLAUDE_MODEL 환경변수로 바꿀 수 있다.

    Opus 5 는 temperature/top_p 같은 샘플링 파라미터를 받지 않으므로(400 에러) 넘기지 않는다.
    """
    return ChatAnthropic(
        model=os.getenv("CLAUDE_MODEL", DEFAULT_MODEL),
        max_tokens=16000,
        **kwargs,
    )
