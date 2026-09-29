"""노트북 공통 설정: .env 로드, MLflow 트레이싱 연결 및 모델 생성."""

import os

from langchain_openai.chat_models.codex import _ChatOpenAICodex

## 맨처음 실행 시, 아래 명령어로 ChatGPT 로그인 필요 (브라우저 OAuth, 토큰은 DEFAULT_STORE_PATH 에 저장·자동 갱신)
from langchain_openai.chatgpt_oauth import DEFAULT_STORE_PATH
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from dotenv import load_dotenv
import mlflow

load_dotenv()

DEFAULT_PROVIDER = "Google"  # Google, Anthropic, Codex 중 하나 선택
DEFAULT_MODEL    = {
    "Google"   : "gemini-3.1-flash-lite",  # Google 모델 이름
    "Codex"   : "gpt-5.5",                 # Codex  모델 이름
    "Anthropic": "claude-sonnet-5",        # Claude 모델 이름

}   

def setup_mlflow(experiment: str | None = None) -> None:
    """MLflow 서버에 연결하고 LangChain/LangGraph 자동 트레이싱을 켠다."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", os.getenv("MLFLOW_TRACKING_URL", "http://localhost:5000")))
    mlflow.set_experiment(experiment or os.getenv("MLFLOW_EXPERIMENT_NAME", "langchain-study"))
    # run_tracer_inline=True: ainvoke 에서도 트레이서가 메인 태스크에서 돌아야 _ModelTagHandler 가 활성 트레이스를 찾을 수 있음
    mlflow.langchain.autolog(run_tracer_inline=True)
    mlflow.tracing.disable_notebook_display()  # 트레이스는 서버에만 기록, 셀 출력에 Trace UI 표시 안 함


class _ModelTagHandler(BaseCallbackHandler):
    """채팅 모델 호출 시 현재 MLflow 트레이스에 llm.provider / llm.model 태그를 단다.

    한 트레이스에서 여러 모델을 쓰면 "Google, Anthropic" 처럼 누적된다.
    """

    run_inline = True                              # async 에서도 트레이서와 같은 컨텍스트에서 실행
    _seen: dict[str, dict[str, list[str]]] = {}    # trace_id -> {"llm.provider": [...], "llm.model": [...]}

    def __init__(self, provider: str, model_name: str):
        self.provider   = provider
        self.model_name = model_name

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        span = mlflow.get_current_active_span()
        if span is None:   # 트레이싱이 꺼져 있으면 아무것도 안 함
            return

        seen = self._seen.setdefault(span.trace_id, {"llm.provider": [], "llm.model": []})
        for key, value in (("llm.provider", self.provider), ("llm.model", self.model_name)):
            if value not in seen[key]:
                seen[key].append(value)
        mlflow.update_current_trace(tags={key: ", ".join(values) for key, values in seen.items()})


def get_model(
                provider  : str = DEFAULT_PROVIDER,
                model_name: str | None = None,   # None 이면 provider 별 DEFAULT_MODEL 사용
                **kwargs
            ) -> ChatGoogleGenerativeAI | _ChatOpenAICodex | ChatAnthropic:

    model_name = model_name or DEFAULT_MODEL.get(provider)
    kwargs["callbacks"] = [
                            *(kwargs.get("callbacks") or []), 
                            _ModelTagHandler(provider, model_name)
                        ]

    if provider == "Google":
        return ChatGoogleGenerativeAI(
            model=model_name,
            **kwargs,
        )

    elif provider == "Codex":

        # 로그인은 최초 1회만 터미널에서 실행 (브라우저 OAuth, 토큰은 DEFAULT_STORE_PATH 에 저장·자동 갱신)
        #   uv run python -c "from langchain_openai.chatgpt_oauth import login_chatgpt; login_chatgpt()"
        if not DEFAULT_STORE_PATH.exists():
            raise FileNotFoundError(
                f"ChatGPT 토큰이 없습니다({DEFAULT_STORE_PATH}). 터미널에서 login_chatgpt() 를 먼저 실행하세요."
            )
        return _ChatOpenAICodex(
            model_name=model_name,
            **kwargs,
        )


    elif provider == "Anthropic":
        return ChatAnthropic(
            model=model_name,
            max_tokens=16000,
            **kwargs,
        )


    else:
        raise ValueError(f"지원되지 않는 provider: {provider}. Google, Codex, Anthropic 중 하나를 선택하세요.")