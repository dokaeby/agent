# Frame-free Agent

LangChain/LangGraph 없이 **순수 OpenAI SDK**로 만든 ReAct(tool-use) 에이전트 샘플.
도구 호출 루프, 스트리밍, 토큰·비용 계산, FastAPI 서빙, pytest까지 직접 구현한다.

## 학습 목표 매핑

| 주제 | 구현 위치 |
|---|---|
| asyncio | `agent/agent.py`, `agent/llm.py` 전체가 async |
| Pydantic | `LLMTurn`/`ToolCallSpec`/`Settings`/`ChatRequest` |
| FastAPI | `agent/server.py` (앱 팩토리 + DI) |
| 패키지 관리(uv) | `pyproject.toml` + `uv sync` |
| 데코레이터/레지스트리 | `@tool` → `_REGISTRY` (`agent/tools.py`) |
| LLM API 직접 사용 | `OpenAIChatClient` (`agent/llm.py`) |
| 구조화 출력 | tool_calls → `ToolCallSpec` 정규화, 툴 결과 JSON |
| 토큰/비용 계산 | `estimate_cost_usd` (모델별 per-1M 가격표) |
| Tool Calling 루프 직접 구현 | `Agent.run` / `Agent.run_stream` (ReAct) |
| 툴 스키마 설계 | 각 도구의 JSON Schema `parameters` |
| 에러 처리 | `execute_tool` — 모든 실패를 에러 JSON으로 모델에 반환 |
| 화이트리스트 검증 | `_REGISTRY`에 없는 도구명은 실행 불가 |
| 스트리밍 | `client.stream()` — 텍스트 델타 + 툴콜 조각 누적 |
| pytest | `tests/` — FakeClient 주입으로 API 호출 없이 루프 검증 |

## 구조

```
agent/
  config.py    # pydantic-settings (.env 로드)
  llm.py       # OpenAIChatClient: complete()/stream() → LLMTurn 정규화 + 비용
  tools.py     # @tool 레지스트리 + 5개 도구 + execute_tool (화이트리스트)
  agent.py     # ReAct 루프: run() / run_stream()
  server.py    # FastAPI: POST /chat, GET /tools, GET /healthz
  cli.py       # REPL 채팅 (python -m agent.cli)
tests/         # FakeClient로 외부 API 없이 전 루프 검증
workspace/     # read_file/list_files 샌드박스 루트
```

## 도구 5종

| 도구 | 설명 |
|---|---|
| `calculator` | AST 화이트리스트 안전 산술 평가 (`eval` 미사용) |
| `current_datetime` | IANA 타임존 현재 시각 |
| `read_file` | workspace/ 내 파일 읽기 (`..` 탈출 차단) |
| `list_files` | workspace/ 목록 |
| `memory_store` | 인메모리 키-값 remember/recall/list |

## 실행

```bash
uv sync                              # 의존성 설치
cp .env.example .env                 # OPENAI_API_KEY 설정

python -m agent.cli                  # REPL 대화
python -m agent.cli "3*(4+5)는?"     # 1회 질문

uvicorn agent.server:app --port 8000 # FastAPI 서버
# POST /chat {"message": "..."} → reply + iterations + tokens + cost_usd
```

## ReAct 루프 핵심 (`agent/agent.py`)

```
messages = [system, history..., user]
while iterations < MAX:
    turn = client.complete(messages, tools=schemas)
    messages.append(assistant_msg_with_tool_calls)   # OpenAI 규약
    if turn.tool_calls:
        for call in turn.tool_calls:
            output = execute_tool(call.name, call.arguments)  # JSON 문자열
            messages.append({"role":"tool", "tool_call_id":..., "content":output})
    else:
        return turn.content
# 한도 초과 시 tools=None으로 최종 답변 강제
```

설계 포인트:
- **모든 툴 결과/오류는 JSON 문자열**로 ToolMessage에 넣는다 — 모델이 스스로
  인자를 고쳐 재시도할 수 있다. 예외를 그대로 올리지 않는다.
- **화이트리스트**: `_REGISTRY`에 없는 이름은 실행되지 않는다.
- **스트리밍 툴콜**: OpenAI는 툴콜도 조각으로 스트림한다 — `index`로 묶어
  `name`/`arguments`를 누적한다 (`OpenAIChatClient.stream`).
- **비용**: 매 턴의 `usage`를 누적해 모델별 가격표로 USD 환산.

## 테스트

```bash
pytest -q    # 35개: 도구/레지스트리/루프/스트림/서버 (외부 API 호출 없음)
ruff check . # 린트
```

`FakeClient`가 `ChatClient` 프로토콜을 구현해 스크립트된 `LLMTurn`을 반환 —
실제 OpenAI 호출 없이 루프 전체를 검증한다. 이것이 `ChatClient` Protocol을
분리해 둔 이유다.

## 다음 단계 아이디어

- `read_file` 결과가 길면 요약/청킹 추가
- 툴 결과를 대화에 남기지 않고 벡터DB 검색으로 교체 (RAG)
- LangGraph의 `StateGraph`와 이 루프 비교해보기
- `memory_store`를 SQLite/파일로 영속화
