# v3 — 멀티에이전트 협업 파이프라인

역할이 다른 에이전트 5개가 협업해 **리서치 리포트**를 생성하는 샘플 프로젝트.
프레임워크 프리 오케스트레이터로 5가지 협업 패턴을 직접 구현하고,
MCP 서버/클라이언트, 운영 가드레일, 프레임워크 비교 문서를 포함한다.

## 실행 화면

```
사용자: "LangGraph의 상태 관리 특징을 조사해줘"
  → planner     : 주제를 3~5개 하위 질문으로 분해        (Plan-and-Execute)
  → researcher×N: 각 하위 질문을 병렬 조사               (병렬 서브에이전트)
                 search_docs / get_current_time (MCP 도구)
  → writer      : 조사 노트를 종합해 초안 작성
  → reviewer    : 0~1점 채점 + 피드백                     (Reviewer/Critic)
      └─ 미달 시 → writer에게 피드백 전달 후 재작성       (Handoff)
  → publisher   : 최종 리포트 저장                        (save_report, write 권한)
```

## 패턴 ↔ 코드 매핑

| 요구 패턴 | 구현 위치 |
|---|---|
| Supervisor (관리자 위임) | `orchestrator.py` — `Orchestrator.run()`이 전체 흐름 제어 |
| Handoff (에이전트 간 인계) | reviewer 피드백 → writer 재작성 루프, publisher로 최종 인계 |
| Plan-and-Execute | `PLANNER_PROMPT`가 JSON 배열로 하위 질문 분해 → 실행 → 종합 |
| 병렬 서브에이전트 | `asyncio.gather`로 researcher N개 동시 실행 |
| 리뷰어/비평가 | `REVIEWER_PROMPT` 채점 → `review_pass_score` 미달 시 재작성 (최대 `max_review_rounds`) |

## 멀티에이전트가 정말 필요한가? (`docs/00-multi-agent-decision.md`)

이 프로젝트가 단일 에이전트가 아닌 이유를 먼저 따져 본다:

- **컨텍스트 분리**: researcher는 도구 결과(검색 발췌)로 컨텍스트가 오염되는데,
  writer/reviewer는 깨끗한 컨텍스트에서 글쓰기/비평에 집중해야 한다.
- **서로 다른 도구 권한**: researcher는 읽기만, publisher만 `save_report`(write) 허용.
  권한 분리가 곧 보안 경계다.
- **비용/지연**: 하위 질문 조사는 서로 독립적이라 병렬화로 지연을 줄인다.
- **검증 루프**: 작성자와 비평가를 분리하면 자기 채점 편향을 줄일 수 있다.

단, 이 규모(5역할)에서도 과하면 안 된다 — 문서에 "언제 멀티에이전트가 과한지" 기준도 정리했다.

## 디렉터리 구조

```
multiagent/           # 핵심 패키지
  orchestrator.py     # 슈퍼바이저 — 패턴 5종 + 파이프라인 실행
  agents.py           # BaseAgent + 역할별 시스템 프롬프트 5종
  llm.py              # LLMClient — OpenAI 래퍼, 비용/토큰 추적, 재시도
  tools.py            # @tool 레지스트리 + MCP 도구 브릿지 + 권한 검사
  mcp_server.py       # MCP 서버 (FastMCP, stdio) — search_docs/get_current_time/
                      #   save_report/list_reports
  mcp_client.py       # stdio로 서버를 spawn해 tools/call 호출
  guardrails.py       # BudgetTracker, 타임아웃, 재시도, 인젝션 방어, 권한 분리
  knowledge.py        # 로컬 지식베이스 키워드 검색 (data/knowledge/*.md)
  config.py           # pydantic-settings (.env)
  service.py          # FastAPI: POST /reports, GET /reports/{id}
  cli.py              # python -m multiagent.cli run "주제" / serve
frameworks/
  langgraph_impl.py   # 같은 파이프라인의 LangGraph 버전 (비교용)
  COMPARISON.md       # LangGraph/CrewAI/OpenAI Agents SDK/Claude Agent SDK 비교
  __init__.py
docs/
  00-multi-agent-decision.md   # "멀티에이전트가 필요한가" 분석
data/knowledge/       # MCP search_docs가 검색하는 문서 4종
tests/                # pytest 30개 — API 키 없이 전체 파이프라인 검증
```

## 운영 가드레일 (`guardrails.py`)

| 이슈 | 구현 |
|---|---|
| 무한 루프 | researcher `MAX_TOOL_ROUNDS`, 리뷰어 `max_review_rounds` 상한 |
| 비용 상한 | `BudgetTracker`가 모든 LLM 호출의 토큰/비용을 합산, `max_cost_usd` 초과 시 `BudgetExceeded` |
| 타임아웃 | 각 에이전트 호출에 `asyncio.wait_for` 적용 (`agent_timeout_sec`) |
| 재시도 | `LLMClient.chat` 내부 `with_retry` (지수 백오프, `max_retries`) |
| 프롬프트 인젝션 | `wrap_untrusted()`가 외부 텍스트를 `<untrusted>` 태그로 감싸고, 검색 결과는 데이터로만 취급하도록 프롬프트에 명시 |
| 권한 분리 | `AgentContext.allow_write` — `save_report` 같은 write 도구는 `ALLOW_WRITE_TOOLS=true`일 때만 등록/실행 |

## MCP (Model Context Protocol)

- **서버** (`mcp_server.py`): `FastMCP`로 4개 도구 노출 — `search_docs`, `get_current_time`, `save_report`, `list_reports`. stdio transport.
- **클라이언트** (`mcp_client.py`): `stdio_client`로 서버 프로세스를 spawn하고 `ClientSession`으로 `tools/call` 표준 메서드 호출.
- 에이전트는 MCP 도구를 일반 도구처럼 사용 — `register_mcp_tools()`가 레지스트리에 브릿지 등록.
- 수동 확인: `python -m multiagent.mcp_server` (stdio로 실행되며 JSON-RPC 대기)

> 참고: `mcp` 패키지는 2.x에서 `FastMCP`가 `MCPServer`로 이름이 바뀌었다.
> 이 프로젝트는 v1 API 기준이라 `pyproject.toml`에 `mcp>=1.8,<2`로 고정했다.

## 프레임워크 비교 (`frameworks/COMPARISON.md`)

같은 파이프라인을 프레임워크 프리(`orchestrator.py`)와 LangGraph
(`frameworks/langgraph_impl.py`)로 각각 구현해 추상화 차이를 비교한다.
CrewAI / OpenAI Agents SDK / Claude Agent SDK는 문서 기준으로 특징·적합 사례·선택 가이드를 정리했다.

## 설치 및 실행

```bash
uv sync
cp .env.example .env   # OPENAI_API_KEY 입력
```

```bash
# 리서치 리포트 생성 (CLI)
python -m multiagent.cli run "RAG와 GraphRAG의 차이점"

# FastAPI 서버
python -m multiagent.cli serve
# → POST /reports {"topic": "..."}

# 테스트 (API 키 불필요)
pytest

# LangGraph 버전 확인
python -m frameworks.langgraph_impl "주제"
```

## 환경 변수 (`.env.example` 참고)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | — | 필수 |
| `AGENT_MODEL` | `gpt-4o-mini` | 모든 에이전트 공용 모델 |
| `MAX_COST_USD` | `0.50` | 실행 1회 비용 상한 |
| `MAX_TOTAL_TOKENS` | `200000` | 실행 1회 토큰 상한 |
| `AGENT_TIMEOUT_SEC` | `60` | 에이전트 호출 타임아웃 |
| `MAX_RETRIES` | `2` | LLM 호출 재시도 |
| `MAX_REVIEW_ROUNDS` | `2` | reviewer↔writer 루프 상한 |
| `REVIEW_PASS_SCORE` | `0.7` | 리뷰 통과 점수 |
| `ALLOW_WRITE_TOOLS` | `false` | `save_report` 등 write 도구 허용 |

## 테스트

`tests/` 30개 — 외부 API 없이 `FakeLLM`/`FakeMCP` 주입으로 검증:

- `test_guardrails.py` — 예산 초과, 타임아웃, 재시도, 인젝션 래핑, 권한 분리
- `test_tools.py` — 도구 레지스트리, 권한 검사, MCP 브릿지
- `test_orchestrator.py` — 전체 파이프라인(계획→병렬조사→작성→리뷰→발행), 리뷰 루프, 예산 초과 경로
- `test_knowledge.py` — 지식베이스 검색 랭킹

```bash
pytest -q   # 30 passed
```

## 학습 포인트

- **왜 멀티에이전트인가**를 코드 전에 문서로 먼저 따진다 (`docs/00`)
- 5가지 협업 패턴을 프레임워크 없이 직접 구현해 원리를 이해한다
- MCP로 도구를 표준 프로토콜로 분리 — 서버/클라이언트가 프로세스 경계를 넘어 통신
- 운영 가드레일(비용/타임아웃/재시도/권한/인젝션)은 프로덕션 필수이며 여기서 직접 구현
- 같은 파이프라인을 LangGraph로 재구현해 추상화 프레임워크와 비교한다
