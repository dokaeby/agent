# 멀티에이전트 프레임워크 비교

> ⚠️ 프레임워크는 빠르게 변한다. 이 문서는 2025년 기준 학습용 요약이며,
> 실제 도입 전 반드시 각 공식 문서를 확인할 것.

같은 파이프라인(Planner→Researcher×N→Writer→Reviewer→Publisher)을
4가지 방식으로 구현했을 때의 비교.

## 구현 방식 비교

| | orchestrator.py (프레임워크 프리) | frameworks/langgraph_impl.py |
|---|---|---|
| 상태 관리 | 직접 (PipelineResult dataclass) | TypedDict + 자동 merge |
| 병렬 실행 | `asyncio.gather` | `Send` API (map-reduce) |
| 조건부 분기 | `for` 루프 + `if` | `add_conditional_edges` |
| 체크포인터 | 없음 (직접 구현 필요) | `SqliteSaver` 내장 |
| HITL | `interrupt()` 없음 — 수동 구현 필요 | `interrupt()` 내장 |
| 학습 곡선 | 낮음 — 모든 게 보임 | 중간 — 추상화 개념 필요 |
| 프로덕션 | 직접 만들어야 함 | 검증된 런타임 |

## 프레임워크별 특징

### LangGraph (사용 중)

- **철학**: 그래프(노드+엣지)로 워크플로우를 명시. 상태는 TypedDict.
- **강점**: 체크포인터(영속화), HITL(`interrupt()`), 조건부 분기, Send 병렬 —
  복잡한 상태ful 워크플로우에 최적.
- **약점**: 노드 함수 시그니처가 상태 객체에 묶여 테스트가 복잡.
  학습 곡선이 가장 가파름.
- **적합**: 상태 관리가 중요한 장기 실행 워크플로우, HITL 필요한 경우.
- **공식**: https://langchain-ai.github.io/langgraph/

### CrewAI

- **철학**: "Crew"(역할 있는 에이전트들) + "Process"(순차/계층/병렬)로 선언.
- **강점**: 역할(role/goal/backstory) 기반 선언이 직관적. `Crew().kickoff()` 한 줄로 실행.
  프로토타이핑에 가장 빠름.
- **약점**: 상태 관리/체크포인터가 약함. 복잡한 조건부 로직은 오히려 불편.
  LangChain 생태계와 독립적이라 통합이 필요.
- **적합**: 역할 분담이 명확한 단기 태스크, 빠른 프로토타입.
- **공식**: https://docs.crewai.com/

### OpenAI Agents SDK (구 Swarm 계승)

- **철학**: Agent + Handoff + Guardrails를 최소 추상화로 제공.
- **강점**: `handoff()` 함수로 에이전트 간 위임이 가장 직관적.
  `input_guardrails`/`output_guardrails`로 검증 내장. Responses API 기반.
- **약점**: 상태 영속화/복잡한 그래프는 직접 구현. OpenAI 모델에 최적화.
- **적합**: OpenAI 생태계, handoff 패턴이 핵심인 경우.
- **공식**: https://openai.github.io/openai-agents-python/

### Claude Agent SDK

- **철학**: Claude Code의 에이전트 런타임을 SDK로 제공. subprocess 기반.
- **강점**: MCP 네이티브 지원(`mcp_servers` 옵션), in-process 커스텀 툴,
  파일시스템/쉘 도구 내장. Anthropic 생태계 최적화.
- **약점**: Python에서 subprocess로 Claude Code를 호출 — 레이턴시/비용 높음.
  커스텀 파이프라인 제어가 제한적.
- **적합**: Claude Code 수준의 도구 생태계가 필요한 경우.
- **공식**: https://code.claude.com/docs/en/agent-sdk/python

## 선택 가이드

| 상황 | 추천 |
|---|---|
| 상태ful 장기 워크플로우, HITL, 체크포인터 | **LangGraph** |
| 역할 기반 빠른 프로토타입 | **CrewAI** |
| OpenAI 모델 + handoff 중심 | **OpenAI Agents SDK** |
| Claude 생태계 + MCP 도구 | **Claude Agent SDK** |
| 학습/이해 목적 | **프레임워크 프리** (orchestrator.py) |

## 이 프로젝트의 선택

- `orchestrator.py`: 모든 패턴을 직접 구현해 원리를 학습
- `frameworks/langgraph_impl.py`: 같은 파이프라인을 LangGraph로 재구현해 추상화 비교
- 실제 프로덕션에서는 LangGraph의 체크포인터/HITL이 필수적이므로 LangGraph 권장
