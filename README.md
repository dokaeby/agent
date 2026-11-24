# agent

LLM 에이전트 학습 샘플 시리즈 — 점진적으로 복잡도가 올라가는 3개 버전.

| 버전 | 주제 | 스택 |
|---|---|---|
| [v1](v1/) | 프레임워크 없는 ReAct 에이전트 | 순수 OpenAI SDK, FastAPI, pytest |
| [v2](v2/) | RAG 에이전트 + LangGraph | ChromaDB, 하이브리드 검색, 체크포인터, HITL, 평가 데이터셋 |
| [v3](v3/) | 멀티에이전트 협업 파이프라인 | Supervisor/Handoff/Plan-and-Execute/병렬/리뷰어, MCP, 운영 가드레일 |

각 버전은 독립된 uv 프로젝트이며, 자체 README와 테스트를 포함한다.
