# RAG Agent v2 — LangGraph 기반 RAG 에이전트

v1(프레임워크 없는 ReAct)에서 한 단계 올라, **LangGraph + RAG + 메모리 + HITL + 평가**를 갖춘 샘플 프로젝트.

## 요구사항 커버리지

| 요구사항 | 구현 |
|---|---|
| RAG: 청킹 | `chunking.py` — 재귀적 분할(문단→문장→문자) + overlap |
| RAG: 임베딩 | `llm.py` — OpenAI `text-embedding-3-small` 배치 처리 |
| RAG: 벡터 DB | `vectorstore.py` — ChromaDB persistent (로컬, 서버 불필요) |
| RAG: 하이브리드 검색 | `retrieval.py` — dense + BM25(`rank_bm25`) → RRF 융합 |
| RAG: 리랭킹 | `retrieval.py` — LLM 리랭커(top-N 관련성 0-10점 재정렬) |
| LangGraph: 노드/엣지 | `graph.py` — StateGraph 7노드 |
| LangGraph: 상태 스키마 | `state.py` — TypedDict + `add_messages` 리듀서 |
| LangGraph: 체크포인터 | `runner.py` — `SqliteSaver` (thread_id별 영속화) |
| LangGraph: 조건부 분기 | `grade` → retrieve 재시도 or generate |
| LangGraph: HITL | `approve` 노드 — `interrupt()` + `Command(resume=...)` |
| 메모리: 단기 | `memory.py` — 대화 요약(8메시지 이상 시 자동 압축) |
| 메모리: 장기 | `memory.py` — 사용자 사실 sqlite 영구 저장 |
| 관측 | `tracing.py` — JSONL 트레이서 + LangSmith 자동 연동 |
| 평가 | `eval/run.py` — 데이터셋 24개 + LLM-as-judge + 회귀 비교 |

## 그래프 구조

```
START → rewrite → retrieve → grade ─┬─(무관 & 재시도 가능)→ retrieve
                                    └─(관련)→ generate → memory_update
                                                                 → extract_facts
                                                                 → approve (HITL)
                                                                       ├─ approved → END
                                                                       └─ rejected → rewrite (피드백 반영)
```

## 실행

```bash
uv sync && cp .env.example .env   # OPENAI_API_KEY 입력

python -m rag_agent.cli ingest    # data/docs/ 문서 → 벡터DB+BM25 인제스천
python -m rag_agent.cli ask "연차 며칠이야?"   # 1회 질문
python -m rag_agent.cli chat                   # 대화형 (HITL 승인 포함)
python -m rag_agent.cli eval                   # 평가 데이터셋 24개 실행
```

## 평가 워크플로우 (핵심 과제)

```bash
# 1. 베이스라인 측정 (프롬프트 변경 전)
python -m rag_agent.cli eval
# → evals/report_<timestamp>.json 저장

# 2. 프롬프트 수정 (예: graph.py의 SYSTEM_PROMPT나 generate 프롬프트)

# 3. 재측정 + 비교
python -m rag_agent.eval.run --dataset evals/dataset.jsonl \
    --baseline evals/report_이전.json --output evals/report_이후.json

# 출력 예시:
#   점수 변화: +0.12 → improved
#   q07: 0.5 → 0.9 (+0.40)
#   q15: 0.8 → 0.6 (-0.20)  ← 회귀 항목 식별
```

**측정 지표**:
- `retrieval_recall`: 기대 문서가 검색 결과에 포함됐는가 (검색 품질)
- `answer_score`: LLM-as-judge 0-1 채점 (답변 품질)
- 항목별 delta로 어떤 질문이 좋아지고 나빠졌는지 추적

## 디렉터리 구조

```
rag_agent/
  config.py      # pydantic-settings
  llm.py         # OpenAI chat + embeddings
  chunking.py    # 재귀적 텍스트 분할
  vectorstore.py # ChromaDB 래퍼
  bm25.py        # BM25Okapi + pickle 캐시
  retrieval.py   # dense+BM25+RRF+LLM 리랭크
  memory.py      # 단기 요약 + 장기 사실(sqlite)
  state.py       # LangGraph 상태 스키마
  graph.py       # 노드/엣지/조건부/HITL
  tracing.py     # JSONL 트레이서
  runner.py      # 세션 러너 (checkpointer+tracer)
  ingest.py      # 문서 인제스천 파이프라인
  eval/run.py    # 평가 실행기 + judge + 비교
  cli.py         # ingest/ask/chat/eval 커맨드
data/docs/       # 샘플 문서 6종 (지식 베이스)
evals/
  dataset.jsonl  # 평가 데이터셋 24개
tests/           # API 키 불필요 단위 테스트 19개
```

## 테스트

```bash
pytest -q        # 19개 — 청킹/BM25/VectorStore/메모리/그래프/HITL/평가비교
ruff check .     # 린트
```

## LangSmith 연동 (선택)

`.env`에 추가하면 langgraph가 자동으로 LangSmith에 트레이스를 보낸다:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=rag-agent-v2
```

로컬 트레이스는 `./traces/YYYY-MM-DD.jsonl`에 노드별 소요시간과 함께 기록된다.

## 설계 포인트

- **Graceful degradation**: BM25 인덱스 없음 → dense만, LLM 리랭크 실패 → RRF 순서, grade 판정 실패 → relevant 간주
- **HITL 재시도 루프**: 거절+피드백이 rewrite 노드로 돌아가 쿼리를 재작성 — 최대 반복은 checkpointer가 관리
- **메모리 분리**: 단기(대화 요약)는 state에, 장기(사용자 사실)는 sqlite에 — 세션 재시작 후에도 프로필 유지
- **평가 격리**: eval은 checkpointer 없이 실행해 세션 오염 방지, thread_id를 항목별로 고유하게
