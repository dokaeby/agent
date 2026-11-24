# LangGraph 기술 개요

## 정의

LangGraph는 LangChain 생태계의 그래프 기반 에이전트 프레임워크다.
노드(함수)와 엣지(전이)로 워크플로우를 명시적으로 정의한다.

## 핵심 개념

### StateGraph
- 상태(TypedDict)를 공유하는 그래프
- 각 노드는 상태를 읽고 부분 업데이트를 반환
- 상태는 자동으로 병합됨

### 노드 (Node)
- 상태를 받아 처리하고 업데이트된 필드를 반환하는 함수
- async 함수 지원

### 엣지 (Edge)
- 노드 간 전이 규칙
- 조건부 분기(add_conditional_edges)로 동적 라우팅 가능

### 체크포인터 (Checkpointer)
- SqliteSaver, PostgresSaver 등으로 상태 영속화
- thread_id별로 대화 상태를 저장/복원
- 프로세스 재시작 후에도 이어서 실행 가능

### Human-in-the-Loop (HITL)
- interrupt()로 실행을 일시정지
- Command(resume=...)로 재개
- 승인/거절/수정 요청 가능

## 장점

- 복잡한 상태ful 워크플로우에 최적
- 체크포인터로 영속성 내장
- HITL 지원
- LangSmith와 통합된 트레이싱

## 단점

- 학습 곡선이 가파름
- 노드 함수가 상태 객체에 묶여 테스트 복잡
- 추상화 계층이 많아 디버깅 어려움
