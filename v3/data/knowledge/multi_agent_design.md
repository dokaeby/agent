# 멀티에이전트 시스템 설계 원칙

## 핵심 원칙

### 1. 역할 분리 (Separation of Concerns)
각 에이전트는 명확한 단일 책임을 가져야 한다.
- Planner: 계획 수립만
- Researcher: 정보 수집만
- Writer: 글쓰기만
- Reviewer: 평가만

### 2. 최소 권한 (Least Privilege)
각 에이전트는 필요한 도구만 접근 가능해야 한다.
- Researcher: read 도구만 (search_docs)
- Publisher: write 도구 포함 (save_report)
- ALLOW_WRITE_TOOLS=false로 전역 차단 가능

### 3. 비용/시간 상한
무한 루프와 비용 폭주를 방지한다.
- BudgetTracker: 토큰/비용 상한
- Timeout: 에이전트당 시간 제한
- Max iterations: 도구 호출 루프 상한

### 4. 실패 격리 (Failure Isolation)
한 에이전트의 실패가 전체 파이프라인을 중단시키지 않도록 한다.
- 각 에이전트 호출을 try-catch로 감싸기
- 재시도 가능한 오류와 불가능한 오류 구분
- 부분 결과라도 반환

### 5. 관측 가능성 (Observability)
각 단계의 입출력과 소요 시간을 기록한다.
- 트레이싱 (LangSmith, JSONL)
- 로깅 (구조화된 로그)
- 메트릭 (토큰 사용량, 비용, 지연시간)

## 안티패턴

- 에이전트가 너무 많은 도구에 접근 (권한 과다)
- 에이전트 간 직접 통신 (슈퍼바이저를 거치지 않음)
- 상태 공유 없이 매번 전체 컨텍스트 전송 (비용 낭비)
- 에러 처리 없이 예외 전파 (한 에이전트 실패 = 전체 실패)
