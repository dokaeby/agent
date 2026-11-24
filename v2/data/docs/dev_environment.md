# 사내 협업 도구 및 개발 환경

## 버전 관리

모든 소스 코드는 Git으로 관리한다. 브랜치 전략은 GitHub Flow를 따른다:

- `main`: 항상 배포 가능한 상태를 유지한다
- `feature/*`: 새 기능 개발용 브랜치, 작업 완료 후 PR을 통해 main에 머지
- `hotfix/*`: 긴급 버그 수정용 브랜치

커밋 메시지는 Conventional Commits 형식을 사용한다: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:` 접두사.

## 협업 도구

- **Slack**: 실시간 커뮤니케이션. 채널은 `#general`, `#dev`, `#random`으로 구성
- **Jira**: 이슈 트래킹. 스프린트는 2주 단위로 운영
- **Confluence**: 문서화. ADR(Architecture Decision Record)은 반드시 기록
- **Figma**: 디자인 협업. 개발자도 직접 확인해야 함

## 개발 환경

로컬 개발은 Docker Compose로 통일한다:

```bash
docker compose up -d  # postgres, redis, mailhog 기동
pnpm install          # 프론트엔드 의존성
pnpm dev              # 개발 서버 (포트 3000)
```

Python 백엔드는 `uv`로 패키지를 관리한다: `uv sync`로 설치, `uv run`으로 실행.

## 코드 리뷰

모든 PR은 최소 1명의 승인과 CI 통과가 필요하다. 리뷰 요청 후 24시간 내 응답을 원칙으로 한다. 큰 PR(500줄 이상)은 작게 나누어 제출하는 것을 권장한다.
