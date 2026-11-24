# API 연동 가이드 — 내부 서비스 공통 규약

## 기본 원칙

모든 내부 API는 RESTful 설계 원칙을 따르며, JSON 형식으로 요청/응답한다.

- Base URL: `https://api.internal.company.com/v1`
- 인증: `Authorization: Bearer <token>` 헤더
- Content-Type: `application/json; charset=utf-8`
- 모든 응답에 `X-Request-Id` 헤더 포함 (문제 추적용)

## 인증 토큰 발급

1. `POST /auth/token`에 `client_id`와 `client_secret` 전송
2. 응답의 `access_token`을 모든 API 요청에 포함
3. 토큰 유효기간은 1시간 — 만료 전 `POST /auth/refresh`로 갱신

## 에러 응답 형식

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "필드 'name'은 필수입니다",
    "details": [{"field": "name", "issue": "required"}]
  }
}
```

HTTP 상태 코드:
- 200: 성공
- 201: 생성됨
- 400: 잘못된 요청 (검증 실패)
- 401: 인증 실패
- 403: 권한 없음
- 404: 리소스 없음
- 429: 요청 한도 초과 (Rate Limit)
- 500: 서버 오류

## Rate Limiting

분당 1000회 요청 제한. 초과 시 `429` 응답과 함께 `Retry-After` 헤더가 포함된다.
응답 헤더 `X-RateLimit-Remaining`으로 남은 요청 수를 확인할 수 있다.

## 페이지네이션

목록 API는 커서 기반 페이지네이션을 사용한다:

```
GET /users?limit=20&cursor=eyJpZCI6MTIzfQ
```

응답에 `next_cursor`가 포함되며, `null`이면 마지막 페이지다.
