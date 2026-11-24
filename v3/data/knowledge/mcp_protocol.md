# MCP (Model Context Protocol) 기술 개요

## 정의

MCP는 AI 에이전트와 외부 도구/데이터 소스를 연결하는 개방 표준 프로토콜이다.
Anthropic이 제안했으며, 2025년 3월에 OpenAI, Google DeepMind 등 주요 AI 기업이 채택을 발표했다.

## 핵심 개념

### MCP Server
- 도구(tools), 리소스(resources), 프롬프트(prompts)를 제공
- stdio, SSE, Streamable HTTP transport 지원
- FastMCP(Python)로 쉽게 구현 가능

### MCP Client
- 서버의 도구를 발견하고 호출
- Claude Desktop, Cursor, VS Code 등에서 MCP 서버 연결 지원

### 도구 (Tools)
- 에이전트가 호출할 수 있는 함수
- JSON Schema로 입력 파라미터 정의
- 예: 파일 읽기, API 호출, 데이터베이스 쿼리

## 장점

- 표준화된 도구 연결 방식
- 플러그인 생태계 (수백 개의 MCP 서버 존재)
- 에이전트 간 도구 공유 가능
- 보안/권한 관리 표준화 가능

## 단점

- 프로토콜 복잡도 (JSON-RPC 기반)
- 서버 관리 부담 (직접 운영하거나 공개 서버 사용)
- 도구 발견(discovery) 메커니즘이 제한적
