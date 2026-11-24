"""도구 레지스트리 + 5개 샘플 도구.

패턴:
- @tool 데코레이터가 함수를 Tool(name, description, JSON Schema, func)로 등록
- tool_schemas()가 OpenAI tools 파라미터 형식으로 변환
- execute_tool()이 모델이 보낸 name+JSON 인자를 화이트리스트 검증 후 실행
  → 미등록 도구/깨진 인자는 예외로 죽지 않고 에러 JSON을 모델에 돌려준다
  (모델이 스스로 수정할 수 있게).
"""

from __future__ import annotations

import ast
import inspect
import json
import operator
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# --- 레지스트리 ---


@dataclass
class Tool:
    """등록된 도구 하나의 정의.

    입력: @tool 데코레이터 인자 + 래핑된 함수
    출력: name/description/parameters/func 4필드 객체
    동작: parameters는 OpenAI function calling의 JSON Schema object.
    func는 동기 또는 async 함수 모두 가능(execute_tool이 awaitable 처리).
    """

    name: str
    description: str
    parameters: dict[str, Any]  # OpenAI function parameters JSON Schema
    func: Callable[..., Any]


_REGISTRY: dict[str, Tool] = {}
"""도구명 → Tool 매핑. 모듈 임포트 시 @tool 데코레이터가 자동 채운다.
화이트리스트의 실체 — 이 dict에 없는 이름은 절대 실행되지 않는다."""


def tool(name: str, description: str, parameters: dict[str, Any]):
    """입력: name(도구명), description(모델에게 보여줄 설명), parameters(JSON Schema).
    출력: 데코레이터 함수.

    동작: 함수를 _REGISTRY에 등록하고 원본 함수를 그대로 반환한다
    (래핑하지 않으므로 시그니처/타입힌트가 보존된다).
    같은 name으로 재등록하면 덮어쓴다.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY[name] = Tool(name=name, description=description, parameters=parameters, func=func)
        return func

    return decorator


def get_tool(name: str) -> Tool:
    """입력: 도구명 / 출력: Tool 객체.

    동작: 등록되지 않은 이름이면 KeyError 발생.
    현재 코드에서 직접 호출하는 곳은 없고(테스트/디버깅용),
    실제 실행 경로는 execute_tool이 _REGISTRY.get으로 조회한다.
    """
    if name not in _REGISTRY:
        raise KeyError(f"알 수 없는 도구: {name}")
    return _REGISTRY[name]


def all_tools() -> list[Tool]:
    """입력: 없음 / 출력: 등록 순서의 Tool 리스트 사본.

    동작: _REGISTRY.values()의 list() — 반환 후 원본 레지스트리를
    수정해도 영향받지 않는 스냅샷이다.
    """
    return list(_REGISTRY.values())


def tool_schemas() -> list[dict[str, Any]]:
    """입력: 없음 / 출력: OpenAI tools 파라미터 형식의 dict 리스트.

    동작: 각 Tool을 {"type":"function","function":{name,description,parameters}}
    형태로 변환한다. Agent.__init__에서 1회 호출해 self._tools에 캐시하고,
    매 LLM 호출마다 messages와 함께 전달한다.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in _REGISTRY.values()
    ]


async def execute_tool(name: str, arguments_json: str) -> str:
    """입력: name(모델이 요청한 도구명), arguments_json(모델이 만든 JSON 문자열).
    출력: 항상 JSON 문자열 — {"result": ...} 또는 {"error": ...}.

    동작:
    1. _REGISTRY 화이트리스트 조회 — 없으면 error JSON 반환(실행 안 함).
    2. arguments_json 파싱 — 깨진 JSON/비객체면 error JSON 반환.
    3. func(**args) 호출 — awaitable이면 await.
    4. 어떤 예외든 잡아 {"error": "Type: msg"}로 변환해 반환.
       → 예외가 루프로 전파되지 않아 대화가 죽지 않고,
         모델이 에러 메시지를 보고 인자를 고쳐 재시도할 수 있다.
    """
    registered = _REGISTRY.get(name)
    if registered is None:
        return json.dumps({"error": f"unknown tool: {name}"}, ensure_ascii=False)
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"invalid JSON arguments: {exc}"}, ensure_ascii=False)
    if not isinstance(args, dict):
        return json.dumps({"error": "arguments must be a JSON object"}, ensure_ascii=False)

    try:
        result = registered.func(**args)
        if inspect.isawaitable(result):
            result = await result
        return json.dumps({"result": result}, ensure_ascii=False, default=str)
    except Exception as exc:  # noqa: BLE001 - 도구 오류는 모델에 전달해야 함
        return json.dumps(
            {"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False
        )


# --- 도구 1: 안전한 산술 계산기 (eval 금지, AST 화이트리스트) ---

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    """입력: ast.parse(mode='eval')가 만든 AST 노드.
    출력: 계산된 숫자(int/float).

    동작: 허용 노드만 재귀 평가한다 — 숫자 상수, 이항연산(+ - * / // % **),
    단항연산(+/-). 이름/호출/속성 접근 등 나머지는 전부
    ValueError로 거부한다. eval()을 쓰지 않아 코드 실행이 원천 불가.
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"허용되지 않는 표현식 요소: {type(node).__name__}")


@tool(
    name="calculator",
    description="Evaluate an arithmetic expression. Supports + - * / // % ** and parentheses.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, e.g. '2 + 3 * (4 - 1)'",
            }
        },
        "required": ["expression"],
    },
)
def calculator(expression: str) -> float:
    """입력: expression(산술식 문자열).
    출력: 계산 결과 float.

    동작: ast.parse로 문법 검사 후 _eval_node로 화이트리스트 평가.
    문법 오류(SyntaxError)/허용 외 요소(ValueError)/0으로 나누기(ZeroDivisionError)는
    execute_tool이 잡아 에러 JSON으로 모델에 돌려준다.
    """
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree)


# --- 도구 2: 현재 날짜/시간 ---


@tool(
    name="current_datetime",
    description="Get the current date and time in a timezone (default Asia/Seoul).",
    parameters={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone name, e.g. 'Asia/Seoul', 'UTC', 'America/New_York'",
            }
        },
    },
)
def current_datetime(timezone: str = "Asia/Seoul") -> dict[str, str]:
    """입력: timezone(IANA tz 이름, 기본 'Asia/Seoul').
    출력: {"timezone","iso","human"} dict.

    동작: ZoneInfo로 tz를 만들고 현재 시각을 isoformat/사람이 읽는 형식으로 반환.
    잘못된 tz는 ZoneInfoNotFoundError → execute_tool이 에러 JSON로 변환.
    Windows에서는 tzdata 패키지가 필요(의존성에 포함됨).
    """
    tz = ZoneInfo(timezone)  # 잘못된 tz는 ZoneInfoNotFoundError → execute_tool이 에러 JSON로 변환
    now = datetime.now(tz)
    return {
        "timezone": str(tz),
        "iso": now.isoformat(),
        "human": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


# --- 도구 3/4: 샌드박스 파일 읽기/목록 (경로 탈출 차단) ---

WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"
"""파일 도구의 샌드박스 루트 — 프로젝트 루트의 workspace/ 디렉터리."""
MAX_READ_CHARS = 20_000
"""read_file이 반환하는 content의 최대 길이. 초과분은 잘리고 truncated=True."""


def _resolve_safe(path: str) -> Path:
    """입력: 모델이 준 상대 경로 / 출력: workspace 안의 절대 Path.

    동작: WORKSPACE / path를 resolve()한 뒤 is_relative_to로 검증한다.
    '../etc/passwd' 같은 탈출 시도나 절대경로 주입은 ValueError.
    이 검증이 없으면 모델이 임의 파일을 읽을 수 있으므로 보안상 필수.
    """
    candidate = (WORKSPACE / path).resolve()
    if not candidate.is_relative_to(WORKSPACE.resolve()):
        raise ValueError(f"path escapes workspace: {path}")
    return candidate


@tool(
    name="read_file",
    description="Read a UTF-8 text file inside the workspace directory.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path relative to workspace/"}},
        "required": ["path"],
    },
)
def read_file(path: str) -> dict[str, Any]:
    """입력: path(workspace 기준 상대경로).
    출력: {"path","content","truncated"} dict.

    동작: _resolve_safe로 샌드박스 검증 → is_file 확인 →
    UTF-8 텍스트를 읽어 MAX_READ_CHARS로 절단.
    없는 파일은 FileNotFoundError, 디렉터리는 is_file=False로 FileNotFoundError.
    """
    target = _resolve_safe(path)
    if not target.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    text = target.read_text(encoding="utf-8")
    truncated = len(text) > MAX_READ_CHARS
    return {"path": path, "content": text[:MAX_READ_CHARS], "truncated": truncated}


@tool(
    name="list_files",
    description="List files and directories inside the workspace directory.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Subdirectory relative to workspace/ (default '.')"}},
    },
)
def list_files(path: str = ".") -> dict[str, Any]:
    """입력: path(workspace 기준 하위 디렉터리, 기본 '.').
    출력: {"path","entries"} — entries는 정렬된 이름 리스트, 디렉터리는 '/' 접미사.

    동작: _resolve_safe 검증 → is_dir 확인 → iterdir() 결과를
    이름순 정렬해 반환. 파일이면 NotADirectoryError.
    """
    target = _resolve_safe(path)
    if not target.is_dir():
        raise NotADirectoryError(f"not a directory: {path}")
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
    return {"path": path, "entries": entries}


# --- 도구 5: 인메모리 키-값 메모리 ---

_MEMORY: dict[str, str] = {}
"""프로세스 수명 동안 유지되는 키-값 저장소. 서버 재시작 시 초기화된다."""


@tool(
    name="memory_store",
    description="Remember or recall a short fact by key. Actions: 'remember' (key+value), 'recall' (key), 'list' (no args).",
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["remember", "recall", "list"]},
            "key": {"type": "string", "description": "Fact key, e.g. 'user_name'"},
            "value": {"type": "string", "description": "Fact value (remember only)"},
        },
        "required": ["action"],
    },
)
def memory_store(action: str, key: str = "", value: str = "") -> dict[str, Any]:
    """입력: action(remember/recall/list), key, value.
    출력: action별 dict — remember: {"stored"}, recall: {"key","value"}, list: {"keys"}.

    동작: remember는 key 필수(없으면 ValueError)이고 _MEMORY에 저장.
    recall은 없는 key면 value=None. list는 정렬된 키 목록.
    알 수 없는 action은 ValueError → execute_tool이 에러 JSON으로 변환.
    """
    if action == "remember":
        if not key:
            raise ValueError("key is required for remember")
        _MEMORY[key] = value
        return {"stored": key}
    if action == "recall":
        return {"key": key, "value": _MEMORY.get(key)}
    if action == "list":
        return {"keys": sorted(_MEMORY)}
    raise ValueError(f"unknown action: {action}")


def reset_memory() -> None:
    """입력: 없음 / 출력: 없음.

    동작: _MEMORY를 비운다. 테스트 격리와 REPL 재시작용.
    """
    _MEMORY.clear()
