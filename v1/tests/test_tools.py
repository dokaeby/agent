"""도구 레지스트리 + 개별 도구 테스트."""

import json

import pytest

from agent import tools as t


@pytest.fixture(autouse=True)
def _clean_state(tmp_path, monkeypatch):
    """입력: pytest 내장 tmp_path/monkeypatch 픽스처 / 출력: 없음.

    동작: 모든 테스트 전에 자동 실행(autouse)된다.
    - _MEMORY를 비워 이전 테스트의 remember 결과가 새지 않게 한다.
    - WORKSPACE를 tmp_path로 바꿔 실제 프로젝트 파일을 건드리지 않는다.
    - tmp_path에 note.txt/sub/a.txt 픽스처 파일을 만든다.
    """
    t.reset_memory()
    monkeypatch.setattr(t, "WORKSPACE", tmp_path)
    (tmp_path / "note.txt").write_text("hello workspace", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text("nested", encoding="utf-8")


# --- calculator ---


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("2+3*4", 14),
        ("(2+3)*4", 20),
        ("2**10", 1024),
        ("10/4", 2.5),
        ("-5 + +3", -2),
        ("7//2", 3),
    ],
)
def test_calculator_ok(expr, expected):
    """입력: 산술식 문자열과 기대값(파라미터라이즈) / 출력: 없음(단언).

    동작: calculator가 우선순위/괄호/거듭제곱/나눗셈/단항부호/정수나눗셈을
    올바르게 계산하는지 확인한다.
    """
    assert t.calculator(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",
        "open('x').read()",
        "1; import os",
        "[1,2,3]",
        "'a'*10",
    ],
)
def test_calculator_rejects_non_arithmetic(expr):
    """입력: 산술이 아닌 표현식 문자열(파라미터라이즈) / 출력: 없음(단언).

    동작: 코드 실행/함수 호출/리스트/문자열 반복 등 _eval_node 화이트리스트
    밖 요소가 ValueError 또는 SyntaxError로 거부되는지 확인한다 —
    eval 미사용 설계의 핵심 안전성 테스트.
    """
    with pytest.raises((ValueError, SyntaxError)):
        t.calculator(expr)


# --- current_datetime ---


def test_current_datetime_default():
    """입력: 없음 / 출력: 없음(단언).

    동작: 인자 생략 시 기본 tz가 Asia/Seoul이고 iso 문자열에 'T'가
    포함되는지 확인한다.
    """
    out = t.current_datetime()
    assert out["timezone"] == "Asia/Seoul"
    assert "T" in out["iso"]


def test_current_datetime_bad_tz():
    """입력: 존재하지 않는 tz 이름 / 출력: 없음(단언).

    동작: ZoneInfoNotFoundError가 발생하는지 확인한다 —
    execute_tool 경유 시 이 예외가 에러 JSON으로 변환된다.
    """
    from zoneinfo import ZoneInfoNotFoundError

    with pytest.raises(ZoneInfoNotFoundError):
        t.current_datetime("Not/AZone")


# --- 파일 샌드박스 ---


def test_read_file_ok():
    """입력: fixture가 만든 note.txt / 출력: 없음(단언).

    동작: workspace 안 파일이 content와 함께 읽히는지 확인한다.
    """
    out = t.read_file("note.txt")
    assert out["content"] == "hello workspace"


def test_read_file_missing():
    """입력: 존재하지 않는 파일명 / 출력: 없음(단언).

    동작: FileNotFoundError가 발생하는지 확인한다.
    """
    with pytest.raises(FileNotFoundError):
        t.read_file("nope.txt")


@pytest.mark.parametrize("path", ["../pyproject.toml", "..\\pyproject.toml", "sub/../../x"])
def test_read_file_blocks_traversal(path):
    """입력: workspace 탈출 경로(유닉스/윈도우/중첩 변형) / 출력: 없음(단언).

    동작: _resolve_safe가 모든 변형을 ValueError로 차단하는지 확인한다 —
    경로 탈출 방어의 핵심 테스트.
    """
    with pytest.raises(ValueError):
        t.read_file(path)


def test_list_files():
    """입력: "sub" 디렉터리와 기본 경로 / 출력: 없음(단언).

    동작: 하위 디렉터리 목록에 a.txt가 있고, 루트 목록에 note.txt가
    포함되는지 확인한다.
    """
    out = t.list_files("sub")
    assert out["entries"] == ["a.txt"]
    out_root = t.list_files()
    assert "note.txt" in out_root["entries"]


# --- memory_store ---


def test_memory_remember_recall_list():
    """입력: remember/recall/list 시퀀스 / 출력: 없음(단언).

    동작: 저장→조회→목록이 모두 동작하고, 없는 key의 recall은
    value=None을 반환하는지 확인한다.
    """
    assert t.memory_store("remember", key="user", value="kim") == {"stored": "user"}
    assert t.memory_store("recall", key="user")["value"] == "kim"
    assert "user" in t.memory_store("list")["keys"]
    assert t.memory_store("recall", key="missing")["value"] is None


def test_memory_bad_action():
    """입력: 허용되지 않은 action("explode") / 출력: 없음(단언).

    동작: ValueError가 발생하는지 확인한다.
    """
    with pytest.raises(ValueError):
        t.memory_store("explode")


# --- 레지스트리/실행기 ---


def test_all_five_tools_registered():
    """입력: 없음 / 출력: 없음(단언).

    동작: all_tools()가 정확히 5개 도구를 반환하는지 확인한다 —
    @tool 데코레이터 등록이 모듈 임포트 시 모두 동작했는지 검증.
    """
    names = {tool.name for tool in t.all_tools()}
    assert names == {
        "calculator",
        "current_datetime",
        "read_file",
        "list_files",
        "memory_store",
    }


def test_tool_schemas_openai_shape():
    """입력: 없음 / 출력: 없음(단언).

    동작: tool_schemas() 결과가 OpenAI tools 파라미터 형식
    (type=function, parameters.type=object)을 만족하는지 확인한다.
    """
    schemas = t.tool_schemas()
    assert len(schemas) == 5
    for s in schemas:
        assert s["type"] == "function"
        assert s["function"]["name"]
        assert s["function"]["parameters"]["type"] == "object"


async def test_execute_tool_success_and_errors():
    """입력: 정상 인자/미등록 도구/깨진 JSON/도구 내부 오류 4가지 호출.
    출력: 없음(단언).

    동작: execute_tool이 모든 경우 예외를 던지지 않고 JSON 문자열을
    반환하는지 확인한다 — 성공은 {"result"}, 실패는 {"error"}.
    이것이 "모델이 에러를 보고 재시도"할 수 있게 하는 핵심 계약이다.
    """
    out = json.loads(await t.execute_tool("calculator", '{"expression": "6*7"}'))
    assert out["result"] == 42

    # 미등록 도구 → 에러 JSON (예외 전파 아님)
    out = json.loads(await t.execute_tool("no_such_tool", "{}"))
    assert "unknown tool" in out["error"]

    # 깨진 JSON 인자
    out = json.loads(await t.execute_tool("calculator", "{not json"))
    assert "invalid JSON" in out["error"]

    # 도구 내부 오류도 에러 JSON으로
    out = json.loads(await t.execute_tool("calculator", '{"expression": "1/0"}'))
    assert "error" in out
