"""평가 비교 로직 테스트 — API 키 불필요."""

from rag_agent.eval.run import compare_reports


def test_compare_improved():
    """입력: baseline 점수 0.5, current 0.8 / 출력: improved.

    동작: 평균 점수가 올랐으면 verdict=improved.
    """
    baseline = {"avg_retrieval_recall": 0.5, "avg_answer_score": 0.5,
                "results": [{"id": "q1", "answer_score": 0.5}]}
    current = {"avg_retrieval_recall": 0.6, "avg_answer_score": 0.8,
               "results": [{"id": "q1", "answer_score": 0.8}]}
    comp = compare_reports(baseline, current)
    assert comp["verdict"] == "improved"
    assert comp["avg_score_delta"] == 0.3


def test_compare_regressed():
    """입력: baseline 0.8, current 0.5 / 출력: regressed."""
    baseline = {"avg_retrieval_recall": 0.8, "avg_answer_score": 0.8,
                "results": [{"id": "q1", "answer_score": 0.8}]}
    current = {"avg_retrieval_recall": 0.7, "avg_answer_score": 0.5,
               "results": [{"id": "q1", "answer_score": 0.5}]}
    comp = compare_reports(baseline, current)
    assert comp["verdict"] == "regressed"
    assert comp["avg_score_delta"] == -0.3


def test_compare_same():
    """입력: 동일 점수 / 출력: same."""
    r = {"avg_retrieval_recall": 0.7, "avg_answer_score": 0.6,
         "results": [{"id": "q1", "answer_score": 0.6}]}
    comp = compare_reports(r, r)
    assert comp["verdict"] == "same"
    assert comp["changed_items"] == []
