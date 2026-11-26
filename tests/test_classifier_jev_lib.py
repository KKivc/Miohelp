"""Jev 建题与取分纯函数:题号与 TOPIC_NAMES 顺序一一对应,缺题必须炸响。"""
import pytest
from langchain_core.decisions import ClassifierResponse

from app.core.taxonomy import TOPIC_NAMES
from scripts.classifier.jev_lib import build_questions, scores_from


def _response(scores: dict[str, float]) -> ClassifierResponse:
    return ClassifierResponse(model="jev-latest", answers={
        qid: {"type": "noul", "noul": p} for qid, p in scores.items()})


def test_one_question_per_topic():
    questions = build_questions()
    assert len(questions) == len(TOPIC_NAMES)
    for i, name in enumerate(TOPIC_NAMES):
        assert f"「{name}」" in questions[f"topic_{i}"].instructions


def test_scores_follow_topic_order_not_answer_order():
    n = len(TOPIC_NAMES)
    shuffled = {f"topic_{i}": i / 100 for i in reversed(range(n))}
    assert scores_from(_response(shuffled)) == [i / 100 for i in range(n)]


def test_missing_answer_raises():
    partial = {f"topic_{i}": 0.1 for i in range(len(TOPIC_NAMES) - 1)}
    with pytest.raises(KeyError):
        scores_from(_response(partial))
