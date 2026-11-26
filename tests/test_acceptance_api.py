"""分类器验收 API:页面读的是落盘报告,不在 API 里重算;服务离线是一种状态,不是异常。"""
import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import acceptance

EVAL_REPORT = {
    "ran_at": "2026-11-26T12:00:00", "test_size": 2, "threshold": 0.8,
    "micro": {"p": 0.5, "r": 0.5, "f1": 0.5}, "macro": {"p": 0.5, "r": 0.5, "f1": 0.5},
    "classes": [], "total_cells": 34, "total_fp": 2, "total_fn": 2, "red_line_passed": False,
    "errors": [
        {"text": "退货运费谁出", "gold": ["运费"], "pred": ["退换货"], "missed": ["运费"],
         "extra": ["退换货"], "kind": "错位", "matrix_entries": 2},
        {"text": "运费险怎么赔", "gold": ["运费"], "pred": ["退换货"], "missed": ["运费"],
         "extra": ["退换货"], "kind": "错位", "matrix_entries": 2},
    ],
}


@pytest.fixture
def reports(tmp_path, monkeypatch):
    monkeypatch.setattr(acceptance, "CLASSIFIER_DIR", tmp_path)
    monkeypatch.setattr(acceptance, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(acceptance, "DATASET", tmp_path / "dataset")
    (tmp_path / "reports").mkdir()
    return tmp_path / "reports"


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    async def offline():
        return {"online": False, "detail": "stub"}

    async def no_db():
        raise RuntimeError("mysql 没起")
    monkeypatch.setattr(acceptance, "_probe_classifier", offline)
    monkeypatch.setattr("app.db.repository.topic_distribution", no_db)


@pytest.fixture
async def client():
    app = FastAPI()
    app.include_router(acceptance.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


def _write(reports, name, data):
    (reports / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


async def test_overview_without_reports_is_missing_not_error(client, reports):
    body = (await client.get("/api/acceptance/overview")).json()
    blocks = {b["key"]: b for b in body["blocks"]}
    assert body["classifier"]["online"] is False
    for key in ("eval", "threshold", "matrix", "errors", "classify"):
        assert blocks[key]["status"] == "missing"
    assert body["total"] == len(body["blocks"])


async def test_overview_gates_follow_reports(client, reports):
    _write(reports, "eval_report.json", EVAL_REPORT)
    _write(reports, "threshold_scan.json", {
        "scan": [{"threshold": 0.8, "micro_f1": 0.9}], "best_threshold": 0.8,
        "best_micro_f1": 0.9, "in_use_threshold": 0.8, "consistent": True})
    blocks = {b["key"]: b for b in (await client.get("/api/acceptance/overview")).json()["blocks"]}
    assert blocks["eval"]["status"] == "fail"        # red_line_passed=False
    assert blocks["threshold"]["status"] == "pass"   # consistent=True


async def test_errors_pairs_count_boundary_friction(client, reports):
    _write(reports, "eval_report.json", EVAL_REPORT)
    body = (await client.get("/api/acceptance/errors")).json()
    assert body["kinds"] == {"错位": 2}
    assert body["matrix_entries"] == 4
    assert body["pairs"][0] == {"missed": "运费", "grabbed": "退换货", "count": 2,
                                "severity": "中"}
