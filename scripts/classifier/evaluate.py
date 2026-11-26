"""评测:留出测试集经 :8110(Jev)打分,出每类 P/R/F1 + 每类混淆矩阵 + 容错红线 + 判错样本。
运行:make classifier-eval(需 :8110 在线)。评测集扎在自家电商场景(dataset/test.jsonl),不引公开榜单。
产物一式两份:.md 给人读、.json 给验收页读(/acceptance/eval、/acceptance/errors),同一次评测同一份数。"""
import datetime as dt
import json
import pathlib

import httpx
import numpy as np

from app.core.taxonomy import LABEL2ID, NUM_CLASSES, SEVERITY, TOPIC_NAMES
from scripts.classifier.jev_lib import THRESHOLD

SERVICE = "http://127.0.0.1:8110/classify"
TEST = pathlib.Path("data/classifier/dataset/test.jsonl")
REPORTS = pathlib.Path("data/classifier/reports")
RED_LINES = {"严": 0.9, "中": 0.8}   # 档位 F1 红线;宽档不设线


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, _f1(p, r)


def _matrix(labels: list[list[str]]) -> np.ndarray:
    m = np.zeros((len(labels), NUM_CLASSES), dtype=bool)
    for i, row in enumerate(labels):
        for lb in row:
            m[i][LABEL2ID[lb]] = True
    return m


def _error(text: str, gold: list[str], pred: list[str]) -> dict:
    """错误方向性:漏打只记放跑、多打只记冤枉、错位两头都记——错例条数与矩阵笔数
    对不上就是从这来的(验收页要把这笔账摊开,别让读者自己减)。"""
    missed = [lb for lb in gold if lb not in pred]
    extra = [lb for lb in pred if lb not in gold]
    kind = "错位" if missed and extra else ("漏打" if missed else "多打")
    return {"text": text, "gold": list(gold), "pred": pred, "missed": missed, "extra": extra,
            "kind": kind, "matrix_entries": len(missed) + len(extra)}


def build_report(samples: list[dict], preds: list[list[str]], threshold: float) -> dict:
    """标准答案与预测标签 → 验收页读的报告。纯函数,不碰网络。"""
    gold, pred = _matrix([s["labels"] for s in samples]), _matrix(preds)
    tp, fp, fn = (gold & pred).sum(0), (~gold & pred).sum(0), (gold & ~pred).sum(0)
    tn = len(samples) - tp - fp - fn
    classes, per_class = [], []
    for i, name in enumerate(TOPIC_NAMES):
        p, r, f1 = _prf(int(tp[i]), int(fp[i]), int(fn[i]))
        per_class.append((p, r, f1))
        line = RED_LINES.get(SEVERITY[name])
        classes.append({
            "name": name, "severity": SEVERITY[name], "p": round(p, 4), "r": round(r, 4),
            "f1": round(f1, 4), "support": int(tp[i] + fn[i]), "red_line": line,
            # 宽档不设线 → passed 为 None,页面画 —,不画 ✅,免得读者以为它也过了某条线
            "passed": None if line is None else bool(f1 >= line),
            "tn": int(tn[i]), "fp": int(fp[i]), "fn": int(fn[i]), "tp": int(tp[i])})
    micro = _prf(int(tp.sum()), int(fp.sum()), int(fn.sum()))
    macro = np.mean(per_class, axis=0)
    return {
        "ran_at": dt.datetime.now().isoformat(timespec="seconds"),
        "test_size": len(samples), "threshold": threshold, "red_lines": RED_LINES,
        "micro": dict(zip(("p", "r", "f1"), (round(float(v), 4) for v in micro))),
        "macro": dict(zip(("p", "r", "f1"), (round(float(v), 4) for v in macro))),
        "classes": classes,
        "errors": [_error(s["text"], s["labels"], p)
                   for s, p in zip(samples, preds, strict=True) if set(p) != set(s["labels"])],
        "total_cells": len(samples) * NUM_CLASSES,
        "total_fp": int(fp.sum()), "total_fn": int(fn.sum()),
        "red_line_passed": all(c["passed"] is not False for c in classes),
    }


def _markdown(report: dict) -> str:
    mi, ma = report["micro"], report["macro"]
    lines = ["# 分类器评测报告(留出测试集)", "",
             f"测试集 {report['test_size']} 条;判定阈值 {report['threshold']}(验证集扫描所得)。", "",
             f"**micro**: P={mi['p']:.3f} R={mi['r']:.3f} F1={mi['f1']:.3f}  |  "
             f"**macro**: P={ma['p']:.3f} R={ma['r']:.3f} F1={ma['f1']:.3f}", "",
             "## 每类指标(容错红线:严档 F1 ≥ 0.9,中档 ≥ 0.8,宽档不设线)", "",
             "| 类目 | 档位 | P | R | F1 | support | 红线 |", "|---|---|---|---|---|---|---|"]
    for c in report["classes"]:
        flag = "—" if c["red_line"] is None else f"{c['red_line']} {'✅' if c['passed'] else '🔴 不达标'}"
        lines.append(f"| {c['name']} | {c['severity']} | {c['p']:.3f} | {c['r']:.3f} "
                     f"| {c['f1']:.3f} | {c['support']} | {flag} |")
    lines += ["", "## 每类混淆矩阵(TN FP / FN TP)", ""]
    lines += [f"- **{c['name']}**: TN={c['tn']} FP={c['fp']} FN={c['fn']} TP={c['tp']}"
              for c in report["classes"]]
    lines += ["", "## 判错样本(人工复核:错在哪一类?标注本身有没有毛病?)", ""]
    lines += [f"- {e['text']}\n  标准: {e['gold']}  预测: {e['pred']}" for e in report["errors"]]
    return "\n".join(lines)


def main() -> None:
    samples = [json.loads(l) for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    r = httpx.post(SERVICE, json={"texts": [s["text"] for s in samples]}, timeout=600)
    r.raise_for_status()
    preds = [res["labels"] for res in r.json()["results"]]
    report = build_report(samples, preds, THRESHOLD)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "eval_report.md").write_text(_markdown(report), encoding="utf-8")
    (REPORTS / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"micro-F1 {report['micro']['f1']:.4f} / macro-F1 {report['macro']['f1']:.4f};"
          f"报告已落 {REPORTS}/(.md + .json)")


if __name__ == "__main__":
    main()
