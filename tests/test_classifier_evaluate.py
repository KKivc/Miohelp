"""评测报告纯函数:指标口径与原 sklearn 版一致(已用 200 组随机样本对过),错例方向分对。"""
from scripts.classifier.evaluate import build_report


def _report():
    samples = [{"text": "退货运费谁出", "labels": ["运费"]},
               {"text": "买大了想退", "labels": ["尺码", "退换货"]},
               {"text": "开发票", "labels": ["发票"]}]
    preds = [["退换货"], ["退换货"], ["发票", "其他"]]
    return build_report(samples, preds, 0.8)


def test_error_kinds_and_matrix_entries():
    errors = {e["text"]: e for e in _report()["errors"]}
    assert errors["退货运费谁出"]["kind"] == "错位"
    assert errors["退货运费谁出"]["matrix_entries"] == 2
    assert errors["买大了想退"]["kind"] == "漏打"
    assert errors["开发票"]["kind"] == "多打"


def test_micro_counts_every_label_cell():
    r = _report()
    # tp=2(退换货、发票) fp=2(退换货、其他) fn=2(运费、尺码)
    assert (r["total_fp"], r["total_fn"]) == (2, 2)
    assert r["micro"] == {"p": 0.5, "r": 0.5, "f1": 0.5}
    assert r["total_cells"] == 3 * 17


def test_broad_class_has_no_red_line():
    classes = {c["name"]: c for c in _report()["classes"]}
    assert classes["其他"]["passed"] is None
    assert classes["运费"]["passed"] is False
