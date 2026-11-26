"""数据集:分层划分 80/10/10。验证集定阈值,测试集只在评测用一次。
运行:make classifier-dataset(只读 corpus_labeled.jsonl,不调模型)。"""
import json
import pathlib

from app.core.taxonomy import TOPIC_NAMES
from scripts.classifier.corpus_lib import split_dataset

SRC = pathlib.Path("data/classifier/corpus_labeled.jsonl")
OUT = pathlib.Path("data/classifier/dataset")


def _dump(path: pathlib.Path, samples: list[dict]) -> None:
    path.write_text("\n".join(
        json.dumps({"text": s["text"], "labels": s["labels"]}, ensure_ascii=False)
        for s in samples), encoding="utf-8")


def _dist(name: str, samples: list[dict]) -> None:
    counts = {n: 0 for n in TOPIC_NAMES}
    for s in samples:
        for lb in s["labels"]:
            counts[lb] += 1
    print(f"{name}({len(samples)} 条): " + " ".join(f"{k}={v}" for k, v in counts.items()))


def main() -> None:
    samples = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, ds in zip(("train", "val", "test"), split_dataset(samples)):
        _dump(OUT / f"{name}.jsonl", ds)
        _dist(name, ds)


if __name__ == "__main__":
    main()
