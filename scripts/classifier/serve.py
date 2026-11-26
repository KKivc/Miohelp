"""推理服务:FastAPI 独立进程 :8110,转调 TypeSafe Jev 做 17 类多标签主题分类。
接口(/healthz、/classify)与原 ONNX 版一致,classify_pool、阈值扫描、验收页不用改。
API key 读进程环境变量 TYPESAFE_API_KEY(不走 .env),缺了启动即报错。
文本发给 TypeSafe 前先脱敏(手机号/单号/微信QQ号),与语料流水线同一套规则。
起停:make classifier-up / classifier-down(仓库惯例 nohup + pid 文件)。"""
import numpy as np
import uvicorn
from fastapi import FastAPI
from langchain_typesafe import TypeSafeClassifier
from pydantic import BaseModel

from app.core.taxonomy import TOPIC_NAMES
from scripts.classifier.corpus_lib import desensitize
from scripts.classifier.inference_lib import apply_threshold
from scripts.classifier.jev_lib import THRESHOLD, build_questions, scores_from

# 一批最多 500 条,每条一个请求;限并发免得一次性打满上游速率限制
MAX_CONCURRENCY = 8

app = FastAPI(title="miohelp topic classifier (Jev)")
_clf = TypeSafeClassifier()
_questions = build_questions()


class ClassifyIn(BaseModel):
    texts: list[str]


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/classify")
async def classify(body: ClassifyIn):
    if not body.texts:
        return {"results": []}
    responses = await _clf.abatch(
        [{"state": desensitize(t), "questions": _questions} for t in body.texts],
        config={"max_concurrency": MAX_CONCURRENCY})
    probs = np.array([scores_from(r) for r in responses])
    preds = apply_threshold(probs, THRESHOLD)   # 与 evaluate 共用同一套阈值+兜底,行为不分叉
    results = []
    for row, pred in zip(probs, preds):
        labels = [TOPIC_NAMES[i] for i in range(len(pred)) if pred[i]]
        results.append({"labels": labels,
                        "scores": {TOPIC_NAMES[i]: round(float(p), 4)
                                   for i, p in enumerate(row)}})
    return {"results": results}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8110)
