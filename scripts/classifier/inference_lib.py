"""推理纯函数:阈值应用与空标签兜底。serve 用它把 17 个 P(yes) 变成标签。
轻依赖(仅 numpy)。"""
import numpy as np


def apply_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    """sigmoid 概率 → 0/1 多标签矩阵:过线即命中;某行全不过线取最高分兜底,不产出空标签。"""
    preds = (probs >= threshold).astype(int)
    for r in range(len(preds)):
        if preds[r].sum() == 0:
            preds[r][probs[r].argmax()] = 1
    return preds
