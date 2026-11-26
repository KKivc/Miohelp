"""Jev(TypeSafe)主题分类的纯函数:建题、取分。serve 与阈值扫描共用,不碰网络。
17 类主题各问一个 Noul(是/否),P(yes) 当该类分数,保持原微调模型的多标签语义。"""
from langchain_core.decisions import ClassifierResponse, Noul

from app.core.taxonomy import TOPIC_CLASSES, TOPIC_NAMES

# Jev 的 P(yes) 与微调模型的 sigmoid 不是同一把尺子,原 threshold.json 不能沿用。
# 0.5 是 Noul「是/否各半」的中点,作起始值;换线先跑 make classifier-threshold-scan 再改这里。
THRESHOLD = 0.5


def _question_id(index: int) -> str:
    return f"topic_{index}"


def build_questions() -> dict[str, Noul]:
    """每类一题,题号按 TOPIC_NAMES 顺序编,取分时按同一顺序还原。"""
    return {
        _question_id(i): Noul(instructions=(
            f"这条电商客服的用户问题是否属于「{c.name}」类?"
            f"边界:{c.boundary}。用户常见说法:{'、'.join(c.examples)}。"))
        for i, c in enumerate(TOPIC_CLASSES)
    }


def scores_from(response: ClassifierResponse) -> list[float]:
    """按 TOPIC_NAMES 顺序取 17 个 P(yes)。缺题直接 KeyError:少一类就会错位,必须炸响。"""
    nouls = response.nouls
    return [nouls[_question_id(i)].noul for i in range(len(TOPIC_NAMES))]
