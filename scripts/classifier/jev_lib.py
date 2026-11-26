"""Jev(TypeSafe)主题分类的纯函数:建题、取分。serve 与阈值扫描共用,不碰网络。
17 类主题各问一个 Noul(是/否),P(yes) 当该类分数,保持原微调模型的多标签语义。"""
from langchain_core.decisions import ClassifierResponse, Noul

from app.core.taxonomy import TOPIC_CLASSES, TOPIC_NAMES

# Jev 的 P(yes) 与微调模型的 sigmoid 不是同一把尺子,原 threshold.json 不能沿用。
# 0.80 来自正式验证集(161 条)两次打分的扫描:0.75~0.85 差距在 Jev 调用波动内,0.80 平均最高,
# 测试集上也最好。改题面后分数会变,先跑 make classifier-threshold-scan 复核再改这里。
THRESHOLD = 0.80

# 只给 Jev 题面用的补充边界,不改 taxonomy 权威表。实测 Jev 见到「坏了/失灵」就判质量问题,
# 把只问维修保修的句子也算进去,严档 F1 掉到 0.72。
_EXTRA_BOUNDARY: dict[str, str] = {
    "质量问题": "只问能不能修、怎么修、保修多久的,归「保修维修」,不算质量问题",
}


def _question_id(index: int) -> str:
    return f"topic_{index}"


def _boundary(name: str, boundary: str) -> str:
    extra = _EXTRA_BOUNDARY.get(name)
    return f"{boundary};{extra}" if extra else boundary


def build_questions() -> dict[str, Noul]:
    """每类一题,题号按 TOPIC_NAMES 顺序编,取分时按同一顺序还原。"""
    return {
        _question_id(i): Noul(instructions=(
            f"这条电商客服的用户问题是否属于「{c.name}」类?"
            f"边界:{_boundary(c.name, c.boundary)}。用户常见说法:{'、'.join(c.examples)}。"))
        for i, c in enumerate(TOPIC_CLASSES)
    }


def scores_from(response: ClassifierResponse) -> list[float]:
    """按 TOPIC_NAMES 顺序取 17 个 P(yes)。缺题直接 KeyError:少一类就会错位,必须炸响。"""
    nouls = response.nouls
    return [nouls[_question_id(i)].noul for i in range(len(TOPIC_NAMES))]
