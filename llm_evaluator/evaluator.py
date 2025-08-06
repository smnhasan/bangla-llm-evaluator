
class LLMEvaluator:
    def __init__(self):
        # You can load configs/models here later
        pass

    def evaluate(self, sample):
        """
        Evaluate a single Bangla conversation sample.
        :param sample: dict with "conversation" key containing Bangla text
        :return: dict with scores
        """
        text = sample.get("conversation", "")

        return {
            "input": text,
            "scores": f"scores: {text}",
            "overall_score": f"overall: {text}",
        }
