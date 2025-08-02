import json
from llm_evaluator.evaluator import LLMEvaluator

if __name__ == "__main__":
    evaluator = LLMEvaluator()
    sample = {"conversation": "হ্যালো, আপনি কেমন আছেন?"}
    result = evaluator.evaluate(sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))
