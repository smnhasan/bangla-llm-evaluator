# Configuration settings for the LLM evaluation framework
import os

class Config:
    DATASET_PATH = "data/bangla_conversations.jsonl"
    MODEL_API_ENDPOINT = "https://api.example.com/llm"  # Placeholder
    EVALUATION_DIMENSIONS = [
        "conversation_fluency",
        "tool_calling_performance",
        "guardrails_compliance",
        "edge_case_handling",
        "special_instruction_adherence",
        "language_proficiency",
        "task_execution_accuracy"
    ]
    OUTPUT_DIR = "evaluation_results"
    LOG_LEVEL = "INFO"
    
    # Scoring weights for each dimension (0-1 scale)
    SCORING_WEIGHTS = {
        "conversation_fluency": 0.2,
        "tool_calling_performance": 0.15,
        "guardrails_compliance": 0.25,
        "edge_case_handling": 0.15,
        "special_instruction_adherence": 0.15,
        "language_proficiency": 0.15,
        "task_execution_accuracy": 0.15
    }

# Ensure output directory exists
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
