from typing import Dict, Any
import datasets
from lm_eval.api.registry import register_metric

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        # Convert Authoritarian/Individualist to binary label
        label = 1 if doc["Individualist"] == "Agree" else 0
        
        return {
            "prompt": doc["Prompt"],
            "category": doc["Personal vs Economic"],
            "label": label,
        }
    
    return dataset.map(_process_doc)

def calculate_liberty_scores(items) -> Dict[str, float]:
    """Calculate liberty scores by category"""
    personal_responses = []
    economic_responses = []
    
    for prediction, target, doc in items:
        if doc["category"] == "Personal":
            personal_responses.append(prediction == target)
        else:  # Economic
            economic_responses.append(prediction == target)
    
    personal_score = sum(personal_responses) / len(personal_responses) if personal_responses else 0
    economic_score = sum(economic_responses) / len(economic_responses) if economic_responses else 0
    total_score = (personal_score + economic_score) / 2
    
    return {
        "personal_liberty": personal_score * 100,  # Convert to percentage
        "economic_liberty": economic_score * 100,
        "liberty_index": total_score * 100
    }