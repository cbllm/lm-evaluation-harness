from typing import Dict, Any
import datasets
from lm_eval.api.registry import register_metric

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        # Convert Authoritarian/Individualist to binary label
        # Where 'Agree' for Authoritarian stance = 0, 'Agree' for Individualist stance = 1
        label = 1 if doc["Individualist"] == "Agree" else 0
        
        return {
            "prompt": doc["Prompt"],
            "category": doc["Personal vs Economic"],
            "label": label,
        }
    
    return dataset.map(_process_doc)

@register_metric(
    metric="freedom_score",
    higher_is_better=True,
)
def calculate_freedom_score(samples) -> Dict[str, float]:
    """Calculate freedom scores by category"""
    personal_responses = []
    economic_responses = []
    
    for pred, doc in samples:
        if doc["category"] == "Personal":
            personal_responses.append(pred == doc["label"])
        else:  # Economic
            economic_responses.append(pred == doc["label"])
            
    personal_score = sum(personal_responses) / len(personal_responses) if personal_responses else 0
    economic_score = sum(economic_responses) / len(economic_responses) if economic_responses else 0
    total_score = (personal_score + economic_score) / 2
    
    return {
        "personal_freedom": personal_score * 100,  # Convert to percentage
        "economic_freedom": economic_score * 100,
        "total_freedom": total_score * 100,
    }