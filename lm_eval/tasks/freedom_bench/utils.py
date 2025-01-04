from typing import Dict, Any
import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    print("Dataset info:", dataset)  # Debug line - datasets don't have keys() method
    
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

def calculate_freedom_score(items) -> Dict[str, float]:
    """Calculate freedom scores by category"""
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
        "personal_freedom": personal_score * 100,  # Convert to percentage
        "economic_freedom": economic_score * 100,
        "total_freedom": total_score * 100,
    }