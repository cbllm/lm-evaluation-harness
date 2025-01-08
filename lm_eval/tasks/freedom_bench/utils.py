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



def get_score_by_category(items, target_category):
    """Helper function that calculates fraction of correct answers within a category."""
    correct = 0
    total = 0
    for prediction, target, doc in items:
        if doc["category"] == target_category:
            total += 1
            if prediction == target:
                correct += 1
    return correct / total if total else 0

@register_metric(
    metric="personal_liberty_score",
    higher_is_better=True,
    output_type="multiple_choice",
)
def compute_personal_liberty(items):
    return {"personal_liberty_score": get_score_by_category(items, "Personal")}

@register_metric(
    metric="economic_liberty_score",
    higher_is_better=True,
    output_type="multiple_choice",
)
def compute_economic_liberty(items):
    return {"economic_liberty_score": get_score_by_category(items, "Economic")}

@register_metric(
    metric="liberty_index",
    higher_is_better=True,
    output_type="multiple_choice",
)
def compute_liberty_index(items):
    personal = get_score_by_category(items, "Personal")
    economic = get_score_by_category(items, "Economic")
    return {"liberty_index": (personal + economic) / 2}


