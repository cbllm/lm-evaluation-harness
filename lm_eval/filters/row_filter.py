from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter

@register_filter("row_filter")
class RowFilter(Filter):
    """
    Filters docs by a Python expression like doc['category'] == 'Personal'.
    The expression is passed in from YAML via 'condition'.
    Any doc that doesn't match is excluded from the pipeline.
    """
    def __init__(self, condition: str):
        self.condition = condition
        self._code = compile(self.condition, "<string>", "eval")

    def apply(self, resps, docs):
        """
        resps: list of model outputs/predictions
        docs: list of dataset docs
        Returns new (resps, docs) only for those rows passing `condition`.
        """
        filtered_resps = []
        filtered_docs = []
        for r, d in zip(resps, docs):
            # Evaluate the condition in a namespace where `doc` is this row's doc
            if eval(self._code, {}, {"doc": d}):
                filtered_resps.append(r)
                filtered_docs.append(d)
        return filtered_resps, filtered_docs
