import importlib


def test_evaluator_module_exposes_expected_functions() -> None:
    evaluator = importlib.import_module("rag_pipeline.evaluator")

    assert hasattr(evaluator, "get_or_create_dataset")
    assert hasattr(evaluator, "target")
    assert hasattr(evaluator, "run_judge")
    assert hasattr(evaluator, "faithfulness_evaluator")
    assert hasattr(evaluator, "correctness_evaluator")
    assert hasattr(evaluator, "context_relevance_evaluator")
    assert hasattr(evaluator, "context_recall_evaluator")
    assert hasattr(evaluator, "eval_caller")
