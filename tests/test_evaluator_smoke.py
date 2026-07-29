import os
import unittest

from rag_pipeline.evaluator import eval_caller, get_or_create_dataset


@unittest.skipUnless(os.getenv("MISTRAL_API_KEY") or os.getenv("API_KEY"), "Mistral API key not configured")
class EvaluatorSmokeTests(unittest.TestCase):
    def test_evaluator_smoke(self) -> None:
        dataset = get_or_create_dataset()
        self.assertIsNotNone(dataset)

        results = eval_caller()
        self.assertIsNotNone(results)


if __name__ == "__main__":
    unittest.main()
