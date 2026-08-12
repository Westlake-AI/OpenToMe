import unittest

from evaluations.inference.ruler import ruler_string_match
from evaluations.needle.needle_in_haystack import needle_rouge1_f1


class InferenceExperimentTest(unittest.TestCase):
    def test_needle_rouge1_fallback(self):
        self.assertAlmostEqual(needle_rouge1_f1("alpha beta", "alpha gamma"), 0.5)

    def test_ruler_string_match_requires_all_references(self):
        score = ruler_string_match(
            "The keys are alpha and gamma.", ["alpha", "beta", "gamma"]
        )
        self.assertAlmostEqual(score, 2 / 3)

    def test_ruler_string_match_accepts_one_string_reference(self):
        self.assertEqual(ruler_string_match("Answer: 42", "42"), 1.0)


if __name__ == "__main__":
    unittest.main()
