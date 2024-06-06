import unittest
from combinations_test import is_mirrored_duplicate, combination_to_string, get_numeric_combination, count_monomers

class TestMonomerCombinations(unittest.TestCase):
    def setUp(self):
        self.monomers = {'A': [1, 0, 2]}
        self.monomers_prime = {"A'": [2, 0, 1]}
        self.all_monomers = {**self.monomers, **self.monomers_prime}

    def test_is_mirrored_duplicate(self):
        self.assertTrue(is_mirrored_duplicate(('A',), ("A'",)))
        self.assertFalse(is_mirrored_duplicate(('A',), ('A',)))

    def test_combination_to_string_and_numeric(self):
        comb_str = combination_to_string(('A', "A'"))
        self.assertEqual(comb_str, 'A A\'')
        numeric_comb = get_numeric_combination(comb_str)
        self.assertEqual(numeric_comb, [1, 0, 2, 2, 0, 1])

    def test_count_monomers(self):
        combinations = [('A',), ("A'",), ('A', "A'")]
        counts = count_monomers(combinations, 'A')
        self.assertEqual(counts, [1, 1, 2])


if __name__ == '__main__':
    unittest.main()
