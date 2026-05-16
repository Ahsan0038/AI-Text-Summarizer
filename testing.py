import unittest
from app import read_article, sentence_similarity, generate_summary


# -------------------------------------------------------------------
# 1. Tests for read_article()
# -------------------------------------------------------------------
class TestReadArticle(unittest.TestCase):

    def test_basic_split(self):
        result = read_article("Hello world this is test")
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], list)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            read_article("")


# -------------------------------------------------------------------
# 2. Tests for sentence_similarity()
# -------------------------------------------------------------------
class TestSentenceSimilarity(unittest.TestCase):

    def test_identical_sentences(self):
        sent = ["the", "cat", "sat"]
        score = sentence_similarity(sent, sent)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_different_sentences(self):
        sent1 = ["apple", "orange"]
        sent2 = ["dog", "cat"]
        score = sentence_similarity(sent1, sent2)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


# -------------------------------------------------------------------
# 3. Tests for generate_summary()
# -------------------------------------------------------------------
class TestGenerateSummary(unittest.TestCase):

    def test_output_is_string(self):
        text = "AI is good. AI is powerful. AI is future."
        result = generate_summary(text, top_n=1)
        self.assertIsInstance(result, str)

    def test_non_empty_summary(self):
        text = "AI is good. AI is powerful. AI is future."
        result = generate_summary(text, top_n=1)
        self.assertTrue(len(result) > 0)


# -------------------------------------------------------------------
# run tests
# -------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()