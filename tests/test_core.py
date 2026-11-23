import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.ingest import clean_text, download, split_text
from src.rag import Passage, cited_passages
from src.risk import DiabetesRiskModel, HeartRiskModel


class CoreTests(unittest.TestCase):
    def test_clean_text_normalizes_whitespace(self):
        self.assertEqual(clean_text("a\u00ad  b\n\n\n\nc"), "a b\n\nc")

    def test_chunks_keep_source_metadata(self):
        source = {
            "id": "DOC",
            "title": "Title",
            "publisher": "Publisher",
            "url": "https://example.org",
        }
        chunks = split_text("First paragraph.\nSecond paragraph.", source, size=20, overlap=4)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertTrue(all(chunk.source_id == "DOC" for chunk in chunks))

    def test_download_reuses_cached_document(self):
        source = {
            "id": "DOC",
            "title": "Title",
            "publisher": "Publisher",
            "url": "https://invalid.example/should-not-be-requested",
        }
        html = "<html><body><main><p>" + ("cached healthcare text " * 20) + "</p></main></body></html>"
        with TemporaryDirectory() as directory:
            Path(directory, "DOC.html").write_text(html, encoding="utf-8")
            self.assertIn("cached healthcare text", download(source, Path(directory)))

    def test_only_cited_passages_are_shown(self):
        base = dict(score=0.5, chunk_id="c", source_id="d", title="t", publisher="p", url="u", text="x")
        passages = [Passage(label="S1", **base), Passage(label="S2", **base)]
        self.assertEqual([p.label for p in cited_passages("Claim [S2].", passages)], ["S2"])

    def test_risk_model_scores_symptom_profile_above_empty_profile(self):
        model = DiabetesRiskModel("data/ml")
        fields = [
            "polyuria", "polydipsia", "sudden_weight_loss", "weakness", "polyphagia",
            "genital_thrush", "visual_blurring", "itching", "irritability", "delayed_healing",
            "partial_paresis", "muscle_stiffness", "alopecia", "obesity",
        ]
        low = {"age": 40, "gender": "Male", **{field: "No" for field in fields}}
        high = {"age": 40, "gender": "Male", **{field: "Yes" for field in fields}}
        self.assertGreater(model.predict(high).model_score, model.predict(low).model_score)

    def test_heart_model_returns_bounded_score(self):
        patient = {
            "age": 55, "sex": 1, "cp": 4, "trestbps": 130, "chol": 240,
            "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0,
            "slope": 2, "ca": 0, "thal": 3,
        }
        score = HeartRiskModel("data/ml").predict(patient).model_score
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)


if __name__ == "__main__":
    unittest.main()
