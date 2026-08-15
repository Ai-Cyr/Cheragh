from contextlib import redirect_stdout
from io import StringIO
import unittest

from cheragh.catalog import TechniqueStatus, get_technique, list_techniques
from cheragh.cli.main import main


class V110TechniqueCatalogTests(unittest.TestCase):
    def test_catalog_distinguishes_stable_and_experimental_techniques(self):
        self.assertTrue(get_technique("self-rag").available)
        self.assertEqual(get_technique("community-graphrag").status, TechniqueStatus.EXPERIMENTAL)
        self.assertTrue(get_technique("community-graphrag").available)

    def test_catalog_filters_are_machine_readable(self):
        planned = list_techniques(status="planned")
        self.assertEqual(planned, [])
        multimodal = list_techniques(family="multimodal", available=True)
        self.assertEqual([item.id for item in multimodal], ["multimodal-rag", "colpali"])

    def test_unknown_technique_is_rejected(self):
        with self.assertRaises(KeyError):
            get_technique("not-a-technique")

    def test_cli_lists_catalog_as_json(self):
        output = StringIO()
        with redirect_stdout(output):
            code = main(["techniques", "list", "--family", "multimodal", "--json"])
        self.assertEqual(code, 0)
        self.assertIn('"id": "multimodal-rag"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
