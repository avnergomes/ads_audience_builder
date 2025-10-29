import unittest

import pandas as pd

from utils import cleaning


class CleaningNormalisationTests(unittest.TestCase):
    def test_clean_dataframe_handles_boolean_headers_and_values(self):
        df = pd.DataFrame(
            {
                True: ["keep", "drop"],
                "Email": ["Valid@example.com", True],
                "Phone": ["1234567890", False],
            }
        )

        cleaned, stats, summary = cleaning.clean_dataframe(df)

        # Boolean headers should not cause failures and should be normalised.
        self.assertIn("true", cleaned.columns)
        self.assertIn("email", cleaned.columns)

        # Only the valid e-mail row should survive cleaning.
        self.assertEqual(stats["initial_rows"], 2)
        self.assertEqual(stats["invalid_emails"], 1)
        self.assertEqual(stats["final_rows"], 1)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["email"], "valid@example.com")

        # Summary should include the canonical email column.
        email_summary = next(item for item in summary if item["column"] == "email")
        self.assertEqual(email_summary["populated"], 1)

    def test_clean_dataframe_handles_nested_series_values(self):
        nested_series = pd.Series(["nested", "values"])
        df = pd.DataFrame(
            {
                "Email": [nested_series, "valid@example.com"],
                "Phone": [{"raw": "123"}, "555-0000"],
            }
        )

        cleaned, stats, summary = cleaning.clean_dataframe(df)

        # Non scalar values should be treated as missing, dropping the invalid row.
        self.assertEqual(stats["initial_rows"], 2)
        self.assertEqual(stats["invalid_emails"], 1)
        self.assertEqual(stats["final_rows"], 1)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["email"], "valid@example.com")

        # Phone column should normalise non scalar entries to empty strings.
        self.assertIn("phone", cleaned.columns)
        self.assertEqual(cleaned.iloc[0]["phone"], "555-0000")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
