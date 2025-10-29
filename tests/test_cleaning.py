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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
