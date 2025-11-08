import unittest

from io import StringIO

import pandas as pd

from utils import cleaning, ingest


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
        self.assertEqual(stats["missing_emails"], 0)
        self.assertEqual(stats["rows_without_contact"], 0)
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
        self.assertEqual(stats["invalid_emails"], 0)
        self.assertEqual(stats["missing_emails"], 1)
        self.assertEqual(stats["rows_without_contact"], 1)
        self.assertEqual(stats["final_rows"], 1)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["email"], "valid@example.com")

        # Phone column should normalise non scalar entries to empty strings.
        self.assertIn("phone", cleaned.columns)
        self.assertEqual(cleaned.iloc[0]["phone"], "555-0000")

    def test_clean_dataframe_autocorrects_common_email_typos(self):
        df = pd.DataFrame(
            {
                "Email": [
                    "user@gmail",
                    "person@hotmail.con",
                    "extra@yahoo.com,",
                ],
                "Phone": ["", "", ""],
            }
        )

        cleaned, stats, _ = cleaning.clean_dataframe(df)

        self.assertEqual(stats["initial_rows"], 3)
        self.assertEqual(stats["invalid_emails"], 0)
        self.assertGreaterEqual(stats["email_corrections"], 3)
        self.assertEqual(len(cleaned), 3)
        self.assertListEqual(
            cleaned["email"].tolist(),
            ["user@gmail.com", "person@hotmail.com", "extra@yahoo.com"],
        )

    def test_clean_dataframe_preserves_phone_only_rows(self):
        df = pd.DataFrame(
            {
                "Email": ["", None],
                "Phone": ["+12125550123", ""],
            }
        )

        cleaned, stats, _ = cleaning.clean_dataframe(df)

        # The phone-only row should be retained while the unreachable one is dropped.
        self.assertEqual(stats["initial_rows"], 2)
        self.assertEqual(stats["missing_emails"], 2)
        self.assertEqual(stats["rows_without_contact"], 1)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["phone"], "+12125550123")

    def test_csv_loader_respects_quoted_names(self):
        csv_text = "Created,Name,Email\n2024-01-01,\"Doe, John\",doe@example.com\n"
        df = ingest.read_audience_csv(StringIO(csv_text))

        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "Name"], "Doe, John")
        self.assertEqual(df.loc[0, "Email"], "doe@example.com")

    def test_csv_loader_detects_semicolon_delimiters(self):
        csv_text = "Email;Phone\nuser@example.com;+1 415 555 0100\n"
        df = ingest.read_audience_csv(StringIO(csv_text))

        self.assertEqual(len(df), 1)
        self.assertIn("Phone", df.columns)
        self.assertEqual(df.loc[0, "Phone"], "+1 415 555 0100")

    def test_normalize_phone_handles_numeric_values(self):
        normalised = cleaning.normalize_phone(2.012342e9)
        self.assertEqual(normalised, "+12012342000")

    def test_clean_dataframe_coalesces_duplicate_contact_columns(self):
        df = pd.DataFrame(
            {
                "Email": ["primary@example.com", ""],
                "Email 1": ["fallback@example.com", "second@example.com"],
                "Phone": ["+1 (212) 555-0100", ""],
                "Phone 2": [pd.NA, 2.125555012e9],
            }
        )

        cleaned, stats, _ = cleaning.clean_dataframe(df)

        self.assertEqual(len(cleaned), 2)
        self.assertListEqual(
            cleaned["email"].tolist(),
            ["primary@example.com", "second@example.com"],
        )
        self.assertListEqual(
            cleaned["phone"].tolist(),
            ["+12125550100", "+12125555012"],
        )
        self.assertEqual(stats["invalid_phones"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
