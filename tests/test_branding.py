import unittest

import branding


class BrandingTest(unittest.TestCase):
    def test_names(self):
        self.assertEqual(branding.TAGLINE, "Tagos Transcribe by Tagos Labs")

    def test_placeholder_helper(self):
        self.assertTrue(branding.is_placeholder("{{SITE}}"))

    def test_links_filled(self):
        self.assertEqual(len(branding.LINKS), 3)
        for _, url in branding.LINKS:
            self.assertFalse(branding.is_placeholder(url))
            self.assertTrue(url.startswith("https://"))


if __name__ == "__main__":
    unittest.main()
