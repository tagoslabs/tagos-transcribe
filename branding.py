"""Product name and public links, kept in one place."""

APP_NAME = "Tagos Transcribe"
ORG_NAME = "Tagos Labs"
TAGLINE = f"{APP_NAME} by {ORG_NAME}"

SITE_URL = "https://tagoslabs.com"
REDDIT_URL = "https://www.reddit.com/user/tagoslabs"
BMC_URL = "https://buymeacoffee.com/tagoslabs"

LINKS = (
    ("Website", SITE_URL),
    ("Reddit", REDDIT_URL),
    ("Buy Me a Coffee", BMC_URL),
)


def is_placeholder(url: str) -> bool:
    """True for an unfilled {{...}} value; such links render as plain text."""
    return url.startswith("{{") and url.endswith("}}")
