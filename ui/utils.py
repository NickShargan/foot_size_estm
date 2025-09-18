import requests


LETTER_PAPER_COUNTRIES = [
    "US",  # United States
    "CA",  # Canada
    "MX",  # Mexico
    "PH",  # Philippines
    "CL",  # Chile
    "CO",  # Colombia
    "CR",  # Costa Rica
    "GT",  # Guatemala
    "PA",  # Panama
    "DO",  # Dominican Republic
    "VE",  # Venezuela

    # US territories (follow US standards)
    "PR",  # Puerto Rico
    "GU",  # Guam
    "VI",  # U.S. Virgin Islands
    "AS",  # American Samoa
    "MP",  # Northern Mariana Islands
]


def get_country():
    try:
        resp = requests.get("https://ipinfo.io/json")
        data = resp.json()
        return data.get("country", "US")  # fallback to US
    except Exception:
        return "US"


def is_a4_format():
    """Returns True if IP/user is from country that uses A4 paper format."""

    country = get_country()

    if country in LETTER_PAPER_COUNTRIES:
        return False

    return True
