import ssl
import nltk


# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Now download punkt
nltk.download('punkt_tab')