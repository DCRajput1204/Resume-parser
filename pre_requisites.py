import sys
import subprocess
import nltk

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


print("Downloading SpaCy 2.x Model...")
try:
    install('https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz')
    print("SpaCy Model installed successfully.")
except Exception as e:
    print(f"Error installing SpaCy model: {e}")


print("Downloading NLTK data...")
nltk_packages = [
    'maxent_ne_chunker',
    'words',
    'stopwords',
    'punkt',
    'wordnet',
    'averaged_perceptron_tagger'
]

for package in nltk_packages:
    try:
        nltk.download(package, quiet=True)
        print(f"Downloaded: {package}")
    except Exception as e:
        print(f"Failed to download {package}: {e}")

print("Pre-requisites setup complete.")