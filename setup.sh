#!/bin/bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')" 