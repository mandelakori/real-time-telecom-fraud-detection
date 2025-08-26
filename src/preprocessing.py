#!usr/bin/python3
"""
Preprocessing file to clean data.
"""

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()


if __name__ == '__main__':
    clean_text()