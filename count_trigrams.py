"""
Trigram counting
Usage: python count_trigrams.py < corpus.txt
"""
__author__ = "Pierre Nugues"

import sys

import regex as re
from collections import Counter


def tokenize(text):
    words = re.findall(r'\p{L}+', text)
    return words


def count_trigrams(words):
    trigrams = [tuple(words[idx:idx + 3])
                for idx in range(len(words) - 2)]
    return Counter(trigrams)


if __name__ == '__main__':
    text = sys.stdin.read().lower()
    words = tokenize(text)
    frequency_trigrams = count_trigrams(words)
    for trigram in frequency_trigrams:
        print(frequency_trigrams[trigram], "\t", trigram)
