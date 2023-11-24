"""
Bigram counting
Usage: python count_bigrams.py < corpus.txt
"""
__author__ = "Pierre Nugues"

import sys
import regex as re
from collections import Counter


def tokenize(text):
    words = re.findall(r'\p{L}+', text)
    return words


def count_bigrams(words):
    bigrams = [tuple(words[idx:idx + 2])
               for idx in range(len(words) - 1)]
    return Counter(bigrams)


if __name__ == '__main__':
    text = sys.stdin.read().lower()
    words = tokenize(text)
    frequency_bigrams = count_bigrams(words)
    for bigram in frequency_bigrams:
        print(frequency_bigrams[bigram], "\t", bigram)
