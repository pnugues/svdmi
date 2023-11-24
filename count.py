"""
A word counting program
Usage: python count.py < corpus.txt
"""
__author__ = "Pierre Nugues"

import sys
import regex as re
from collections import Counter


def tokenize(text):
    words = re.findall(r'\p{L}+', text)
    return words


def count_unigrams(words):
    return Counter(words)


if __name__ == '__main__':
    text = sys.stdin.read().lower()
    words = tokenize(text)
    frequency = count_unigrams(words)
    for word in sorted(frequency.keys(), key=frequency.get, reverse=True):
        print(word, '\t', frequency[word])
