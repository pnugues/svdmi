"""
Counting n-grams of any size: N
Usage: python count_ngrams.py N < corpus.txt
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


def count_ngrams(words, n):
    ngrams = [tuple(words[idx:idx + n])
              for idx in range(len(words) - n + 1)]
    # "\t".join(words[idx:idx + n])
    return Counter(ngrams)


if __name__ == '__main__':
    n = 2
    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    text = sys.stdin.read().lower()
    words = tokenize(text)
    # frequency = count_bigrams(words)
    frequency = count_ngrams(words, n)
    for word in sorted(frequency, key=frequency.get, reverse=True):
        print(word, '\t', frequency[word])
