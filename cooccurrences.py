"""
Cooccurrence matrix
Pierre Nugues
"""
import time
import os
import corpus as cp
import regex as re


class Cooccurrences:
    def __init__(self, corpus, context_size=3, unordered_pairs=True):
        self.corpus = corpus
        self.unigrams = corpus.unigrams
        self.word_context = dict()
        self.contexts = dict()
        self.context_size = context_size
        self.unordered_pairs = unordered_pairs

    def cooccurrences(self):
        """
        Cooccurrence (word, context), where the context is a bag of word
        to the left and to the right
        :return:
        """
        # The word index
        mid_idx = self.context_size // 2
        # we count all the (word x context)
        for ngram in self.corpus.ngrams:
            word = ngram[mid_idx]
            context = sorted(ngram[:mid_idx] + ngram[mid_idx + 1:])
            context = tuple(context)
            if context in self.contexts:
                self.contexts[context] += self.corpus.ngrams[ngram]
            else:
                self.contexts[context] = self.corpus.ngrams[ngram]
            if (word, context) in self.word_context:
                self.word_context[(word, context)] += self.corpus.ngrams[ngram]
            else:
                self.word_context[(word, context)] = self.corpus.ngrams[ngram]
        if self.unordered_pairs:
            self.word_context, self.contexts = self.word_pairs_from_context()

    def word_pairs_from_context(self):
        """
        Generates (word, word) pairs from the (word, context) pairs
        for example with w1, w2, w3, from (w2, (w1, w3)) to (w2, w1) and (w2, w3)
        with w1, w2, w3, w4, w5 from (w3, (w1, w2, w4, w5)) to (w3, w1), (w3, w2), (w3, w4), (w3, w5)
        :return:
        """
        word_pairs = {}
        for (word, context) in self.word_context:
            for context_word in context:
                if (word, context_word) in word_pairs:
                    word_pairs[(word, context_word)] += self.word_context[(word, context)]
                else:
                    word_pairs[(word, context_word)] = self.word_context[(word, context)]
        # We redefine the contexts
        contexts = dict()
        for (word1, word2) in word_pairs.keys():
            if word2 in contexts:
                contexts[word2] += word_pairs[(word1, word2)]
            else:
                contexts[word2] = word_pairs[(word1, word2)]
        return word_pairs, contexts

    def index_contexts(self):
        """
        Creates an index of the contexts
        :param contexts:
        :return:
        """
        sorted_contexts = sorted(self.contexts)
        self.context2idx = {context: i for (i, context) in enumerate(sorted_contexts)}
        self.idx2context = {v: k for k, v in self.context2idx.items()}


def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        if file.endswith(suffix):
            files.append(file)
    return files


def symmetric_contexts():
    """
    Function to extract symmetric contexts. Used for debugging
    :return:
    """
    text = open('alice.txt').read().lower()

    # Extract contexts of three words, where there is a symmetry in the corpus
    # given that there is word1 focus_word word2, we have also word2 focus_word word1
    for match in re.finditer('\\b(\w+) (\w+) (\w+)\\b(?= .+ \\b\\3 \\2 \\1\\b)', text, re.S):
        print(match.group(1), match.group(2), match.group(3))


def unit_tests():
    """
    To reimplement
    :return:
    """
    text = open('alice.txt').read()
    corpus = cp.Corpus(text)
    corpus.stats()
    corpus.index_vocabulary()

    cooc = Cooccurrences(corpus, unordered_pairs=False)
    cooc.cooccurrences()
    cooc.index_contexts()

    assert cooc.corpus.text_size == 27333
    assert corpus.ngrams[('i', 'wish', 'you')] == 5
    c = [idx for idx, word in enumerate(corpus.words) if corpus.words[idx] == 'i' and
         corpus.words[idx + 1] == 'wish' and corpus.words[idx + 2] == 'you']
    print(c)

    assert corpus.ngrams[('it', 'say', 'to')] == 1
    assert corpus.ngrams[('to', 'say', 'it')] == 3
    assert cooc.word_context[('say', ('it', 'to'))] == 4
    assert cooc.contexts[('it', 'to')] == 61
    assert cooc.corpus.unigrams['say'] == 51
    c = [idx for idx, word in enumerate(corpus.words) if corpus.words[idx] == 'it' and
         corpus.words[idx + 1] == 'say' and corpus.words[idx + 2] == 'to']
    d = [idx for idx, word in enumerate(corpus.words) if corpus.words[idx] == 'to' and
         corpus.words[idx + 1] == 'say' and corpus.words[idx + 2] == 'it']
    e = [idx for idx, word in enumerate(corpus.words) if corpus.words[idx] == 'say']
    f = [idx for idx, word in enumerate(corpus.words) if corpus.words[idx] == 'it'
         and corpus.words[idx + 2] == 'to']
    g = [idx for idx, word in enumerate(corpus.words) if corpus.words[idx] == 'to'
         and corpus.words[idx + 2] == 'it']
    print(len(c) + len(d), len(e), len(f) + len(g))

    walked = [(k, v) for k, v in cooc.word_context.items() if k[0] == 'walked']
    off_they = [(k, v) for k, v in cooc.word_context.items() if k[1] == ('off', 'they')]
    print('Word: walked', walked)
    print('Context: off___they', off_they)

    assert corpus.ngrams[('i', 'think', 'you')] == 4
    assert corpus.ngrams[('you', 'think', 'i')] == 1
    assert cooc.word_context[('think', ('i', 'you'))] == 5
    assert cooc.contexts[('i', 'you')] == 28
    assert cooc.corpus.unigrams['think'] == 53
    c = [idx for idx, word in enumerate(corpus.words) if corpus.words[idx] == 'i' and
         corpus.words[idx + 1] == 'think' and corpus.words[idx + 2] == 'you']
    d = [idx for idx, word in enumerate(corpus.words) if corpus.words[idx] == 'you' and
         corpus.words[idx + 1] == 'think' and corpus.words[idx + 2] == 'i']
    e = [idx for idx, word in enumerate(corpus.words) if corpus.words[idx] == 'think']
    f = [idx for idx, word in enumerate(corpus.words) if corpus.words[idx] == 'i'
         and corpus.words[idx + 2] == 'you']
    g = [idx for idx, word in enumerate(corpus.words) if corpus.words[idx] == 'you'
         and corpus.words[idx + 2] == 'i']
    print(len(c) + len(d), len(e), len(f) + len(g))

    assert corpus.ngrams[('more', 'and', 'more')] == 2
    assert cooc.word_context[('and', ('more', 'more'))] == 2
    assert cooc.contexts[('more', 'more')] == 2
    assert cooc.corpus.unigrams['and'] == 872
    exit()


if __name__ == '__main__':
    # unit_tests()
    dickens = True
    UNORDERED_PAIRS = True
    CONTEXT_SIZE = 5
    BASE = '../../../corpus/'
    if dickens:
        path = BASE + 'Dickens/'
        test_words = ['he', 'she', 'paris', 'table', 'rare', 'monday', 'sunday', 'man', 'woman', 'king', 'queen', 'boy',
                      'girl']
    else:
        path = BASE + 'Selma/'
        test_words = ['han', 'hon', 'att', 'bord', 'bordet', 'måndag', 'söndag', 'man', 'kvinna', 'kung', 'drottning',
                      'pojke', 'flicka']

    t1 = time.time()
    files = get_files(path, 'txt')
    files = [path + file for file in files]
    print(files)
    text = ''
    for file in files:
        text += open(file).read()

    # if dickens:
    #    text += open('../../../corpus/big.txt').read()

    # text = open('../../../corpus/Selma/gosta.txt').read()

    text_string = """Earlier this month, Rep. Charlie Dent of Pennsylvania became the first House Republican to formally back legislation to protect special counsel Robert Mueller's job, co-sponsoring a bipartisan bill.
    Less than a week later, he announced he was beginning his retirement from Congress months earlier than expected: Instead of leaving at the end of the session, he'd head for the exit in May.
    The move may have seemed sudden but the seven-term congressman, who had initially announced plans to retire last September, has spent much of the past year eyeing the door — and speaking his mind.
    month this earlier  we month this earlier do
    """
    text = text_string
    # text = open('alice.txt').read()

    corpus = cp.Corpus(text, context_size=CONTEXT_SIZE)
    corpus.stats()
    corpus.index_vocabulary()

    cooc = Cooccurrences(corpus, context_size=CONTEXT_SIZE,
                         unordered_pairs=UNORDERED_PAIRS)
    t2 = time.time()
    cooc.cooccurrences()
    cooc.index_contexts()
    t3 = time.time()

    print(cooc.word_context)
    print(cooc.contexts)
    """
    # Debugging code
    symmetric_contexts()
    cnt = 0
    for ngram in corpus.ngrams:
        if ngram[1] == 'say':
            cnt += corpus.ngrams[ngram]
            print(ngram, corpus.ngrams[ngram])
    print(cnt)

    cnt = 0
    for word_context in cooc.word_context:
        if word_context[0] == 'say':
            print(word_context, cooc.word_context[word_context])
            cnt += cooc.word_context[word_context]
    print(cnt)"""
