import count
import count_bigrams
import count_ngrams
import os


class Corpus:
    def __init__(self, text, context_size=3):
        self.text = text.lower()
        self.context_size = context_size
        self.words = None
        self.text_size = None
        self.unique_words = None
        self.vocab_size = None
        self.unigrams = None
        self.bigrams = None
        self.ngrams = None

    def stats(self):
        self.words = count.tokenize(self.text)
        self.text_size = len(self.words)
        self.unique_words = sorted(list(set(self.words)))
        self.vocab_size = len(self.unique_words)
        self.unigrams = count.count_unigrams(self.words)
        self.bigrams = count_bigrams.count_bigrams(self.words)
        self.ngrams = count_ngrams.count_ngrams(self.words, self.context_size)

    def index_vocabulary(self):
        self.word2idx = {word: i for (i, word) in enumerate(self.unique_words)}
        self.idx2word = {v: k for k, v in self.word2idx.items()}


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


if __name__ == '__main__':
    dickens = True
    context_size = 3
    BASE = '../../../corpus/'
    if dickens:
        path = BASE + 'Dickens/'
        test_words = ['he', 'she', 'paris', 'table', 'rare', 'monday', 'sunday']
    else:
        path = BASE + 'Selma/'
        test_words = ['han', 'hon', 'att', 'bord', 'bordet', 'måndag', 'söndag']

    files = get_files(path, 'txt')
    files = [path + file for file in files]
    print(files)
    text = ''
    for file in files:
        text += open(file).read()
    # text = open('../../../corpus/Selma/gosta.txt').read()

    # text = """Earlier this month, Rep. Charlie Dent of Pennsylvania became the first House Republican to formally back legislation to protect special counsel Robert Mueller's job, co-sponsoring a bipartisan bill.
    # Less self.contextsthan a week later, he announced he was beginning his retirement from Congress months earlier than expected: Instead of leaving at the end of the session, he'd head for the exit in May.
    # The move may have seemed sudden but the seven-term congressman, who had initially announced plans to retire last September, has spent much of the past year eyeing the door — and speaking his mind.
    # month this earlier  we
    # """
    # text = open('alice.txt').read()
    corpus = Corpus(text, context_size)
    corpus.stats()
    corpus.index_vocabulary()
    print(corpus.word2idx)
    print(corpus.vocab_size)
    most_freq_ngrams = sorted(corpus.ngrams, key=corpus.ngrams.get, reverse=True)[:10]
    print([(ngram, corpus.ngrams[ngram]) for ngram in most_freq_ngrams])
