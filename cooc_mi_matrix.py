"""
Computes mutual information from cooccurrences and applies a SVD
Pierre Nugues
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import fbpca
import time
import math
from sklearn.preprocessing import StandardScaler, Normalizer
import os
import corpus as cp
import cooccurrences as cooccur
import regex as re
from scipy.sparse import lil_matrix
from tqdm import tqdm


class MutualInfo:
    def __init__(self, coocs, power=0.75,
                 unordered_pairs=True, cutoff_c=1, cutoff_w=1):
        self.unigrams = coocs.unigrams
        self.contexts = coocs.contexts
        self.word_context = coocs.word_context

        self.unordered_pairs = unordered_pairs
        self.cutoff_c = cutoff_c
        self.cutoff_w = cutoff_w
        self.power = power
        self.word_context_pruned = dict()

    def build_matrix(self):
        # self.COOC = np.zeros((self.corpus.vocab_size, len(self.contexts_cutoff)))
        self.COOC = lil_matrix((len(self.word2idx_m), len(self.context2idx_m)))
        for (word, context) in tqdm(self.mutual_info):
            i = self.word2idx_m[word]
            j = self.context2idx_m[context]
            self.COOC[i, j] = self.mutual_info[(word, context)]
        self.COOC = self.COOC.tocsc()
        # self.COOC = self.COOC.tocsc()
        # print("Creating sparse matrix")
        # self.COOC = csc_matrix(self.COOC)

    def mutual_info(self):
        """
        Computes the mutual information between a word and a context
        :return:
        """
        # We prune the dataset
        self.word_context_pruned = {(w, c): v
                                    for (w, c), v in self.word_context.items()
                                    if self.unigrams[w] >= self.cutoff_w and
                                    self.contexts[c] >= self.cutoff_c}
        # get all the words used in the matrix
        words_m = sorted(set([w for w, c in self.word_context_pruned.keys()]))
        contexts_m = sorted(set([c for w, c in self.word_context_pruned.keys()]))
        # Build indices
        self.idx2word_m = {i: w for i, w in enumerate(words_m)}
        self.word2idx_m = {w: i for i, w in enumerate(words_m)}
        self.idx2context_m = {i: c for i, c in enumerate(contexts_m)}
        self.context2idx_m = {c: i for i, c in enumerate(contexts_m)}

        # P(w, C) = #(w, c)^a/(∑_i #(w_i, C_i)^a)
        self.mutual_info = {k: np.power(v, self.power)
                            for k, v in self.word_context_pruned.items()}
        mi_divisor = sum(self.mutual_info.values())
        self.mutual_info = {k: v / mi_divisor for k, v in self.mutual_info.items()}

        # P(w) = #w^a/∑_i #(w_i)^a
        p_reweighted_unigrams = {k: np.power(v, self.power) for k, v in self.unigrams.items()}
        unigram_divisor = sum(self.unigrams.values())
        p_reweighted_unigrams = {k: v / unigram_divisor for k, v in p_reweighted_unigrams.items()}

        # P(C) = #C^a/∑_i #(C_i)^a
        p_reweighted_contexts = {k: np.power(v, self.power) for k, v in self.contexts.items()}
        context_divisor = sum(p_reweighted_contexts.values())
        p_reweighted_contexts = {k: v / context_divisor for k, v in p_reweighted_contexts.items()}

        for word, context in self.mutual_info:
            self.mutual_info[(word, context)] *= 1 / (
                    p_reweighted_unigrams[word] * p_reweighted_contexts[context])
            self.mutual_info[(word, context)] = math.log(self.mutual_info[(word, context)], 2)
            if self.mutual_info[(word, context)] < 0:
                self.mutual_info[(word, context)] = 0
        return self.mutual_info

    def closest_words(self, word, U, nbr_words=10):
        # Here cosine distance and not cosine
        # distance between equal vectors: 0. max distance: 2
        vector = U[self.word2idx_m[word], :]
        return self.closest_words_to_vector(vector, U, nbr_words)

    def closest_words_to_vector(self, vector, U, nbr_words=10):
        # Here cosine distance and not cosine
        # distance between equal vectors: 0. max distance: 2
        dist = [cosine(vector, U[i, :]) if np.any(U[i, :]) else 2
                for i in range(U.shape[0])]
        sort_dist = sorted(range(len(dist)), key=lambda k: dist[k])
        return [self.idx2word_m[x] for x in sort_dist[:nbr_words]]


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


def load_corpus(path):
    files = get_files(path, 'txt')
    files = [path + file for file in files]
    print(files)
    text = ''
    for file in files:
        text += open(file).read()
    return text


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

    cooc = cooccur.Cooccurrences(corpus, unordered_pairs=False)
    cooc.cooccurrences()
    cooc.index_contexts()

    mi = MutualInfo(cooc, power=1.0, cutoff_c=1, cutoff_w=1)
    mi.mutual_info()
    print('Building matrix...')
    mi.build_matrix()

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
    assert mi.mutual_info[('say', ('it', 'to'))] == 5.1351935150753905
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
    assert mi.mutual_info[('think', ('i', 'you'))] == 6.525008912876331
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
    assert mi.mutual_info[('and', ('more', 'more'))] == 4.970171869832846
    exit()


if __name__ == '__main__':
    # unit_tests()
    # Parameters:
    # PPMI (positive PMI)
    # U = U @ np.diag(s) (eigen)
    # normalization
    # centering (raw and standardize)
    # vector size
    VECTOR_SIZE = 300
    STANDARDIZE = False
    RAW = True  # RAW = False is the same as STANDARDIZE = True
    EIGEN_POWER = 1.0  # 0 (no eigen values), 0.5, or 1 (True SVD)
    POWER = 1.0  # Power to smooth the means
    UNORDERED_PAIRS = False
    CONTEXT_SIZE = 3
    CUTOFF_C = 5  # Minimal number of contexts, for dickens 5
    CUTOFF_W = 3  # Minimal number of words, for dickens 3
    dataset = 'dickens'  # 'dickens' 'selma' 'alice' 'short'
    if dataset == 'dickens':
        path = '../../../corpus/Dickens/'
        text = load_corpus(path)
        test_words = ['he', 'she', 'paris', 'table', 'rare', 'monday', 'sunday', 'man', 'woman', 'king', 'queen', 'boy',
                      'girl']
    elif dataset == 'selma':
        path = '../../../corpus/Selma/'
        text = load_corpus(path)
        test_words = ['han', 'hon', 'att', 'bord', 'bordet', 'måndag', 'söndag', 'man', 'kvinna', 'kung', 'drottning',
                      'pojke', 'flicka']
    elif dataset == 'alice':
        text = open('alice.txt').read()
        test_words = ['the', 'he', 'rabbit', 'alice']
    elif dataset == 'short':
        text = """Earlier this month, Rep. Charlie Dent of Pennsylvania became the first House Republican to formally back legislation to protect special counsel Robert Mueller's job, co-sponsoring a bipartisan bill.
        Less than a week later, he announced he was beginning his retirement from Congress months earlier than expected: Instead of leaving at the end of the session, he'd head for the exit in May.
        The move may have seemed sudden but the seven-term congressman, who had initially announced plans to retire last September, has spent much of the past year eyeing the door — and speaking his mind.
        month this earlier  we month this earlier do
        """
        VECTOR_SIZE = 50
        test_words = ['the', 'he']

    np.random.seed(0)
    t1 = time.time()
    corpus = cp.Corpus(text, context_size=CONTEXT_SIZE)
    corpus.stats()
    corpus.index_vocabulary()

    print('Computing cooccurrences...')
    coocs = cooccur.Cooccurrences(corpus, context_size=CONTEXT_SIZE,
                                  unordered_pairs=UNORDERED_PAIRS)
    t2 = time.time()
    coocs.cooccurrences()
    t3 = time.time()
    mi = MutualInfo(coocs, power=POWER, cutoff_c=CUTOFF_C, cutoff_w=CUTOFF_W)
    print('Computing mutual info...')
    mi.mutual_info()
    t4 = time.time()
    print('Building matrix...')
    mi.build_matrix()
    t5 = time.time()
    print('Time to compute MI:', t5 - t2)
    # exit()

    # The next instruction is to check that we obtain similar vectors
    # cooc.context_to_word()
    # print(cooc.word_context)
    # print(cooc.word_word)

    COOC = mi.COOC
    # print(COOC)
    print(COOC.shape)

    print('Normalizing the matrix...')
    COOC = Normalizer().fit_transform(COOC)
    if STANDARDIZE:
        print('Centering the matrix...')
        COOC = StandardScaler(with_mean=False).fit_transform(COOC)
    t6 = time.time()
    print('Time to normalize/standardize the matrix:', t6 - t5)

    print('Computing SVD...')
    U, s, V = fbpca.pca(COOC, raw=RAW, k=VECTOR_SIZE)
    print(np.shape(U))
    t7 = time.time()
    print('Time to compute SVD:', t7 - t6)

    # We compute the power of the eigen matrix
    s = np.power(s, EIGEN_POWER)
    Us = U @ np.diag(s)

    # We normalize Us
    Us = Normalizer().fit_transform(Us)
    # We list words close to the test words
    closest_to_testwords = {}
    for w in test_words:
        if (w in corpus.word2idx and
                w in mi.word2idx_m and
                np.any(Us[mi.word2idx_m[w], :])):
            closest_to_testwords[w] = mi.closest_words(w, Us)
            print(w, closest_to_testwords[w])

    if dataset == 'dickens':
        print('distance 10 closest words sunday',
              cosine(Us[mi.word2idx_m['sunday'], :], Us[mi.word2idx_m[closest_to_testwords['sunday'][1]], :]), '...',
              cosine(Us[mi.word2idx_m['sunday'], :], Us[mi.word2idx_m[closest_to_testwords['sunday'][9]], :]))
        print('distance 10 closest words monday',
              cosine(Us[mi.word2idx_m['monday'], :], Us[mi.word2idx_m[closest_to_testwords['monday'][1]], :]), '...',
              cosine(Us[mi.word2idx_m['monday'], :], Us[mi.word2idx_m[closest_to_testwords['monday'][9]], :]))

        print('distance sunday-saturday', cosine(Us[mi.word2idx_m['sunday'], :], Us[mi.word2idx_m['saturday'], :]))
        print('distance monday-sunday', cosine(Us[mi.word2idx_m['monday'], :], Us[mi.word2idx_m['sunday'], :]))
        print('distance monday-saturday', cosine(Us[mi.word2idx_m['monday'], :], Us[mi.word2idx_m['saturday'], :]))

        print('Semantic operations:')
        vec_1 = (Us[mi.word2idx_m['king']] - Us[mi.word2idx_m['man']] + Us[mi.word2idx_m['woman']])
        # print(vec_1)
        print('king - man + woman:', mi.closest_words_to_vector(vec_1, Us))
        vec_2 = (Us[mi.word2idx_m['queen']] - Us[mi.word2idx_m['woman']] + Us[mi.word2idx_m['man']])
        print('queen - woman + man:', mi.closest_words_to_vector(vec_2, Us))
        vec_3 = (Us[mi.word2idx_m['he']] - Us[mi.word2idx_m['man']] + Us[mi.word2idx_m['woman']])
        print('he - man + woman:', mi.closest_words_to_vector(vec_3, Us))
        vec_4 = (Us[mi.word2idx_m['she']] - Us[mi.word2idx_m['woman']] + Us[mi.word2idx_m['man']])
        print('she - woman + man:', mi.closest_words_to_vector(vec_4, Us))
        vec_5 = (Us[mi.word2idx_m['boy']] - Us[mi.word2idx_m['man']] + Us[mi.word2idx_m['woman']])
        print('boy - man + woman:', mi.closest_words_to_vector(vec_5, Us))

    # exit()
    for w in closest_to_testwords:
        idx = mi.word2idx_m[w]
        plt.plot(Us[idx, 0], Us[idx, 1], 'o')
        plt.text(Us[idx, 0], Us[idx, 1], mi.idx2word_m[idx])
        for closest in closest_to_testwords[w]:
            idx = mi.word2idx_m[closest]
            plt.plot(Us[idx, 0], Us[idx, 1], '+')
            plt.text(Us[idx, 0], Us[idx, 1], corpus.unique_words[idx])
    plt.show()

    plt.clf()
    for w in closest_to_testwords:
        idx = mi.word2idx_m[w]
        plt.plot(Us[idx, 0], Us[idx, 2], 'o')
        plt.text(Us[idx, 0], Us[idx, 2], mi.idx2word_m[idx])
        for closest in closest_to_testwords[w]:
            idx = mi.word2idx_m[closest]
            plt.plot(Us[idx, 0], Us[idx, 2], '+')
            plt.text(Us[idx, 0], Us[idx, 2], mi.idx2word_m[idx])
    plt.show()

    for w in closest_to_testwords:
        idx = mi.word2idx_m[w]
        plt.plot(Us[idx, 1], Us[idx, 2], 'o')
        plt.text(Us[idx, 1], Us[idx, 2], mi.idx2word_m[idx])
        for closest in closest_to_testwords[w]:
            idx = mi.word2idx_m[closest]
            plt.plot(Us[idx, 1], Us[idx, 2], '+')
            plt.text(Us[idx, 1], Us[idx, 2], mi.idx2word_m[idx])
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    for w in closest_to_testwords:
        i = mi.word2idx_m[w]
        ax.plot(Us[i, 0], Us[i, 1], Us[i, 2], '*')
        ax.text(Us[i, 0], Us[i, 1], Us[i, 2], mi.idx2word_m[i])
        for closest in closest_to_testwords[w]:
            i = mi.word2idx_m[closest]
            ax.plot(Us[i, 0], Us[i, 1], Us[i, 2], '+')
            ax.text(Us[i, 0], Us[i, 1], Us[i, 2], mi.idx2word_m[i])
    plt.show()

    exit()
    # Below testing visualization
    for i in range(len(mi.idx2word_m))[:100]:
        plt.plot(Us[i, 0], Us[i, 1], '+')
        plt.text(Us[i, 0], Us[i, 1], mi.idx2word_m[i])
    plt.show()

    plt.clf()
    for i in range(len(mi.idx2word_m))[:100]:
        plt.plot(Us[i, 0], Us[i, 2], '+')
        plt.text(Us[i, 0], Us[i, 2], mi.idx2word_m[i])
    plt.show()

    plt.clf()
    for i in range(len(mi.idx2word_m))[:100]:
        plt.plot(Us[i, 1], Us[i, 2], '+')
        plt.annotate(mi.idx2word_m[i], (Us[i, 1], Us[i, 2]))
    plt.show()

# eigen = True, PPMI = True, centering = True, std = False, raw = False, size = 500
# he ['he', 'she', 'have', 'it', 'i', 'and', 'had', 'they', 'never', 'we']
# she ['she', 'he', 'florence', 'nobody', 'mother', 'i', 'they', 'father', 'leicester', 'have']
# paris ['paris', 'rome', 'england', 'richmond', 'venice', 'italy', 'bevis', 'naples', 'yarmouth', 'ipswich']
# table ['table', 'chair', 'floor', 'breakfast', 'room', 'dinner', 'sofa', 'stool', 'parlour', 'tray']
# rare ['rare', 'treacherous', 'fortunate', 'stupid', 'queer', 'genuine', 'glimpses', 'clever', 'sore', 'jolly']
# monday ['monday', 'tuesday', 'saturday', 'thursday', 'wednesday', 'sunday', 'week', 'morning', 'evening', 'christmas']
# sunday ['sunday', 'saturday', 'wednesday', 'evening', 'week', 'day', 'monday', 'morning', 'summer', 'tuesday']

# eigen = True, PPMI = True, centering = True, std = False, raw = False, size = 300
# Centered by sklearn and fbpca
# he ['he', 'she', 'have', 'they', 'it', 'leicester', 'we', 'had', 'florence', 'wegg']
# she ['she', 'he', 'florence', 'nobody', 'they', 'mother', 'joe', 'never', 'father', 'paul']
# paris ['paris', 'london', 'england', 'yarmouth', 'india', 'richmond', 'italy', 'venice', 'bevis', 'naples']
# table ['table', 'chair', 'room', 'sofa', 'floor', 'basket', 'bed', 'parlour', 'desk', 'glass']
# rare ['rare', 'queer', 'clever', 'trifling', 'horrible', 'sore', 'ridiculous', 'genuine', 'treacherous', 'fortunate']
# monday ['monday', 'tuesday', 'saturday', 'wednesday', 'thursday', 'friday', 'sunday', 'week', 'morrow', 'evening']
# sunday ['sunday', 'saturday', 'evening', 'morning', 'night', 'day', 'wednesday', 'week', 'summer', 'afternoon']

# eigen = True, PPMI = True, centering = True, std = False, raw = True, size = 300
# Centered by sklearn. not centered by fbpca
# he ['he', 'she', 'have', 'they', 'it', 'leicester', 'we', 'had', 'florence', 'wegg']
# she ['she', 'he', 'florence', 'nobody', 'they', 'mother', 'joe', 'never', 'father', 'paul']
# paris ['paris', 'london', 'england', 'yarmouth', 'india', 'richmond', 'italy', 'venice', 'bevis', 'naples']
# table ['table', 'chair', 'room', 'sofa', 'floor', 'basket', 'bed', 'parlour', 'desk', 'glass']
# rare ['rare', 'queer', 'clever', 'trifling', 'horrible', 'sore', 'ridiculous', 'genuine', 'treacherous', 'fortunate']
# monday ['monday', 'tuesday', 'saturday', 'wednesday', 'thursday', 'friday', 'sunday', 'week', 'morrow', 'evening']
# sunday ['sunday', 'saturday', 'evening', 'morning', 'night', 'day', 'wednesday', 'week', 'summer', 'afternoon']

# eigen = True, PPMI = True, centering = False, std = False, raw = False, size = 300
# Not centered by sklearn. Centered by fbpca
# he ['he', 'she', 'have', 'they', 'it', 'leicester', 'we', 'had', 'florence', 'wegg']
# she ['she', 'he', 'florence', 'nobody', 'they', 'mother', 'joe', 'never', 'father', 'paul']
# paris ['paris', 'london', 'england', 'yarmouth', 'india', 'richmond', 'italy', 'venice', 'bevis', 'naples']
# table ['table', 'chair', 'room', 'sofa', 'floor', 'basket', 'bed', 'parlour', 'desk', 'glass']
# rare ['rare', 'queer', 'clever', 'trifling', 'horrible', 'sore', 'ridiculous', 'genuine', 'treacherous', 'fortunate']
# monday ['monday', 'tuesday', 'saturday', 'wednesday', 'thursday', 'friday', 'sunday', 'week', 'morrow', 'evening']
# sunday ['sunday', 'saturday', 'evening', 'morning', 'night', 'day', 'wednesday', 'week', 'summer', 'afternoon']

# eigen = True, PPMI = True, centering = False, std = False, raw = True, size = 300
# Not centered
# he ['he', 'haroun', 'she', 'have', 'du', 'they', 'compn', 'leicester', 'it', 'we']
# she ['she', 'he', 'pum', 'disappoints', 'florence', 'nobody', 'they', 'plautus', 'mother', 'joe']
# paris ['paris', 'england', 'india', 'richmond', 'london', 'yarmouth', 'bevis', 'italy', 'ipswich', 'naples']
# table ['table', 'chair', 'sofa', 'room', 'floor', 'basket', 'desk', 'bed', 'parlour', 'glass']
# rare ['rare', 'queer', 'sore', 'clever', 'trifling', 'treacherous', 'genuine', 'glimpses', 'horrible', 'fortunate']
# monday ['monday', 'tuesday', 'saturday', 'wednesday', 'thursday', 'friday', 'week', 'morrow', 'sunday', 'month']
# sunday ['sunday', 'saturday', 'evening', 'morning', 'night', 'wednesday', 'day', 'week', 'summer', 'afternoon']

# eigen = True, PPMI = True, centering = False, std = False, raw = True, size = 500
# he ['he', 'haroun', 'she', 'du', 'have', 'it', 'compn', 'i', 'and', 'had']
# she ['she', 'pum', 'he', 'plautus', 'compn', 'florence', 'nobody', 'mother', 'disappoints', 'father']
# paris ['paris', 'bevis', 'rome', 'england', 'richmond', 'venice', 'italy', 'naples', 'ipswich', 'yarmouth']
# table ['table', 'chair', 'floor', 'breakfast', 'sofa', 'stool', 'dinner', 'tray', 'room', 'desk']
# rare ['rare', 'treacherous', 'fortunate', 'stupid', 'glimpses', 'genuine', 'conveniences', 'queer', 'sore', 'duster']
# monday ['monday', 'tuesday', 'saturday', 'thursday', 'wednesday', 'sunday', 'week', 'christmas', 'morrow', 'sabbath']
# sunday ['sunday', 'saturday', 'wednesday', 'evening', 'week', 'monday', 'day', 'morning', 'summer', 'tuesday']

# eigen = True, PPMI = True, centering = False, std = False, raw = True, size = 100
# he ['he', 'she', 'haroun', 'they', 'we', 'have', 'never', 'i', 'it', 'soon']
# she ['she', 'he', 'never', 'they', 'florence', 'nobody', 'have', 'often', 'had', 'we']
# paris ['paris', 'england', 'yarmouth', 'india', 'london', 'venice', 'france', 'newgate', 'marseilles', 'clerkenwell']
# table ['table', 'chair', 'window', 'door', 'room', 'shop', 'box', 'wall', 'bed', 'breakfast']
# rare ['rare', 'fearful', 'delightful', 'fortunate', 'painful', 'subtle', 'sad', 'ridiculous', 'stronger', 'trifling']
# monday ['monday', 'saturday', 'week', 'thursday', 'tuesday', 'christmas', 'sunday', 'month', 'wednesday', 'morrow']
# sunday ['sunday', 'saturday', 'week', 'evening', 'hour', 'morning', 'day', 'afternoon', 'stage', 'quarter']

# eigen = True, PPMI = True, centering = False, std = False, raw = True, size = 300, normalized=True
# he ['he', 'she', 'it', 'they', 'i', 'then', 'there', 'by', 'we', 'him']
# she ['she', 'he', 'it', 'they', 'i', 'then', 'we', 'there', 'now', 'never']
# paris ['paris', 'india', 'ipswich', 'naples', 'richmond', 'bevis', 'london', 'england', 'yarmouth', 'rome']
# table ['table', 'room', 'door', 'house', 'window', 'chair', 'fire', 'ground', 'hand', 'them']
# rare ['rare', 'common', 'ridiculous', 'clever', 'painful', 'queer', 'treacherous', 'liberal', 'sore', 'dreadful']
# monday ['monday', 'saturday', 'tuesday', 'sunday', 'wednesday', 'thursday', 'christmas', 'morrow', 'sabbath', 'morning']
# sunday ['sunday', 'saturday', 'night', 'day', 'evening', 'morning', 'week', 'christmas', 'month', 'wednesday']
# man ['man', 'woman', 'gentleman', 'lady', 'boy', 'girl', 'child', 'person', 'face', 'always']
# woman ['woman', 'man', 'lady', 'boy', 'gentleman', 'girl', 'child', 'creature', 'fellow', 'face']
# king ['king', 'saint', 'dog', 'chandler', 'duke', 'name', 'crow', 'comforters', 'cheesemonger', 'intelligibility']
# queen ['queen', 'cavendish', 'bloomsbury', 'smith', 'montagu', 'grosvenor', 'belgrave', 'fitzroy', 'hanover', 'walcot']
# boy ['boy', 'girl', 'woman', 'child', 'man', 'lady', 'gentleman', 'mother', 'father', 'creature']
# girl ['girl', 'boy', 'woman', 'lady', 'child', 'man', 'creature', 'gentleman', 'mother', 'fellow']

# eigen = False, PPMI = True, centering = False, std = False, raw = True, size = 300, normalized=True, UNORDERED=True
# he ['he', 'she', 'it', 'they', 'by', 'i', 'then', 'him', 'there', 'had']
# she ['she', 'he', 'they', 'it', 'then', 'i', 'him', 'by', 'never', 'we']
# paris ['paris', 'doctus', 'dictator', 'daintier', 'diabolic', 'confederates', 'selections', 'unbuilt', 'recommendations', 'sapparised']
# table ['table', 'room', 'house', 'them', 'him', 'door', 'again', 'himself', 'window', 'me']
# rare ['rare', 'common', 'platonic', 'economical', 'stipendiaries', 'liberal', 'doggerel', 'wiolincellers', 'featherless', 'shaming']
# monday ['monday', 'saturday', 'tuesday', 'wednesday', 'unrefreshed', 'sunday', 'ensuing', 'january', 'christmas', 'morrow']
# sunday ['sunday', 'saturday', 'night', 'monday', 'christmas', 'wednesday', 'month', 'morning', 'wintry', 'summer']

# eigen = True, PPMI = True, centering = False, std = False, raw = True, size = 300, normalized=True, UNORDERED=False
# he ['he', 'she', 'i', 'they', 'sir', 'you', 'who', 'it', 'there', 'yes']
# she ['she', 'he', 'i', 'sir', 'you', 'they', 'it', 'there', 'who', 'yes']
# paris ['paris', 'london', 'yarmouth', 'england', 'them', 'everything', 'us', 'hers', 'vain', 'him']
# table ['table', 'door', 'street', 'room', 'fire', 'window', 'house', 'world', 'wall', 'ground']
# rare ['rare', 'fine', 'handsome', 'justly', 'humble', 'low', 'obscure', 'strange', 'pretty', 'childish']
# monday ['monday', 'garter', 'lookers', 'pikes', 'turrets', 'hangers', 'alms', 'cipher', 'zigzagged', 'driftin']
# sunday ['sunday', 'curtsey', 'canter', 'week', 'splash', 'smile', 'moment', 'glance', 'whisper', 'fortnight']
# man ['man', 'gentleman', 'lady', 'woman', 'boy', 'child', 'girl', 'fellow', 'person', 'moment']
# woman ['woman', 'man', 'gentleman', 'lady', 'boy', 'girl', 'child', 'fellow', 'person', 'dog']
# king ['king', 'widow', 'coachman', 'mendicant', 'clergyman', 'mayor', 'chemist', 'wheelwright', 'seaman', 'bride']
# queen ['queen', 'truck', 'littleness', 'margin', 'model', 'pilgrim', 'king', 'cottager', 'sneezing', 'chairman']
# boy ['boy', 'child', 'man', 'girl', 'gentleman', 'lady', 'house', 'room', 'woman', 'door']
# girl ['girl', 'boy', 'child', 'man', 'doctor', 'gentleman', 'lady', 'captain', 'woman', 'world']

# Last experiments, reproducing the one above
# VECTOR_SIZE = 300
# STANDARDIZE = False
# RAW = True  # RAW = False is the same as STANDARDIZE = True
# EIGEN_POWER = 1  # 0 (no eigen values), 0.5, or 1 (True SVD)
# POWER = 1 # Power to smooth the means
# UNORDERED_PAIRS = False
# CONTEXT_SIZE = 3
# CUTOFF_C = 1  # Minimal number of contexts
# CUTOFF_W = 1  # Minimal number of words
# he ['he', 'she', 'i', 'they', 'sir', 'you', 'who', 'it', 'there', 'yes']
# she ['she', 'he', 'i', 'sir', 'you', 'they', 'it', 'there', 'who', 'yes']
# paris ['paris', 'london', 'yarmouth', 'england', 'them', 'everything', 'us', 'hers', 'vain', 'him']
# table ['table', 'door', 'street', 'room', 'fire', 'window', 'house', 'world', 'wall', 'ground']
# rare ['rare', 'fine', 'handsome', 'justly', 'humble', 'low', 'obscure', 'strange', 'pretty', 'childish']
# monday ['monday', 'garter', 'lookers', 'pikes', 'turrets', 'hangers', 'alms', 'cipher', 'zigzagged', 'driftin']
# sunday ['sunday', 'curtsey', 'canter', 'week', 'splash', 'smile', 'moment', 'glance', 'whisper', 'fortnight']
# man ['man', 'gentleman', 'lady', 'woman', 'boy', 'child', 'girl', 'fellow', 'person', 'moment']
# woman ['woman', 'man', 'gentleman', 'lady', 'boy', 'girl', 'child', 'fellow', 'person', 'dog']
# king ['king', 'widow', 'coachman', 'mendicant', 'clergyman', 'mayor', 'chemist', 'wheelwright', 'seaman', 'bride']
# queen ['queen', 'truck', 'littleness', 'margin', 'model', 'pilgrim', 'king', 'cottager', 'sneezing', 'chairman']
# boy ['boy', 'child', 'man', 'girl', 'gentleman', 'lady', 'house', 'room', 'woman', 'door']
# girl ['girl', 'boy', 'child', 'man', 'doctor', 'gentleman', 'lady', 'captain', 'woman', 'world']
# distance 10 closest words sunday 0.5393080052355959 ... 0.5984043910131067
# distance 10 closest words monday 0.25144281036563376 ... 0.44325403965415455
# distance sunday-saturday 0.844149857098924
# distance monday-sunday 0.8589831856665873
# distance monday-saturday 0.7076419493364201
# Semantic operations:
# king - man + woman: ['king', 'seaman', 'wheelwright', 'mendicant', 'pastrycook', 'widow', 'chandler', 'cheesemonger', 'dyer', 'packer']
# queen - woman + man: ['queen', 'truck', 'model', 'margin', 'chairman', 'littleness', 'mug', 'pilgrim', 'midshipman', 'funeral']
# he - man + woman: ['he', 'she', 'i', 'they', 'sir', 'who', 'there', 'never', 'it', 'you']
# she - woman + man: ['she', 'he', 'i', 'you', 'sir', 'they', 'it', 'that', 'there', 'yes']
# boy - man + woman: ['boy', 'woman', 'girl', 'child', 'gentleman', 'lady', 'man', 'house', 'dog', 'room']
