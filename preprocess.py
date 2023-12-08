from math import log
import pandas as pd
from nltk.corpus import stopwords
import re
import string

# Constants
MAGIC_NUMBER = 9e999
STOPWORDS = stopwords.words('russian') + ['что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'к', 'за', 'из', 'из-за', 'с',
                                          'на', 'ок', 'кстати', 'который', 'мочь', 'весь', 'еще', 'также', 'свой',
                                          'ещё',
                                          'самый', 'ул', 'главные', 'играет', 'и', 'y', 'c', 'для', 'prosept',
                                          'просепт',
                                          'для', 'средство', 'кг', 'г', 'мл', 'л', 'шт']
PUNCTUATION = string.punctuation + '«»–'
NUMBERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def infer_spaces(s):
    """
    Uses dynamic programming to infer the location of spaces in a string
    without spaces.
    """
    words = open("new_words.txt").read().split()
    wordcost = dict((k, log((i + 1) * log(len(words)))) for i, k in enumerate(words))
    nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for n in nums:
        wordcost[n] = log(2)
    maxword = max(len(x) for x in words)

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i - maxword):i]))
        return min((c + wordcost.get(s[i - k - 1:i], MAGIC_NUMBER), k + 1) for k, c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1, len(s) + 1):
        c, k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i > 0:
        c, k = best_match(i)
        assert c == cost[i]
        out.append(s[i - k:i])
        i -= k

    return " ".join(reversed(out))


def remove_stopwords(text:str)->str:
    """
    Removes stopwords from the given text and returns the modified text.
    :param text: The input text from which stopwords need to be removed.
    :return: str The modified text after removing stopwords.
    """
    return ' '.join(word for word in text.split() if word not in STOPWORDS)


def create_volume_column(name:pd.Series)->pd.Series:
    """
    Extracts volume information from each element in the name and returns a new Series with the extracted volumes.
    :param name: column contains names
    :return: pd.Series with the extracted volumes.
    """
    volume = name.apply(lambda x: (re.findall(r'\s*\d+\s*(?:л|г|мл|кг|шт|штук)', x)))
    volume = volume.apply(lambda x: "".join(x).replace(" ", ""))
    return volume


def clean_text_dealer(dealer_name:pd.Series)->pd.Series:
    """
     Cleans the given dealer name
    :param dealer_name: pd.Series with names from dealers
    :return: pd.Series with cleaned names
    """
    # Lowercase
    dealer_name = dealer_name.apply(lambda x: x.lower())

    # Remove punctuation
    dealer_name = dealer_name.apply(lambda x: re.sub('[%s]' % re.escape(PUNCTUATION), '', x))

    # Create volume column
    dealer_volume = create_volume_column(dealer_name)

    # Infer spaces for short dealer names
    for i in dealer_name[dealer_name.apply(lambda x: len(x.split()) < 2)].index:
        temp = dealer_name.loc[i]
        infer = infer_spaces(temp)
        dealer_name.loc[i] = infer

    # Remove stopwords and additional cleaning
    dealer_name = dealer_name.apply(lambda x: ' '.join(x.split()))
    dealer_name = dealer_name.apply(lambda x: remove_stopwords(x))
    dealer_name = dealer_name.apply(lambda x: re.sub('\w*\d\w*', '', x))
    dealer_name = dealer_name.apply(lambda x: ' '.join(x.split()))
    dealer_name = dealer_name.apply(lambda x: re.sub('шпаклевка', 'шпатлевка', x))

    # Combine cleaned name with volume
    dealer_corpus = dealer_name + ' ' + dealer_volume

    return dealer_corpus


def clean_text_prosept(product_name:pd.Series)->pd.Series:
    """
     Cleans the given dealer name
    :param dealer_name: pd.Series with names from dealers
    :return: pd.Series with cleaned names
    """
    # Lowercase
    product_name = product_name.apply(lambda x: x.lower())

    # Remove punctuation
    product_name = product_name.apply(lambda x: re.sub('[%s]' % re.escape(PUNCTUATION), '', x))

    # Separate Latin and Cyrillic characters
    product_name = product_name.apply(lambda x: ' '.join(re.split(r'([a-zA-Z]+|[a-zA-Z]+)', x)))

    # Additional cleaning
    product_name = product_name.apply(lambda x: ' '.join(x.split()))
    product_name = product_name.apply(lambda x: x.replace(' редство', ' средство').replace('c ', ''))

    # Create volume column
    volume = create_volume_column(product_name)

    # Remove stopwords and additional cleaning
    product_name = product_name.apply(lambda x: remove_stopwords(x))
    product_name = product_name.apply(lambda x: re.sub('\w*\d\w*', '', x))
    product_name = product_name.apply(lambda x: ' '.join(x.split()))

    # Combine cleaned name with volume
    product_corpus = product_name + ' ' + volume

    return product_corpus
