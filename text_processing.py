import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

stopword_list_nl = stopwords.words('dutch')
stopword_list_en = stopwords.words('english')


def remove_stopwords(text: str, lan: str) -> str:
    """
    Removes the stopwords using the stopword list.
    Args:
        text: Text to use.
        lan: language the use the stopwords from.

    Returns:
        Filtered text.
    """
    if lan == 'dutch':
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token not in stopword_list_nl]
        filtered_text = ' '.join(filtered_tokens)

    if lan == 'english':
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token not in stopword_list_en]
        filtered_text = ' '.join(filtered_tokens)

    return filtered_text


def stem_text(text: str) -> str:
    """
    Stems the text.
    Args:
        text: Text to filter.

    Returns:
        Filtered text.
    """
    ps = PorterStemmer()
    tokens = word_tokenize(text)
    filtered_tokens = [ps.stem(token) for token in tokens]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_characters(text: str) -> str:
    """
    Removes special characters.
    Args:
        text: Text to filter.

    Returns:
        Filtered text.
    """
    re_map = {'[^A-Za-z0-9.?]+': ' '}
    for r, map in re_map.items():
        text = re.sub(r, map, text)
    return text


def normalize_corpus(corpus: str, stopwords_removal: bool=True, lan: str="english",
                     stemming: bool=True, remove_special_characters: bool=True) -> str:
    """
    Filter text with given options
    Args:
        corpus: Text to filter
        stopwords_removal: if stopwords need to be removed.
        lan: The language of the text.
        stemming: If the stemming needs to be removed.
        remove_special_characters: If the special characters need to be removed.

    Returns:
        Filtered text.
    """
    if remove_special_characters:
        doc = remove_characters(corpus)

    if stopwords_removal:
        doc = remove_stopwords(corpus, lan)

    if stemming:
        doc = stem_text(corpus)

    return doc


def split_text(text: str, lan: str) -> str:
    """
    Split the text using sentence tokenize
    Args:
        text: Text to split.
        lan: Language of the text.

    Returns:
        The splitted text.
    """
    text_split = sent_tokenize(text, language=lan)
    return text_split
