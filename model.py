import numpy as np
import pandas as pd
from tensorflow_text import SentencepieceTokenizer
import tensorflow_hub as hub
from sklearn.metrics import pairwise
from typing import List, Tuple
from text_processing import normalize_corpus, split_text

# load the MUSE language model
module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'
model = hub.load(module_url)


def embed_text(input):
    """
    Embeds the sentences using the MUSE language model.
    Args:
        input: list of senctences to embed.
    Returns:
        list of Tensors.

    """
    return model(input)


def get_similarities(simtype: str, original_article: dict, found_articles: List[dict]) -> List[Tuple[str, float]]:
    """
    Manipulates the articles to get the similarty between the two.
    Args:
        type: The type for getting the similarity.
        original_article: One of the articles to get the similarity.
        found_articles: One of the articles to get the similarity.

    Returns:
        List[Tuple[str, float]]: A list of tuples containing the url's of the found articles with their corresponding similarity scores.
    """

    original_article_cleaned = normalize_corpus(original_article["article"], stopwords_removal=False,
                                                lan=original_article["language"],
                                                stemming=False, remove_special_characters=True)
    split_original_article = split_text(original_article_cleaned, original_article["language"])
    embedded_original_article = embed_text(split_original_article)
    sim_scores = []

    for found_article in found_articles:
      found_article_cleaned = normalize_corpus(found_article["article"], stopwords_removal=False, lan=found_article["language"],
                                             stemming=False, remove_special_characters=True)
      split_found_articles = split_text(found_article_cleaned, found_article["language"] )
      embedded_found_aricle = embed_text(split_found_articles)
      sim = 0
      if simtype == "cosine":
          sim = pairwise.cosine_similarity(embedded_original_article, embedded_found_aricle)
      else:
          sim = pairwise.euclidean_distances(embedded_original_article, embedded_found_aricle) 
      df = convert_to_dataframe(embedded_original_article, embedded_found_aricle, sim)
      high_avg_sim = highest_similarity(df)
      sim_scores.append({"url": found_article["url"], "sim": str(high_avg_sim)})

    return sim_scores

def convert_to_dataframe(embeddings_1, embeddings_2, sim) -> pd.DataFrame:
    """
    Convert the arguments in one dataframe.
    Args:
        embeddings_1: The embeddings from one of the articles.
        embeddings_2: The embeddings from one of the articles.
        sim: List of similarities of the two articles.

    Returns:
        A Dataframe containing all the arguments.
        And a list of the similarities.

    """
    sim_col, col_1_num, col_2_num = [], [], []

    for i in range(len(embeddings_1)):
        for j in range(len(embeddings_2)):
            sim_col.append(sim[i][j])
            col_1_num.append(i)
            col_2_num.append(j)

    df = pd.DataFrame(zip(col_1_num, col_2_num, sim_col),
                      columns=['index_sent_1', 'index_sent_2', 'sim'])

    return df


def avg_similarity(similar, split_text) ->  np.ndarray:
    """
    Gets the average similarity
    Args:
        similar: List of similarities of two articles
        split_text: List of sentences to use.

    Returns:
        Average similarity of the split_text
    """
    highest_similarity = []
    for i in range(0, len(split_text)):
        # print(similar[i:i+len(english_text_split)])
        num = max(similar[i:i + len(split_text)])
        highest_similarity.append(num)

    average_similarity = np.mean(highest_similarity)
    print(f'Average similarity for the document is {average_similarity}')
    return average_similarity


def highest_similarity(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Gets the average similarity from the highest of one article.
    Args:
        dataframe: The dataframe containing all sentences an similarities between them.

    Returns:
        the average similarity from the highest similarity of the found article.
    """
    max_sim = []
    for i in dataframe["index_sent_1"].unique():
        max_sim.append(dataframe.loc[dataframe["index_sent_1"] == i, ["sim"]].max())
    average_similarity = np.mean(max_sim)
    return average_similarity
