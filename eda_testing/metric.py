from typing import List
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
from IPython.display import display


def accuracy_k(scores: List[int]) -> float:
    """
    Function for accuracy for top-k recommendations calculation
    :param scores: list of bools
    :return: accuracy metric
    """
    return round(sum(scores) / len(scores), 2)


def get_recommendations(model, corpus_embeddings: torch.Tensor, dealer_names: pd.DataFrame,
                        dealer_product_key: int, kn=3) -> List[int]:
    """
    Function that gives k-recommended names from Procept product base
    :param model: model for embeddings
    :param corpus_embeddings: corpus of product names from Prosept transformed to embeddings
    :param dealer_names: pd.Series with names from dealers
    :param dealer_product_key: dealer product name to which match recommendations
    :param k: number of recommended items
    :return best_idx: list of recommended products_id
    """
    query = dealer_names[dealer_product_key]
    query_embedding = model.encode(query, convert_to_tensor=True)

    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=kn)

    best_idx = []

    for score, idx in zip(top_results[0], top_results[1]):
        score = score.cpu().data.numpy()
        idx = idx.cpu().data.numpy()
        best_idx.append(idx)

    return best_idx


def recommendations_for_metric(model, corpus_embeddings: torch.Tensor, dealer_names_embeddings: pd.DataFrame,
                               marketing_dealerprice: pd.DataFrame, marketing_productdealerkey: pd.DataFrame,
                               marketing_product: pd.DataFrame, k=3) -> float:
    """
    Generates recommendations for a accuracy metric.
    :param marketing_product:
    :param marketing_productdealerkey:
    :param marketing_dealerprice:
    :param model: model for embeddings
    :param dealer_names_embeddings: pd.Series with names from dealers
    :param corpus_embeddings: corpus of product names from Prosept transformed to embeddings
    :param k: number of recommendations to generate (default is 3).
    :return:  A list of scores indicating whether the recommended products match the true product ID.
    """
    scores = []

    for idx in tqdm(dealer_names_embeddings.index):

        product_key = marketing_dealerprice.product_key[idx]
        dealer_id = marketing_dealerprice.dealer_id[idx]
        true_id = marketing_productdealerkey[
            (marketing_productdealerkey.dealer_id == dealer_id) & (marketing_productdealerkey.key == product_key)][
            'product_id'].values
        if not true_id:
            pass
        else:
            best_idx = get_recommendations(model, corpus_embeddings, dealer_names_embeddings, idx, k)

            recom_product_id = [x for x in marketing_product.id.iloc[best_idx]]

            score = any(i == true_id for i in recom_product_id)
            scores.append(score)

    return accuracy_k(scores)


def show_recommendations(marketing_dealerprice: pd.DataFrame, marketing_productdealerkey: pd.DataFrame,
                         marketing_product: pd.DataFrame, dealer_product_key: int, best_idx: List[int]):
    print(f'Query: {marketing_dealerprice.product_name[dealer_product_key]}')
    print('-' * 50)
    product_key = marketing_dealerprice.product_key[dealer_product_key]
    dealer_id = marketing_dealerprice.dealer_id[dealer_product_key]
    true_id = marketing_productdealerkey[
        (marketing_productdealerkey.dealer_id == dealer_id) & (marketing_productdealerkey.key == product_key)][
        'product_id'].values

    recommendations = marketing_product.iloc[best_idx]
    score = any(i == true_id for i in recommendations.id.values)
    print('Есть совпадение' if score else 'Совпадений нет')
    print('-' * 50)
    display(recommendations)
