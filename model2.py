from typing import List
import pandas as pd
import pickle
import torch
from sentence_transformers import SentenceTransformer, util
from preprocess import clean_text_dealer


def get_recommendations(marketing_dealerprice: pd.DataFrame,
                        dealer_product_key: int, k=3) -> List[int]:
    """
    Function that gives k-recommended names from Procept product base
    :param marketing_dealerprice: dataframe from dealerprice
    :param dealer_product_key: dealer product name to which match recommendations
    :param k: number of recommended items
    :return products_id: list of recommended products_id
    """

    try:
        with open('labse_model.pkl', 'rb') as file:
            model = pickle.load(file)
            print("Модель успешно загружена")

    except:
        model = SentenceTransformer('sentence-transformers/LaBSE')

    try:
        with open('corpus_embeddings.pkl', 'rb') as file:
            corpus_embeddings = pickle.load(file)
            print("Эмбеддинги успешно загружены")

    except FileNotFoundError as err:
        print(f"{err} Необходимо получить эмбеддинги для названий от Procept")

    query = marketing_dealerprice.loc[dealer_product_key][['product_name']]
    query = clean_text_dealer(query)
    query_embedding = model.encode(query, convert_to_tensor=True)

    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=k)

    best_idx = []

    for score, idx in zip(top_results[0], top_results[1]):
        score = score.cpu().data.numpy()
        idx = idx.item()
        best_idx.append(idx)

    return best_idx
