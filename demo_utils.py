from tqdm import tqdm
import pandas as pd
import numpy as np

def batch_insert_chroma(collection, documents, metadatas, ids, batch_size=1000, desc="Ajout dans ChromaDB"):
    #fonction permettant de suivre l'avancée de l'insertion
    total = len(documents)
    for i in tqdm(range(0, total, batch_size), desc=desc):
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )

def cosine_distance_to_similarity(distances):
    similarities = []
    for distance in distances:
        similarities.append(1-distance)
    return similarities


def query_db(vec, start_idx, query_collection_class, df_label, ft = False):

    query_result = query_collection_class.query(
            query_embeddings=vec,
            n_results=10
        )
    results = []
    for i, metas in enumerate(query_result["metadatas"]):
        closest_class = metas[0]['Topic_group']
        top10_classes = [meta['Topic_group'] for meta in metas]
        index = df_label.iloc[start_idx + i].name
        similarities = cosine_distance_to_similarity(query_result["distances"][i])
        results.append({
            "ticket_id": index,
            "true_class": df_label.at[index,"Topic_group"],
            "closest_class": closest_class,
            "top10_classes": top10_classes,
            "similarities": similarities,
            "ft" : ft
        })

    return results


def apply_majority_vote(results_list):
    global_votes = []
    for result in results_list:
        votes = {}
        for label, sim in zip(result["top10_classes"], result['similarities']):
            votes[label] = votes.get(label, 0) + sim
        global_votes.append(votes)

    # Créer le DataFrame des votes
    votes_df = pd.DataFrame(global_votes)
    votes_df.replace(np.nan, 0, inplace=True)

    # Déterminer la classe majoritaire pour chaque ligne
    for idx, row in votes_df.iterrows():
        maximum = (None, 0)
        for index, value in row.items():
            if value > maximum[1]:
                maximum = (index, value)
        results_list[idx]["majority_vote"] = maximum[0]

    # Retourner le DataFrame des votes et le DataFrame des résultats enrichis
    return votes_df, pd.DataFrame(results_list)