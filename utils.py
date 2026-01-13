from scipy.spatial.distance import cosine

def find_best_match(embedding, persons):
    best_id = None
    best_score = 0.0

    for pid, emb_list in persons:
        for e in emb_list:
            score = 1 - cosine(embedding, e)
            if score > best_score:
                best_score = score
                best_id = pid

    return best_id, best_score
