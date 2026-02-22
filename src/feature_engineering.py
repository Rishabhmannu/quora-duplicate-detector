"""
Feature extraction for Quora question pairs.
"""
import distance
from fuzzywuzzy import fuzz
import numpy as np

from .preprocessing import preprocess

# Use NLTK stopwords (no pickle dependency)
try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))

SAFE_DIV = 0.0001


def _common_words(q1: str, q2: str) -> int:
    w1 = set(word.lower().strip() for word in q1.split())
    w2 = set(word.lower().strip() for word in q2.split())
    return len(w1 & w2)


def _total_words(q1: str, q2: str) -> int:
    w1 = set(word.lower().strip() for word in q1.split())
    w2 = set(word.lower().strip() for word in q2.split())
    return len(w1) + len(w2)


def _fetch_token_features(q1: str, q2: str) -> list:
    token_features = [0.0] * 8

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set(w for w in q1_tokens if w not in STOP_WORDS)
    q2_words = set(w for w in q2_tokens if w not in STOP_WORDS)
    q1_stops = set(w for w in q1_tokens if w in STOP_WORDS)
    q2_stops = set(w for w in q2_tokens if w in STOP_WORDS)

    common_word_count = len(q1_words & q2_words)
    common_stop_count = len(q1_stops & q2_stops)
    common_token_count = len(set(q1_tokens) & set(q2_tokens))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features


def _fetch_length_features(q1: str, q2: str) -> list:
    length_features = [0.0] * 3

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    # Guard against empty lcsubstrings (IndexError)
    strs = list(distance.lcsubstrings(q1, q2))
    if strs:
        length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    else:
        length_features[2] = 0.0

    return length_features


def _fetch_fuzzy_features(q1: str, q2: str) -> list:
    return [
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2),
    ]


def _jaccard_similarity(q1: str, q2: str) -> float:
    """|intersection| / |union| of word sets."""
    w1 = set(word.lower().strip() for word in q1.split())
    w2 = set(word.lower().strip() for word in q2.split())
    if not w1 and not w2:
        return 0.0
    inter = len(w1 & w2)
    union = len(w1 | w2)
    return inter / union if union else 0.0


def _sentence_length_ratio(q1: str, q2: str) -> float:
    """min(word_count) / max(word_count)."""
    n1, n2 = len(q1.split()), len(q2.split())
    if max(n1, n2) == 0:
        return 0.0
    return min(n1, n2) / max(n1, n2)


def query_point_creator(
    q1: str, q2: str, vectorizer, embedding_model=None
) -> np.ndarray:
    """
    Build feature vector for a question pair.
    Requires a fitted CountVectorizer or TfidfVectorizer.
    If embedding_model provided, adds cosine similarity between question embeddings.
    """
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    input_query = [
        len(q1),
        len(q2),
        len(q1.split()),
        len(q2.split()),
        _common_words(q1, q2),
        _total_words(q1, q2),
        round(_common_words(q1, q2) / (_total_words(q1, q2) + SAFE_DIV), 2),
    ]
    input_query.extend(_fetch_token_features(q1, q2))
    input_query.extend(_fetch_length_features(q1, q2))
    input_query.extend(_fetch_fuzzy_features(q1, q2))
    input_query.append(_jaccard_similarity(q1, q2))
    input_query.append(_sentence_length_ratio(q1, q2))

    # Sentence Transformer cosine similarity (semantic)
    if embedding_model is not None:
        from .embeddings import embedding_cosine_similarity
        input_query.append(embedding_cosine_similarity(q1, q2, embedding_model))

    q1_vec = vectorizer.transform([q1]).toarray()
    q2_vec = vectorizer.transform([q2]).toarray()

    n_handcrafted = len(input_query)
    return np.hstack((np.array(input_query).reshape(1, n_handcrafted), q1_vec, q2_vec))
