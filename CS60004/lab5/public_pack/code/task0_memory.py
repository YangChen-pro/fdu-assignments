import numpy as np
import re


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5\s]", " ", text)
    return text


def lexical_overlap(query, memory_text):
    q_words = set(normalize_text(query).split())
    m_words = set(normalize_text(memory_text).split())
    if len(q_words) == 0:
        return 0.0
    return len(q_words & m_words) / len(q_words)


def retrieve_memories(
    query,
    query_embedding,
    memories,
    top_k=5,
    alpha=0.65,
    beta=0.25,
    gamma=0.10,
):
    """
    memories: List[Dict]
    Each memory contains:
    {
        "memory_id": str,
        "text": str,
        "embedding": List[float],
        "importance": float,
        "type": str
    }

    score = alpha * semantic_similarity
          + beta * lexical_overlap
          + gamma * importance
    """

    # ====================== code starts here ======================
    # TODO: compute semantic_score, lexical_score, importance_score
    # TODO: sort memories by the final score
    # TODO: return top_k memories with score fields
    #
    # Hints:
    # 1. semantic_similarity should use cosine similarity.
    # 2. lexical_score should use lexical_overlap(query, memory["text"]).
    # 3. importance_score should use memory["importance"].
    # 4. Each returned memory should contain:
    #    score / semantic_score / lexical_score / importance_score.
    # ====================== code ends here ======================

    return selected_memories


def print_check(name, passed):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")


def run_demo():
    query = "How was your first Mars mission?"
    query_embedding = np.array([1.0, 0.0])

    memories = [
        {
            "memory_id": "m_space",
            "type": "experience",
            "text": "I still remember my first Mars mission and the zero gravity training before launch.",
            "embedding": [0.99, 0.10],
            "importance": 0.7,
        },
        {
            "memory_id": "m_food",
            "type": "hobby",
            "text": "I enjoy cooking pasta for my friends during holidays.",
            "embedding": [0.0, 1.0],
            "importance": 0.9,
        },
        {
            "memory_id": "m_award",
            "type": "career",
            "text": "The NASA award ceremony after the mission made me proud.",
            "embedding": [0.70, 0.70],
            "importance": 0.8,
        },
    ]

    results = retrieve_memories(
        query=query,
        query_embedding=query_embedding,
        memories=memories,
        top_k=2,
    )

    print("Retrieved memories:")
    for rank, item in enumerate(results, start=1):
        print(
            f"{rank}. {item.get('memory_id')} "
            f"score={item.get('score'):.4f} "
            f"semantic={item.get('semantic_score'):.4f} "
            f"lexical={item.get('lexical_score'):.4f} "
            f"importance={item.get('importance_score'):.4f}"
        )

    print("\nChecks:")
    print_check("top-1 memory is m_space", len(results) > 0 and results[0].get("memory_id") == "m_space")
    print_check("top-2 memory is m_award", len(results) > 1 and results[1].get("memory_id") == "m_award")


if __name__ == "__main__":
    run_demo()
