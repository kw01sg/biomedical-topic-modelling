from gensim.corpora import Dictionary
from gensim.models import CoherenceModel


def create_dictionary(docs: list, filter_extreme=True):
    """Helper function to create a Gensim Dictionary object
    """
    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)

    # Filter out words more than 50% of the documents.
    if filter_extreme:
        dictionary.filter_extremes(no_above=0.5)

    return dictionary


def get_coherence(model, docs: list, dictionary: Dictionary):
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(
        model=model, texts=docs, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    return coherence_lda
