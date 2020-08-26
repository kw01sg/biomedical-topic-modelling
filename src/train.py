import time
import logging

import click
import pandas as pd
from pathlib import Path

from gensim.models import LdaModel, LdaMulticore
from src.process_data import process_data
from src.gensim_helper import create_dictionary, get_coherence
from src.artefacts_helper import save_model


# Set training parameters.
CHUNK_SIZE = 4000
PASSES = 20
ITERATIONS = 200
EVAL_EVERY = None
ALPHA = 'auto'
ETA = 'auto'
RANDOM_STATE = 100


def train_lda_single_core(corpus, id2word, num_topics, params: dict):
    chunk_size = params.get('chunk_size', CHUNK_SIZE)
    passes = params.get('passes', PASSES)
    iterations = params.get('iterations', ITERATIONS)
    eval_every = None
    alpha = params.get('alpha', 'auto')
    eta = params.get('eta', 'auto')
    random_state = params.get('random_state', RANDOM_STATE)
    eval_every = params.get('eval_every', EVAL_EVERY)

    return LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        chunksize=chunk_size,
        alpha=alpha,
        eta=eta,
        iterations=iterations,
        passes=passes,
        eval_every=eval_every,
        random_state=random_state
    )


def train_lda_multi_core(corpus, id2word, num_topics, params: dict):
    chunk_size = params.get('chunk_size', CHUNK_SIZE)
    passes = params.get('passes', PASSES)
    iterations = params.get('iterations', ITERATIONS)
    eval_every = None
    alpha = params.get('alpha', 'asymmetric')
    eta = params.get('eta', 'auto')
    random_state = params.get('random_state', RANDOM_STATE)
    eval_every = params.get('eval_every', EVAL_EVERY)

    return LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        chunksize=chunk_size,
        alpha=alpha,
        eta=eta,
        iterations=iterations,
        passes=passes,
        eval_every=eval_every,
        random_state=random_state
    )


@click.command()
@click.option('-i', '--input', 'input_path', required=True, type=str)
@click.option('-n', '--num_topics', required=True, type=int)
@click.option('-s', '--suffix', 'model_suffix', required=True, type=str)
@click.option('--save-model', 'save_model_flag', is_flag=True)
def train(input_path, num_topics, model_suffix, save_model_flag):
    df = pd.read_excel(Path(input_path), sheet_name='cordis-h2020projects')
    print('shape of input:', df.shape)

    # combine title and objective
    data = (df['title'] + ' ' + df['objective']).values.tolist()
    docs = process_data(data)
    dictionary = create_dictionary(docs)

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    start = time.time()
    model = train_lda_multi_core(corpus, dictionary, num_topics, params={})
    training_time = time.time() - start
    training_time = time.strftime('%Hh:%Mm:%Ss', time.gmtime(training_time))
    print(f'Training time: {training_time}')
    coherence_score = round(get_coherence(model, docs, dictionary), 5)
    print(f'Coherence Score: {coherence_score}')

    if save_model_flag:
        save_model(model, suffix=model_suffix, path='./artefacts/model')
    print('training completed')


if __name__ == '__main__':
    train()     # pylint: disable=no-value-for-parameter
