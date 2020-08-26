from pathlib import Path
import pickle
from gensim.corpora import Dictionary
from gensim.models import LdaModel


def save_model(model: LdaModel, path='../artefacts/model', suffix=''):
    if suffix:
        path = path + '_' + suffix
    model.save(path)
    print(f'model saved at {path}')


def load_model(artefacts_path='../artefacts', suffix=''):
    model_path = str(Path(artefacts_path) / 'model')
    if suffix:
        model_path = model_path + f'_{suffix}'
    model = LdaModel.load(model_path)
    # dictionary = Dictionary.load(str(Path(artefacts_path) / 'dictionary'))
    # with open(Path(artefacts_path) / 'corpus.pkl', 'rb') as f:
    #     corpus = pickle.load(f)

    # return {'model': model, 'dictionary': dictionary, 'corpus': corpus}
    return model
