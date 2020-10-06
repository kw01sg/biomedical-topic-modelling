from pathlib import Path

from gensim.models import LdaModel
from gensim.models.wrappers import LdaMallet


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
    return model


def load_mallet_model(artefacts_path='./artefacts', suffix=''):
    model_path = str(Path(artefacts_path) / 'model')
    if suffix:
        model_path = model_path + f'_{suffix}'
    model = LdaMallet.load(model_path)
    return model
