from pathlib import Path

import pyLDAvis
import pyLDAvis.gensim
from gensim.corpora import Dictionary


def generate_ldavis(model, corpus, dictionary: Dictionary, file_name: str, path='../references'):
    vis = pyLDAvis.gensim.prepare(model, corpus, dictionary, sort_topics=False)
    save_path = Path(path) / f'{file_name}.html'
    pyLDAvis.save_html(vis, str(save_path))
    print(f'vis generated to {save_path}')
