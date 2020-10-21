# biomedical-topic-modelling

## Project Structure

```
biomedical-topic-modelling
├── artefacts                                       <- artefacts generated from modelling process
├── data
│   ├── Associated words.xlsx
│   ├── cordis-h2020projects.xlsx                   <- list of projects/grants
│   ├── Text mining word list test 200823.xlsx
│   └── topics_300_SYinput_LW.csv                   <- manual labels topics from baseline topic model
├── mallet-2.0.8    <- executable for LDA Mallet. Refer to Topic Modeling with Gensim (Python) in Reference
├── notebooks
│   ├── 1_EDA.ipynb
│   ├── 2_LDA.ipynb
│   ├── 3_filter_dataset.ipynb
│   └── 4_validate_topics.ipynb
├── output
├── references
├── src
│   ├── artefacts_helper.py
│   ├── gensim_helper.py
│   ├── __init__.py
│   ├── predict.py
│   ├── process_data.py
│   ├── train.py
│   └── visualize.py
├── environment.yml
└── README.md
```

## Data

* `cordis-h2020projects.xlsx` contains list of projects/grants with `title` and `objective` fields that are used for topic modelling
* `topics_300_SYinput_LW.csv` contains manual labels of relevant and irrelevant topics generated from baseline topic model. Labels are used in `notebooks/3. filter_dataset.ipynb` to filter `cordis-h2020projects.xlsx`.

## Getting Started

### Installation

Python virtual environment is managed using Conda. 

```bash
$ conda env create -f environment.yml
$ conda activate textmining

# then run your python scripts or notebook
$ jupyter notebook
```

### Run Jupyter Notebooks

Most of the experimentations are done in Jupyter Notebooks.

```bash
$ conda activate textmining
$ jupyter notebook
```

### Training Gensim LDAMallet Model

```bash
$ python -m src.train --help
Usage: train.py [OPTIONS]

Options:
  -i, --input TEXT          [required]
  -n, --num_topics INTEGER  [required]
  -s, --suffix TEXT         [required]
  --save-model
  --help                    Show this message and exit.

$ python -m src.train --input ./data/cordis-h2020projects.xlsx --num_topics 10 --suffix temp --save-model
```

### Loading LDA Model

```python
from src.artefacts_helper import load_mallet_model

model = load_mallet_model(artefacts_path='./artefacts/', suffix='temp')
```

### General Usage

```python
import pandas as pd
from src.gensim_helper import create_dictionary
from src.predict import predict_and_format_topics
from src.process_data import process_data
from src.train import train_lda_mallet

sheets_dict = pd.read_excel('./data/cordis-h2020projects.xlsx', None)
df = sheets_dict['cordis-h2020projects']

# combine title and objective
data = (df['title'] + ' ' + df['objective']).values.tolist()

docs = process_data(data)
dictionary = create_dictionary(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]
model = train_lda_mallet(corpus, dictionary, 10,
                         params={
                             'mallet_path': './mallet-2.0.8/bin/mallet',
                             'prefix_path': './artefacts/mallet_tmp/',
                             'prefix': 'example'
                         })
predict_df = predict_and_format_topics(model, corpus, data)
```

## Possible Improvements

* Using other types of topic models beside LDA
* Using bigrams and trigrams to process raw data
* Using other metrics to measure performance of generated topics

## References

* [Intuitive Guide to Latent Dirichlet Allocation](https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158)
* [Topic Modeling with Gensim (Python)](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python)
* [Evaluate Topic Models: Latent Dirichlet Allocation (LDA)](https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0)
* [Demo of Gensim’s LDA model](https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html)
* [Gensim LDA API Documentation](https://radimrehurek.com/gensim/models/ldamodel.html#gensim.models.ldamodel.LdaModel.get_document_topics)
* [pyLDAvis](https://pyldavis.readthedocs.io/en/latest/readme.html)
* [Explanation of LDAvis visualizations](https://cran.r-project.org/web/packages/LDAvis/vignettes/details.pdf)
