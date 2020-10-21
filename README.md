# biomedical-topic-modelling

## Project Structure

```
biomedical-topic-modelling
├── artefacts                                       <- artefacts generated from modelling process
├── data
│   ├── Associated words.xlsx
│   ├── cordis-h2020projects.xlsx                   <- list of projects/grants
│   ├── Text mining word list test 200823.xlsx
│   └── topics_300_SYinput_LW.csv
├── mallet-2.0.8    <- executable for LDA Mallet. Refer to Topic Modeling with Gensim (Python) in Reference
├── notebooks
│   ├── bigrams.ipynb
│   ├── EDA.ipynb
│   ├── filter_dataset.ipynb
│   ├── LDA.ipynb
│   ├── Top2Vec.ipynb
│   └── validate_topics.ipynb
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
model.num_topics
```

## References

* [Intuitive Guide to Latent Dirichlet Allocation](https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158)
* [Topic Modeling with Gensim (Python)](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python)
* [Evaluate Topic Models: Latent Dirichlet Allocation (LDA)](https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0)
* [Demo of Gensim’s LDA model](https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html)
* [Gensim LDA API Documentation](https://radimrehurek.com/gensim/models/ldamodel.html#gensim.models.ldamodel.LdaModel.get_document_topics)
* [pyLDAvis](https://pyldavis.readthedocs.io/en/latest/readme.html)
* [Explanation of LDAvis visualizations](https://cran.r-project.org/web/packages/LDAvis/vignettes/details.pdf)
