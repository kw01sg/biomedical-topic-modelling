import gensim
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('wordnet')
nltk.download('stopwords')

# NLTK Stop words
stop_words = stopwords.words('english')


def tokenize(data: list):
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    return [tokenizer.tokenize(i) for i in data]


def lemmatize_data(docs: list):
    lemmatizer = WordNetLemmatizer()
    return [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]


def process_data(data: list):
    docs = tokenize(data)

    # convert to lower case
    docs = [[token.lower() for token in doc] for doc in docs]

    # remove stop words
    docs = [[token for token in doc if token not in stop_words]
            for doc in docs]

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]

    # lemmatize
    docs = lemmatize_data(docs)

    # Possible Area of Improvement: bigrams and trigrams

    return docs
