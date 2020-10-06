import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel


def predict_and_format_topics(ldamodel: LdaModel, corpus, texts, doc_id=[], n_topics=5):
    df = pd.DataFrame()

    # Get main topic in each document
    for row in ldamodel[corpus]:
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the top n topic and topic probability for each document
        temp_list = []
        for topic_num, prob_topic in row[:n_topics]:
            wp = ldamodel.show_topic(topic_num)
            topic_keywords = ", ".join([word for word, prop in wp])
            temp_list = temp_list + \
                [int(topic_num), round(prob_topic, 4), topic_keywords]
        df = df.append(pd.Series(temp_list), ignore_index=True)

    # Add original text to the end of the output
    # df = pd.concat([df, pd.Series(texts)], axis=1)
    if doc_id:
        df.insert(0, 'Document_No', doc_id)
    else:
        df.reset_index(inplace=True)

    df.columns = ['Document_No'] + np.array(
        [(f'Dominant_Topic_{i+1}', f'Topic_Prob_{i+1}', 'Topic Keywords') for i in range(n_topics)]).flatten().tolist()

    return df


def get_topic_most_dominant_document(formatted_df: pd.DataFrame):
    # Group top 5 sentences under each topic
    df = pd.DataFrame()

    for _, grp in formatted_df.groupby('Dominant_Topic'):
        df = pd.concat([df, grp.sort_values(
            ['Topic_Prob'], ascending=[0]).head(1)])

    # Reset Index
    df.reset_index(drop=True, inplace=True)

    # Format
    df.rename(columns={'Dominant_Topic': 'Topic_Num'}, inplace=True)
    df = df[['Topic_Num', 'Document_No', 'Topic_Prob', 'Topic_Keywords', 'Text']]

    return df


def get_topics_distribution(ldamodel: LdaModel, formatted_df: pd.DataFrame, n_topics=1):
    dist_array = np.zeros((ldamodel.num_topics, n_topics))

    for i in range(n_topics):
        # Number of Documents for Each Topic
        for topic_id, topic_count in formatted_df[f'Dominant_Topic_{i+1}'].value_counts().iteritems():
            if i == 0:
                dist_array[int(topic_id)][i] = int(topic_count)
            else:
                dist_array[int(topic_id)][i] = int(
                    topic_count) + dist_array[int(topic_id)][i-1]

    # Topic Number and Keywords
    df = get_all_topics(ldamodel)

    temp_df = pd.DataFrame(dist_array, columns=[
                           f"top_{i+1}_topics" for i in range(n_topics)])

    return df.merge(temp_df, left_index=True, right_index=True)


def get_term_topics(model: LdaModel, dictionary: Dictionary, term: str):
    if term in dictionary.token2id:
        return model.get_term_topics(dictionary.token2id[term])
    return None


def format_term_search_results(model: LdaModel, search_results: dict):
    temp_list = []

    for key, value in search_results.items():
        sorted_value = sorted(value, key=lambda x: x[1], reverse=True)
        for i in sorted_value:
            topic_id, topic_prob = i
            wp = model.show_topic(topic_id)
            topic_keywords = ", ".join([word for word, prop in wp])
            temp_list.append([key, topic_id, topic_prob, topic_keywords])

    return pd.DataFrame(temp_list,
                        columns=['Search_Term', 'Topic_ID', 'Topic_Prob', 'Topic_Keywords'])


def get_all_topics(model: LdaModel, num_words=10):
    results = []
    for topic in model.show_topics(-1, num_words=num_words, formatted=False):
        results.append([topic[0], ", ".join([word[0] for word in topic[1]])])
    return pd.DataFrame(results, columns=['Topic_Id', 'Topic_Keywords'])
