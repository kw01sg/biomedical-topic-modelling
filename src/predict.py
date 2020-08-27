import pandas as pd
from gensim.models import LdaModel
from gensim.corpora import Dictionary


def predict_and_format_topics(ldamodel: LdaModel, corpus, texts):
    df = pd.DataFrame()

    # Get main topic in each document
    for row in ldamodel[corpus]:
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Topic Probability and Keywords for each document
        topic_num, prob_topic = row[0]
        wp = ldamodel.show_topic(topic_num)
        topic_keywords = ", ".join([word for word, prop in wp])
        df = df.append(pd.Series(
            [int(topic_num), round(prob_topic, 4), topic_keywords]), ignore_index=True)

    # Add original text to the end of the output
    df = pd.concat([df, pd.Series(texts)], axis=1)
    df.reset_index(inplace=True)
    df.columns = ['Document_No', 'Dominant_Topic',
                  'Topic_Prob', 'Topic_Keywords', 'Text']

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


def get_topics_distribution(formatted_df: pd.DataFrame):
    # Number of Documents for Each Topic
    topic_counts = formatted_df['Dominant_Topic'].value_counts()

    # Percentage of Documents for Each Topic
    topic_contribution = round(topic_counts/topic_counts.sum(), 4)

    # Topic Number and Keywords
    df = formatted_df[['Dominant_Topic', 'Topic_Keywords']
                      ].sort_values('Dominant_Topic').drop_duplicates()
    df.set_index('Dominant_Topic', inplace=True)

    # Concatenate Column wise
    df['Num_Documents'] = topic_counts
    df['Perc_Documents'] = topic_contribution

    df.reset_index(inplace=True)

    return df


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


def get_all_topics(model: LdaModel):
    results = []
    for topic in model.show_topics(-1, formatted=False):
        results.append([topic[0], ", ".join([word[0] for word in topic[1]])])
    return pd.DataFrame(results, columns=['Topic_Id', 'Topic_Keywords'])
