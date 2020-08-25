# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import re

import os
import glob


# %%
#target_dir = "C:\\Users\\mayuan\\Desktop\\BMRCproject\\evaluation\\target\\"
target_df = pd.read_csv(r"C:\Users\mayuan\Desktop\BMRCproject\evaluation\target\Text-mining-word-list-test-200823.csv",encoding='ISO-8859-1', header=None)


# %%
target_list = target_df[0].tolist()
print(target_list)


# %%
source_df = pd.read_csv(r"C:\Users\mayuan\Desktop\BMRCproject\evaluation\source\cordis-h2020projects.csv",encoding='ISO-8859-1')


# %%
found_source = set()

for index, row in source_df.iterrows():
    row_data = row['title'] + " " + row['objective']
    sent = re.sub(r"[^a-zA-Z0-9]+", ' ', row_data).lower()
    print(sent)
    for word in target_list:
        if re.search(r'\b' + word + r'\b', sent):
            found_source.add(word)

    if index == 10:
        break
    # print(index)
    # if index == 200:
    #     break
# pd.reset_option('display')
print(found_source)


# %%
for index, row in source_df.iterrows():
    row_data = row['title'] + " " + row['objective']
    sent = re.sub(r"[^a-zA-Z0-9]+", ' ', row_data).lower()
    
    for word in target_list:
        if re.search(r'\b' + word + r'\b', sent):
            print(sent)

    if index == 10:
        break


# %%
output_df = pd.DataFrame(columns=source_df.columns)

for index, row in source_df.iterrows():
    row_data = row['title'] + " " + row['objective']
    sent = re.sub(r"[^a-zA-Z0-9]+", ' ', row_data).lower()
    
    for word in target_list:
        #if word in sent:
        if re.search(r'\b' + word + r'\b', sent):
            print(row)
            #output_df = pd.concat([row, output_df], ignore_index=True)
            output_df = output_df.append(row, ignore_index=True)
            break
    
    # if index == 10:
    #     break


# %%
print(output_df)


# %%
output_df.to_excel(r'C:\Users\mayuan\Desktop\BMRCproject\evaluation\output_df.xlsx')


# %%
topics_df = pd.read_csv(r"C:\Users\mayuan\Desktop\BMRCproject\evaluation\source\Topics-300.csv", header=None, encoding='ISO-8859-1')

# print(topics_df)

found_topics = set()

for index, row in topics_df.iterrows():
    sent = re.sub(r"[^a-zA-Z0-9]+", ' ', row[0]).lower()

    for word in target_list:
        if re.search(r'\b' + word + r'\b', sent):
            found_topics.add(word)

# # pd.reset_option('display')
print(found_topics)


# %%
print(len(found_source))
print(len(found_topics))


# %%
text = "Cultural tourism is changing The traditional forms still exist museums art galleries landscapes historical sites festivals but both cultural destinations and the tourists are under transformation Many cultural tourists see themselves neither as seeking culture nor as tourists there is increasing evidence of people seeking to experience culture rather than merely observing it That is agri tourism where visitors want to experience rural life people wanting to visit the actual venues of TV crime thrillers culture being explored by those using themed routes in winery regions or via pilgrimage These trends provide opportunities to both revitalise poorer and rural areas through economic and social development while protecting local cultures and landscapes The project brings an extension of existing policies and the promotion of new approaches The project s aim is to develop a new approach to understanding and addressing cultural tourism and to promote development of disadvantaged areas Based on an Innovation Tool and digital technology the project identifies layers of data and capitalise on existing practice explores emerging forms of cultural tourism identifies opportunities and develop strategies allowing local people to gain local benefit from their precious cultural assets The project uses case studies across 15 European regions consolidates definitions of cultural tourism engages academics and stakeholders in developing policy proposals in practice and posits means of generalising the lessons via an Innovation Tool to assist policy makers at all levels as well as practitioners Positive and negative aspects of cultural tourism exist a balanced development path needs to be sought The project will help to identify themes and areas where intervention at local regional national and European levels may assist in achieving successful developments it will help in managing that balance and offering solutions "

for w in target_list:
    if w in text:
        print(w)


# %%
s = 'This is correct1'
for words in ['This is correct', 'This', 'is', 'correct']:
    if re.search(r'\b' + words + r'\b', s):
        print('{0} found'.format(words))
