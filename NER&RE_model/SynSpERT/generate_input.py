from glob import glob
import json
from ann2json_22 import Annotation
import os

epoch = 3  # Number of epochs for training
date = '0217_1'  # Date or version identifier for the dataset

# Paths to data files
annotated_file_path = r"your/annotated/data/path/"  # Set your annotated data path here
train_file_path = r'your/train/file/path/md_train_KG_' + date + '.json'  # Set your train file path here
test_file_path = r'your/test/file/path/md_test_KG_' + date + '.json'  # Set your test file path here
all_file_path = r'your/all/file/path/md_KG_all_' + date + '.json'  # Set your all data file path here


def replace_quotes_in_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)

            with open(filepath, 'r', encoding='utf8') as file:
                lines = file.readlines()


            lines = [line.replace('"', "'") for line in lines]

            with open(filepath, 'w', encoding='utf8') as file:
                file.writelines(lines)

replace_quotes_in_files(annotated_file_path)

files = glob(annotated_file_path+'*')
ids = [f.split('\\')[-1].split('.')[0] for f in files]

for id in ids:
    ann = Annotation(annotated_file_path, id)
    ann.to_json()
files = sorted(glob(annotated_file_path+"*.json"))


data = list()
for f in files:
    data_ = json.load(open(f, 'r'))
    for d in data_:
        data.append(d)       


from sklearn.model_selection import train_test_split

import collections
ys = [x['relations'][0]['type'] if len(x['relations'])>0 else 'no_relation' for x in data]
counter=collections.Counter(ys)
print(counter)

xs = [x['relations'][0]['type'] if len(x['relations'])>0 else 'no_relation' for x in data]
counter=collections.Counter(ys)

token = []
for x in data:
    for y in x['tokens']:
        token.append(y)


tokens = [token.append(y) for y in x['tokens'] for x in data ]

print(counter)
x_train, x_test, y_train, y_test = train_test_split(data, ys,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=ys)

entities = []
for d in data:
    if len(d)!=0:
        entities = entities+d['entities']

#
with open(train_file_path, 'w') as f:
    json.dump(x_train, f)


with open(test_file_path, 'w') as f:
    json.dump(x_test, f)

with open(all_file_path, 'w') as f:
    json.dump(data, f)



