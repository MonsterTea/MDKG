import json
import numpy
import csv
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import pandas as pd
import faiss
import torch
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans, MiniBatchKMeans
import copy

# embedding1 = open('D:\\pycharm\\spert.pl_prediction\\InputsAndOutputs\\data\\datasets\\h_pooler\\h_pooler_123.txt', 'r').read()
# embedding = []
#
# for emd in embedding1.split(')\n'):
#     emd = emd.replace('\n', '')
#     emd = emd.replace(' ', '')
#     emd = emd.replace('[', '')
#     emd = emd.replace(']', '')
#     emd = emd.replace('tensor(', '')
#     embedding.append(emd)
#
# embedding.pop()
# # print(len(embedding))
# # print(embedding)
# num_embedding = []
# for i in range(len(embedding)):
#     sentence_embedding = []
#     for ele in embedding[i].split(','):
#         ele = float(ele)
#         sentence_embedding.append(ele)
#     num_embedding.append(sentence_embedding)
# #         # num_embedding.extend(float(ele))
#
# embedding_tensor = torch.Tensor(num_embedding)
# print(embedding_tensor.shape)
# entropy1 = open(r'D:\pycharm\spert.pl_prediction\InputsAndOutputs\data\datasets\treatmentdata\entropy_12345.txt','r').readlines()
# # entropy = np.numpy(entropy)
# entropy = []
# for i in entropy1:
#     i = float(i.replace('\n', ''))
#     entropy.append(i)
Names = 'all_sentences_5_agu_new'
root_path = 'F:/spert.pl_prediction/InputsAndOutputs/data/datasets/treatmentdata/'
n_sample = 39298
topK = 20
sample_per_group = 10
beta=0.1
weight=True
ncentroids= 20
gamma = 1


weights = np.ones(n_sample,)
entropy_relation= torch.load(root_path+'entropy_relation'+ Names + '.pt').cpu().numpy().reshape(n_sample,)
entropy_entities= torch.load(root_path+'entropy_entities'+ Names + '.pt').cpu().numpy().reshape(n_sample,)
entropy = 0.3*entropy_relation + gamma*entropy_entities
torch.save(entropy, root_path+'total_entropy'+ Names + '.pt')
unlabeled_pred = torch.load(root_path +'labelprediction'+ Names + '.pt').cpu().numpy()
unlabeled_feat = torch.load(root_path + 'pooler_output'+ Names + '.pt').cpu().numpy()
d = unlabeled_feat.shape[-1]

if weight:  # use weighted K-Means Clustering
    kmeans = MiniBatchKMeans(n_clusters=ncentroids, random_state=0, batch_size=256, n_init=3, max_iter=100)
    kmeans.fit(unlabeled_feat, sample_weight=copy.deepcopy(entropy))
    index = faiss.IndexFlatL2(d)
    index.add(kmeans.cluster_centers_.astype('float32'))
    D, I = index.search(unlabeled_feat.astype('float32'), 1)
else:
    kmeans = faiss.Clustering(int(d), ncentroids)
    index = faiss.IndexFlatL2(d)
    kmeans.train(unlabeled_feat.numpy(), index)
    centroid = faiss.vector_to_array(kmeans.centroids).reshape(ncentroids, -1)
    index.add(centroid.astype('float32'))
    D, I = index.search(unlabeled_feat, 1)
I = I.flatten()
scores = []
indexes = []
class_number = np.ones(n_sample,)
for i in range(ncentroids):
    idx = (I == i)
    class_number[idx] = i
    weights[idx] = entropy[idx]/np.sum(entropy[idx])
    # calculate the mean entropy of samples
    if True in idx:
        mean_entropy = np.mean(entropy[idx])
    else:
        mean_entropy = 0
    class_sum = torch.sum(torch.from_numpy(unlabeled_pred[idx]), dim=0, keepdim=False).numpy()
    if np.sum(class_sum) == 0:
        class_frequency = np.zeros(9,)
    else:
        class_frequency = class_sum / np.sum(class_sum)

    class_entropy = np.sum(abs(class_frequency * np.log(class_frequency + 1e-12)))
    value = mean_entropy + beta * class_entropy
    scores.append(value)
    sorted_idx = np.argsort(entropy[idx])
    idxs = np.arange(len(I))[idx][sorted_idx]
    indexes.append(list(idxs))
sample_idx = []
remains = n_sample
weighted_embedding = np.ones((n_sample,768))
for i in range(n_sample):
    weighted_embedding[i,:] = weights[i]*unlabeled_feat[i,:]
SaveNames = 'all_sentences_5_agu_new_round_1'
torch.save(weighted_embedding, root_path + 'weighted_embedding'+SaveNames+'.pt')
torch.save(class_number, root_path + 'class_number'+SaveNames+'.pt')
# sample_idx2 = torch.load(root_path + 'final_idx'+SaveNames+'.pt')
for i in np.argsort(scores)[::-1][0:topK]:
    sample_idx += indexes[i][-min(remains, sample_per_group, len(indexes[i])):]
    # print(len(indexes[i]))
    indexes[i] = indexes[i][:-min(remains, sample_per_group, len(indexes[i]))]
    remains -= len(indexes[i][-min(remains, sample_per_group, len(indexes[i])):])
    if remains <= 0:
        break

torch.save(sample_idx, root_path + 'sample_idx_'+ str(ncentroids) + SaveNames +'.pt')


# with open(root_path + 'idsample.json','r') as fr: #同上
# 	json_file2 = json.loads(fr.read())
#
# json_file1 = []
# json_file_not = []
# for i in range(len(json_file2)):
#     if json_file2[i]['orig_id'] in sample_idx:
#         json_file1.append(json_file2[i])
#     else:
#         json_file_not.append(json_file2[i])
#
# # for y in indexes:
# #     print(y)
# #     sample_idx += y
#
# complete_sample_ids = []
# for i in sample_idx:
#     i = '123_' + str(i)
#     complete_sample_ids.append(i)
#
#
# with open('D:\\pycharm\\spert.pl_prediction\\InputsAndOutputs\\data\\datasets\\treatmentdata\\123_new.json', 'r') as fr:
# 	json_file = json.loads(fr.read())
#
# selected_tokens = []
# for j in range(len(json_file)):
#     if json_file[j]['orig_id'] in complete_sample_ids:
#         selected_tokens.append(json_file[j]['tokens'])
#
# selected_sents = []
# for i in range(len(selected_tokens)):
#     tokens = ' '.join(selected_tokens[i])
#     tokens = tokens.replace(' ,', ',')
#     tokens = tokens.replace(' )', ')')
#     tokens = tokens.replace(' :', ':')
#     tokens = tokens.replace(' %', '%')
#     tokens = tokens.replace(' ;', ';')
#     tokens = tokens.replace(' .', '.')
#     tokens = tokens.replace(' ]', ']')
#     tokens = tokens.replace('( ', '(')
#     tokens = tokens.replace('[ ', '[')
#     tokens = tokens.replace("' ", "'")
#     tokens = tokens.replace(" '", "'")
#     selected_sents.append(tokens)
# txt = ' '.join(selected_sents)
# with open('D:\\pycharm\\spert.pl_prediction\\InputsAndOutputs\\data\\datasets\\treatmentdata\\selected_sents.txt', 'w') as f:
#     f.write(txt)
#
# with open('D:\\pycharm\\spert.pl_prediction\\InputsAndOutputs\\data\\datasets\\treatmentdata\\new_agu.json', 'w') as f:
#     json.dump(json_file1, f)
#
# with open('D:\\pycharm\\spert.pl_prediction\\InputsAndOutputs\\data\\datasets\\treatmentdata\\new_agu_not.json', 'w') as f:
#     json.dump(json_file_not, f)


# print(len(sample_idx))
# d = int(embedding_tensor.shape[1])
# ncentroids = 5
# index = faiss.IndexFlatL2(d)
# print(index.is_trained)
# index.add(embedding_tensor.numpy())
# print(index.ntotal)

# kmeans = faiss.Clustering(d, ncentroids)
# index = faiss.IndexFlatL2(d)  # build the index
# kmeans.train(embedding_tensor.numpy(), index)
# centroid = faiss.vector_to_array(kmeans.centroids).reshape(ncentroids, -1)
# index.add(centroid)   # add vectors to the index
# D, I = index.search(embedding_tensor.numpy(), 4)   # we want to see 4 nearest neighbors
# print(I[:5])  # neighbors of the 5 first queries 的id
# print(D[-5:])  # neighbors of the 5 last queries


# ncentroids = 10
# niter = 20
# verbose = True
# d = embedding_tensor.shape[1]
# print(d)
# kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
# kmeans.train(embedding_tensor.numpy())



