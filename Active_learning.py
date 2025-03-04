import json
import numpy as np
import faiss
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
import copy

# Parameters to set
Names = 'Name1'
SaveNames = 'Name2'
root_path = 'your/root/path/'  # Set your root path here
all_file_json_path = 'your/json/file/path.json'  # Set your JSON file path here

n_sample = 40000  # Total number of samples in the dataset
topK = 20  # Number of top categories to select based on their scores
sample_per_group = 10  # Number of samples to select from each top category
beta = 0.1
weight = True
ncentroids = 20
gamma = 0.1

# Load data
weights = np.ones(n_sample,)
entropy_relation = torch.load(root_path + 'entropy_relation' + Names + '.pt', map_location=torch.device('cpu')).reshape(n_sample, )
entropy_entities = torch.load(root_path + 'entropy_entities' + Names + '.pt', map_location=torch.device('cpu')).reshape(n_sample, )
entropy = entropy_relation + gamma * entropy_entities
torch.save(entropy, root_path + 'total_entropy' + Names + '.pt')
unlabeled_pred = torch.load(root_path + 'labelprediction' + Names + '.pt', map_location=torch.device('cpu'))
unlabeled_feat = torch.load(root_path + 'pooler_output' + Names + '.pt', map_location=torch.device('cpu')).cpu().numpy()
d = unlabeled_feat.shape[-1]

# K-Means Clustering
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

# Calculate scores and indexes
class_number = np.ones(n_sample,)
for i in range(ncentroids):
    idx = (I == i)
    class_number[idx] = i
    weights[idx] = entropy[idx] / (np.sum(entropy[idx]) + 1e-12)
    mean_entropy = np.mean(entropy[idx]) if np.any(idx) else 0
    class_sum = torch.sum(torch.from_numpy(unlabeled_pred[idx]), dim=0, keepdim=False).numpy()
    class_frequency = np.zeros(9,) if np.sum(class_sum) == 0 else class_sum / (np.sum(class_sum) + 1e-12)
    class_entropy = np.sum(abs(class_frequency * np.log(class_frequency + 1e-12)))
    value = mean_entropy + beta * class_entropy
    scores.append(value)
    sorted_idx = np.argsort(entropy[idx])
    idxs = np.arange(len(I))[idx][sorted_idx]
    indexes.append(list(idxs))

# Select samples
sample_idx = []
remains = n_sample
weighted_embedding = np.ones((n_sample, 768))
for i in range(n_sample):
    weighted_embedding[i, :] = weights[i] * unlabeled_feat[i, :]
torch.save(weighted_embedding, root_path + 'weighted_embedding' + SaveNames + '.pt')
torch.save(class_number, root_path + 'class_number' + SaveNames + '.pt')
for i in np.argsort(scores)[::-1][:topK]:
    samples_to_take = min(remains, sample_per_group, len(indexes[i]))
    selected_samples = indexes[i][-samples_to_take:]
    sample_idx.extend(selected_samples)
    indexes[i] = indexes[i][:-samples_to_take]
    remains -= len(selected_samples)
    if remains <= 0:
        break

# Read the entire initial JSON
with open(all_file_json_path, 'r') as fr:
    all_json_text = json.loads(fr.read())

# Active Learning sampling sentences
sampling_json_text = []
for i in sample_idx:
    sampling_json_text.append(all_json_text[i])
with open(root_path + 'sampling_json' + SaveNames + '.json', 'w') as fw:
    json.dump(sampling_json_text, fw)
