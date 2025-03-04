import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import faiss
import gc
from tqdm import tqdm
import json
import os
import spacy
from spacy.tokens import Span
import torch
import heapq
import scispacy
from scispacy.linking import EntityLinker


class EntityLinker:
    def __init__(self, config):
        """
        Initialize the entity linker
        Args:
            config (dict): Configuration dictionary containing:
                - embedding_model_path: Path to SapBERT model
                - database_embedding_path: Base path for database embedding vectors
                - unique_entity_list_path: Path to unique entity list
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config['embedding_model_path'])
        self.model = BertModel.from_pretrained(
            config['embedding_model_path'],
            return_dict=False,
            output_hidden_states=True
        ).to(self.device)

        # Load entity list and terminology data
        self._load_data()

        # Initialize FAISS indices
        self._initialize_indices()

        # Initialize UMLS linker
        self._initialize_umls_linker()

    def _load_data(self):
        """Load required data files"""
        base_path = self.config['database_embedding_path']

        # Load unique entity list
        with open(self.config['unique_entity_list_path'], "r", encoding='utf-8') as f:
            self.unique_entity_texts = json.load(f)

        # Load terms for each ontology
        self.ontology_terms = {}
        for ontology in ['go', 'hpo', 'mondo', 'uberon']:
            path = os.path.join(base_path, f'{ontology}_terms.json')
            with open(path, 'r') as f:
                self.ontology_terms[ontology.upper()] = json.load(f)

    def _initialize_indices(self):
        """Initialize FAISS indices"""
        self.indices = {}
        for ontology, terms in self.ontology_terms.items():
            self.indices[ontology] = self.create_faiss_index(terms, use_gpu=torch.cuda.is_available())

    def _initialize_umls_linker(self):
        """Initialize UMLS linker"""
        self.nlp = spacy.load("en_core_sci_sm")
        self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        self.linker = self.nlp.get_pipe("scispacy_linker")

    def get_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            # 添加 return_dict=True 来确保输出包含 last_hidden_state
            outputs = self.model(**inputs, return_dict=True)
        return outputs.last_hidden_state.mean(dim=1).cpu().squeeze().numpy()

    def get_best_match(self, entity_embeddings, index, terms_with_embeddings):
        if entity_embeddings.ndim == 1:
            entity_embeddings = entity_embeddings[np.newaxis, :]
        D, I = index.search(entity_embeddings, 1)
        best_idx = I[0][0]
        best_term = terms_with_embeddings[best_idx]  #
        return best_term

    def cosine_similarity(self, vec1, vec2):
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def create_faiss_index(self, terms_with_embeddings, use_gpu):
        embedding_size = len(terms_with_embeddings[0]['sapbert_embedding'])
        embedding_matrix = np.array([term['sapbert_embedding'] for term in terms_with_embeddings]).astype('float32')
        faiss.normalize_L2(embedding_matrix)

        if use_gpu and hasattr(faiss, 'StandardGpuResources'):  # 检查是否支持GPU
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(embedding_size)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(embedding_matrix)
            return gpu_index
        else:
            index = faiss.IndexFlatL2(embedding_size)
            index.add(embedding_matrix)
            return index

    def process_entities(self, entities, processed_entity, index, terms_with_embeddings, ontology_name, id_key,
                         term_name_key):
        """
        Process entities and update the `processed_entity` dictionary.

        Args:
            entities (list): List of entities to be processed.
            processed_entity (dict): Dictionary to store processed entity information.
            index (dict): Index data for matching terms.
            terms_with_embeddings (dict): Embedding data for ontology terms.
            ontology_name (str): Name of the ontology (e.g., HPO, GO, etc.).
            id_key (str): Key to retrieve the ontology term ID.
            term_name_key (str): Key to retrieve the ontology term name.
        """
        for entity in tqdm(entities, desc=f'Processing entities for {ontology_name}'):
            entity_embeddings = self.get_bert_embedding(entity).reshape(1, -1)  # 确保为二维
            best_term = self.get_best_match(entity_embeddings, index, terms_with_embeddings)
            best_term_embedding = np.array(best_term['sapbert_embedding']).flatten()  # 确保为一维数组
            similarity = self.cosine_similarity(entity_embeddings, best_term_embedding)

            if similarity > 0.88:
                entity_info = {
                    'entity_id': best_term[id_key],
                    'entity_ontology': ontology_name,
                    'entity_onto_term': best_term[term_name_key],
                    'similarity': similarity
                }
                if entity in processed_entity:
                    if processed_entity[entity]['similarity'] < similarity:
                        processed_entity[entity] = entity_info
                else:
                    processed_entity[entity] = entity_info

    def link_entities(self):
        """
        Execute entity linking process
        Returns:
            dict: Processed entity mapping results
        """
        processed_entity = {}

        # Process ontology matching
        ontologies = [
            {
                "name": name,
                "index": self.indices[name],
                "terms": self.ontology_terms[name],
                "id_key": "id",
                "term_name_key": "name"
            }
            for name in ["HPO", "GO", "UBERON", "MONDO"]
        ]

        for ontology in ontologies:
            self.process_entities(
                self.unique_entity_texts,
                processed_entity,
                ontology["index"],
                ontology["terms"],
                ontology["name"],
                ontology["id_key"],
                ontology["term_name_key"]
            )

        # Process UMLS matching
        self._process_umls_entities(processed_entity)

        return processed_entity

    def _process_umls_entities(self, processed_entity):
        """Process UMLS entity linking"""
        for text in tqdm(self.unique_entity_texts, desc='Processing unique entities'):
            doc = self.nlp(text)
            self.linker(doc)
            entity_embeddings = self.get_bert_embedding(text)  # 获取标准实体的embeddings
            entity_matches = []
            entity_info = {}

            if doc.ents:
                for ent in doc.ents:
                    if ent._.kb_ents:
                        for umls_ent in ent._.kb_ents:
                            linked_entity = self.nlp.get_pipe("scispacy_linker").kb.cui_to_entity[umls_ent[0]]
                            for name in [linked_entity.canonical_name] + linked_entity.aliases:
                                umls_embedding = self.get_bert_embedding(name)
                                similarity = self.cosine_similarity(entity_embeddings, umls_embedding)
                                if similarity > 0.85:
                                    entity_matches.append(
                                        {'umls_name': name, 'umls_cui': umls_ent[0], 'similarity': similarity})

            doc = self.nlp(text)
            span = Span(doc, start=0, end=len(doc), label="CHEMICAL")
            doc.ents = [span]
            self.linker(doc)
            if doc.ents:
                for ent in doc.ents:
                    if ent._.kb_ents:
                        for umls_ent in ent._.kb_ents:
                            linked_entity = self.nlp.get_pipe("scispacy_linker").kb.cui_to_entity[umls_ent[0]]
                            for name in [linked_entity.canonical_name] + linked_entity.aliases:
                                umls_embedding = self.get_bert_embedding(name)
                                similarity = self.cosine_similarity(entity_embeddings, umls_embedding)
                                if similarity > 0.85:
                                    entity_matches.append(
                                        {'umls_name': name, 'umls_cui': umls_ent[0], 'similarity': similarity})
                # Select the best match (highest similarity)
            top_matches = heapq.nlargest(1, entity_matches, key=lambda x: x['similarity'])
            if top_matches:
                top_match = top_matches[0]
                if text in processed_entity:
                    if top_match['similarity'] > processed_entity[text]['similarity']:
                        entity_info = {
                            'entity_id': top_match['umls_cui'],
                            'entity_ontology': 'UMLS',
                            'entity_onto_term': top_match['umls_name'],
                            'similarity': top_match['similarity']
                        }
                        processed_entity[text] = entity_info
                else:
                    entity_info = {
                        'entity_id': top_match['umls_cui'],
                        'entity_ontology': 'UMLS',
                        'entity_onto_term': top_match['umls_name'],
                        'similarity': top_match['similarity']
                    }
                    processed_entity[text] = entity_info


# Usage example
if __name__ == "__main__":
    config = {
        "embedding_model_path": "/Users/gs/Desktop/Paper2/Sapbert",
        "database_embedding_path": "Linked_ontology/",
        "unique_entity_list_path": "Linked_ontology/entity_sample.json"
    }

    linker = EntityLinker(config)
    results = linker.link_entities()
