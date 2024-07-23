from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch import nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel

from spert import sampling
from spert import util
import sys

###################


import numpy as np
#from torch import nn
#import torch.nn.functional as F
from torch.autograd import Variable

import torch.nn.functional as F

###################




#h: vector from BERT of the input sentence. 
#If there are E tokens in a sentence and each token is transformed to a vector of size S (= emb_size = 768), then shape of h =  BatchSize x E x S.
#x: numeric ids of the encodings obtained after tokenization for whole batch; shape = BatchSize x E. Each slot is a token id (int).
#token: numeric id of the desired token, e.g., 'CLS'
#Observe 'h' contains the BERT embedding of each token in the sentence. So 
#jump to the index of h[token] 
#h.shape= (10, 40, 768) => 10 sentences each 40 tokens long. [(Sentence embedding)+]
#x.shape= (10, 40) => 10 sentences each 40 tokens long. [(Sentence encoding)+]

def get_token(h: torch.tensor, x: torch.tensor, token: int):  #'token' is the numeric id, e.g., 'CLS'
    """ Get specific token embedding (e.g. [CLS]) """

    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)  #There will be emb_size columns. #rows = #tokens
                                    #-1 means actual size will be inferred.
                                    #So h.shape=(10*40, 768)
    flat = x.contiguous().view(-1)  #flattens the encoding, i.e., 
                                    #flat = vector of 10*40 token ids

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :] #Pick those embeddings that corr. to token
                                        #token_h.shape = (10, 768) if each sentence
                                        #contains the token only once.
                                        #If token occurs K times across all sentences,
                                        #token_h.shape = (K, 768)
    #print("&&&&&&&  token_h = ", token_h, ", &&&&&&&&&& shape = ", token_h.shape)
    return token_h


class SpERT(BertPreTrainedModel):
    """ Span-based model to jointly extract entities and relations """

    VERSION = '1.1'

    def __init__(self, config: BertConfig, cls_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, max_pairs: int = 100,
                 #DKS
                 use_syntax: bool = False, 
                 use_pos: bool = False, 
                 use_entity_clf: str = "none"
                 ):
        super(SpERT, self).__init__(config)


        self._use_syntax = use_syntax
        self._use_pos = use_pos
        self._pos_embedding = 25 #config.hidden_size #POS embedding size must match BERT embedding size
        self._use_entity_clf = use_entity_clf
        
        # Syntax BERT model
        self.bert = BertModel(config)

        # layers
        if (self._use_pos == False):
            entc_in_dim = (config.hidden_size) * 2 + size_embedding  #CLS + ENT + SIZE = 2H + SIZE
        else:
            #CLS + ENT + SIZE + POS =  H*2 + SIZE + POS
            entc_in_dim = (config.hidden_size)*2  + size_embedding   + self._pos_embedding
           
        self.entity_classifier = nn.Linear( entc_in_dim, entity_types )
        #self.entity_classifier1 = nn.Linear( entc_in_dim, 100 )
        #self.entity_classifier2 = nn.Linear( 100, entity_types )

        #(ENT + SIZE)*2 + SPAN = 3H + SIZE
        relc_in_dim =  (config.hidden_size ) * 3 + size_embedding * 2
        if (self._use_entity_clf != "none"):
            relc_in_dim +=  entity_types * 2
        if (self._use_pos):
            relc_in_dim +=  self._pos_embedding * 3
        #print("########### REL CLASSIFIER INPUT DIM = ", relc_in_dim)
            #relc_in_dim = (config.hidden_size ) * 3 + size_embedding * 2
            #relc_in_dim = (config.hidden_size) * 3 + size_embedding * 2
        #else:
            #(ENT + POS + SIZE + TYPE) *2 + CTX = (H + POS + SIZE)*2 + H = 3 * H + 2 * POS + 2 * SIZE
            #relc_in_dim = (config.hidden_size ) * 3 + size_embedding * 2 + entity_types * 2
            
        #print("####  relc_in_dim = ", relc_in_dim)

        self.rel_classifier = nn.Linear(relc_in_dim, relation_types)

           
        self.size_embeddings = nn.Embedding(100, size_embedding)
        self.pos_embeddings = nn.Embedding(52, self._pos_embedding, padding_idx=0)
        self.dropout = nn.Dropout(prop_drop)

        self._cls_token = cls_token
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs

        # weight initialization
        self.init_weights()

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    #encodings, h, entity_masks, size_embeddings, pooled_output, hlarge1
    def _run_entity_classifier (self, x: torch.tensor):
        y = self.entity_classifier( x )

        return y
    
    def _run_rel_classifier (self, x: torch.tensor):
        y = self.rel_classifier( x )
        return y
        
    def _classify_entities(self, encodings, h, entity_masks, size_embeddings, 
                           pos=None, hlarge=None):
        # max pool entity candidate spans
        if (hlarge == None):
            hlarge = h
        #print("### _classify_entities  hlarge.shape = ", hlarge.shape)
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)  #torch.Size([10, 105, 40, 1])

        entity_spans_pool = m + hlarge.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1) #torch.Size([10, 105, 40, 768])

        entity_spans_pool = entity_spans_pool.max(dim=2)[0]
        # print("#### entity_spans_pool maxooled .shape = ", entity_spans_pool.shape)

        entity_ctx = get_token(h, encodings, self._cls_token)
        
        entity_repr = torch.cat([entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                                 entity_spans_pool, size_embeddings], dim=2)
        entity_repr = self.dropout(entity_repr)


        entity_clf = self._run_entity_classifier(entity_repr)

        return entity_clf, entity_spans_pool
    
    def _classify_relations(self, entity_spans, size_embeddings, 
                            relations, rel_masks, h, chunk_start,
                            entity_clf = None, hlarge1=None):
        batch_size = relations.shape[0]

        # create chunks if necessary
        # print("#### relations.shape = ", relations.shape)
        if (hlarge1 == None):
            hlarge1 = h
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            hlarge1 = hlarge1[:, :relations.shape[1], :]


        
        entity_pairs = util.batch_index(entity_spans, relations)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        # get corresponding size embeddings
        size_pair_embeddings = util.batch_index(size_embeddings, relations)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # relation context (context between entity candidate pair)
        # mask non entity candidate tokens
        m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)
        #rel_ctx = m + h
        #print("#### m.shape = ", m.shape)
        rel_ctx =  m + hlarge1
        # max pooling
        rel_ctx = rel_ctx.max(dim=2)[0]
        # set the context vector of neighboring or adjacent entity candidates to zero
        rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0

        # create relation candidate representations including context, max pooled entity candidate pairs
        # and corresponding size embeddings
        rel_repr = torch.cat([rel_ctx, entity_pairs, size_pair_embeddings], dim=2)

        
        #print("###### Rel rep.shape = ", rel_repr.shape, ", entity_clf.shape = ", entity_clf.shape)
        
        #DKS
        if (entity_clf != None):
         if (self._use_entity_clf == "logits" or self._use_entity_clf == "softmax" 
                                              or self._use_entity_clf == "onehot"):
            if (self._use_entity_clf == "softmax"):

                entity_clf = torch.softmax(entity_clf, dim=-1)

                
            elif (self._use_entity_clf == "onehot"):
                #print("########## entity_clf shape = ", entity_clf.shape)
                dim = entity_clf.shape[-1]
                entity_clf = torch.argmax(entity_clf, dim=-1)
                entity_clf = torch.nn.functional.one_hot(entity_clf, dim) # get entity type (including none)
            #Following lines execute if self._use_entity_clf is one of "logits", "softmax", "onehot"   
            entity_clf_pairs =  util.batch_index(entity_clf, relations)
            entity_clf_pairs =  entity_clf_pairs.view(batch_size, entity_clf_pairs.shape[1], -1)
            rel_repr = torch.cat([ rel_repr, entity_clf_pairs], dim=2)
        
        
        rel_repr = self.dropout(rel_repr)
        chunk_rel_logits = self._run_rel_classifier(rel_repr)
        return chunk_rel_logits



    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()  # get entity type (including none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # get spans classified as entities
            non_zero_indices = (entity_logits_max[i] != 0).nonzero(as_tuple=False).view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()

            # create relations and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.rel_classifier.weight.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks


    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)


##############################################################


class SynSpERTConfig(BertConfig):
    def __init__(self, **kwargs):
        super(SynSpERTConfig, self).__init__(**kwargs)




class SynSpERT(SpERT):
    config_class = SynSpERTConfig
    VERSION = '1.0'

    def __init__(self, config: SynSpERTConfig, cls_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, max_pairs: int = 100, 
                 #DKS: added for SynSpert
                 use_syntax: bool = False,
                 use_pos: bool = False,  
                 use_entity_clf: str = "none"                 
                 ):
        super(SynSpERT, self).__init__(config, cls_token, relation_types, entity_types,
                                       size_embedding, prop_drop, freeze_transformer, max_pairs,
                                       #DKS: added
                                       use_syntax,
                                       use_pos,
                                       use_entity_clf)


        self.config = config

        self.init_weights()


    
    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, relations: torch.tensor, rel_masks: torch.tensor, 
                       #Extra params follow
                       dephead: torch.tensor, deplabel: torch.tensor, pos: torch.tensor ):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        
        seq_len = torch.count_nonzero(context_masks, dim=1)

        batch_size = encodings.shape[0]
        token_len = h.shape[1]
        
        #print(seq_len,dephead,deplabel)

        hlarge1 = None
        #Add pos embeddings to the tokens
        if (self._use_pos):
           pos_em = self.pos_embeddings(pos).to(self.rel_classifier.weight.device)
           hlarge1 = h
           hlarge1 = torch.cat((hlarge1, pos_em), -1)

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, 
                                        entity_masks, size_embeddings, pos, hlarge1)

        # classify relations
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        hlarge2 = None
        if (self._use_pos):
           hlarge2 = hlarge1.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i, 
                                                        entity_clf, hlarge2)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits
        ############ [dks]
        return entity_clf, rel_clf

    def _forward_eval(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                      entity_sizes: torch.tensor, entity_spans: torch.tensor, entity_sample_masks: torch.tensor, 
                      dephead: torch.tensor, deplabel: torch.tensor, pos: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        h_pooler = self.bert(input_ids=encodings, attention_mask=context_masks)['pooler_output']

        batch_size = encodings.shape[0]
        ctx_size = context_masks.shape[-1]
        
        seq_len = torch.count_nonzero(context_masks, dim=1)      
        #print(seq_len,dephead,deplabel)

        hlarge1 = None
        if (self._use_pos):
           pos_em = self.pos_embeddings(pos).to(self.rel_classifier.weight.device)
           #print("### h.shape = ", h.shape, ", pos.shape = ", pos_em.shape)
           hlarge1 = h
           hlarge1 = torch.cat((hlarge1, pos_em), -1)
          
    
        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, 
                                entity_masks, size_embeddings, pos, hlarge1)

        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        relations, rel_masks, rel_sample_masks = self._filter_spans(entity_clf, entity_spans,
                                                                    entity_sample_masks, 
                                                                    ctx_size)

        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        hlarge2 = None
        if (self._use_pos):
           hlarge2 = hlarge1.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)

        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, 
                                                        h_large, i, 
                                                        entity_clf, hlarge2)
            # apply sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        rel_clf = rel_clf * rel_sample_masks  # mask

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, rel_clf, relations, h_pooler

# Model access

_MODELS = {
    'spert': SpERT,
    'syn_spert': SynSpERT,
}


def get_model(name):
    return _MODELS[name]