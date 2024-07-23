import os
import logging
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
import string
import json
sentence_tokenizer = PunktSentenceTokenizer()

def span_tokenizer(txt):
    tokens=nltk.word_tokenize(txt)
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset)
        yield offset, offset+len(token)
        offset += len(token)

def takeID(elem):
    return int(elem.split('\t')[0][1:])

def order_ann(annotations):
    entity_anns = [x for x in annotations if x[0]=='T']
    relation_anns = [x for x in annotations if x[0]=='R']
    entity_anns.sort(key=takeID)
    relation_anns.sort(key=takeID)
    return entity_anns, relation_anns

def find_interval_index(ind, interval_list):
    '''
    ind: [0, 10]
    interval list: a non-overlapping list of interval [[0,1], [2,3], [4,10]]
    '''
    word_start = int(ind[0])
    word_end = int(ind[1])
    for i, j in enumerate(interval_list):
        sentence_start = j[0]
        sentence_end = j[1]
        if word_start >= sentence_start and word_end <= sentence_end:
            return i
    return None


def find_character_index(ind, interval_list):
    '''
    ind: 13
    interval list: a non-overlapping list of interval [[0,1], [2,3], [4,10]]
    '''
    for i, j in enumerate(interval_list):
        word_start = j[0]
        word_end = j[1]
        if ind >= word_start and ind <= word_end:
            return i
    return None

def find_span_index(entity_span, sent_id, words_offset_):
    '''
    find word index
    '''
    entity_start = int(entity_span[0])
    entity_end = int(entity_span[1])
    words_offset = words_offset_[sent_id]
    start_id = find_character_index(entity_start, words_offset)
    end_id = find_character_index(entity_end, words_offset)
    return start_id, end_id+1

def find_entity_id(enn, entity_dict, sent_id):
    '''
    enn: T1
    '''
    enns = [enn for enn in entity_dict.keys() if entity_dict[enn]['sent_id']==sent_id]
    enns.sort(key=takeID)
    id = enns.index(enn)
    return id



def process_entity_span(x):
    left = 0
    right = 0
    while x[0] in string.punctuation:
        x = x[1:]
        left = left + 1
    while x[-1] in string.punctuation:
        x = x[:-1]
        right = right + 1
    return left, right, x



class Annotation:
    def __init__(self, root_path, pmid):
        '''
        '''
        self.root_path = root_path
        self.pmid = pmid
        self.txt_f = os.path.join(self.root_path, pmid + '.txt')
        self.ann_f = os.path.join(self.root_path, pmid + '.ann')

    def check_file_exists(self):
        if not os.path.exists(self.txt_f):
            logging.warning('No {}.txt file in {}.'.format(self.pmid, self.root_path))
            return False
        if not os.path.exists(self.ann_f):
            logging.warning('No {}.ann file in {}.'.format(self.pmid, self.root_path))
            return False
        return True

    def read_data(self):
        if self.check_file_exists():
            annotation = open(self.ann_f, 'r').readlines()
            text = open(self.txt_f, 'r').readlines()[0]
            return annotation, text
        else:
            logging.error('Read data failed!')

    def correct_entity_ann(self, entity_ann, text):
        entity_dict = dict()
        for e_ann in entity_ann:
            entity_id = e_ann.split('\t')[0]
            entity_type = e_ann.split('\t')[1].split(' ')[0]
            entity_start = int(e_ann.split(' ')[1])
            entity_end = int(e_ann.split('\t')[1].split(' ')[2])
            entity_span = text[entity_start:entity_end]
            left, right, entity_span = process_entity_span(entity_span)
            entity_start = entity_start + left
            entity_end = entity_end - right
            if entity_type not in ['species', 'pathway', 'imaging']:
                entity_dict[entity_id] = dict({'entity_id': entity_id, 'type': entity_type, 'entity_start': entity_start, 'entity_end': entity_end, 'span': entity_span})
        return entity_dict

    def relation_ann(self, relation_ann):
        relation_dict = dict()
        for r_ann in relation_ann:
            relation_id = r_ann.split('\t')[0]
            relation_type = r_ann.split('\t')[1].split(' ')[0]
            if relation_type in 'hyponym_of' or relation_type in 'Alias':
                continue
            head_id = r_ann.split('Arg1:')[1].split(' ')[0]
            tail_id = r_ann.split('Arg2:')[1].split('\t')[0]
            relation_dict[relation_id] = dict({'type': relation_type, 'head_id': head_id, 'tail_id': tail_id})
        return relation_dict

    def sents_ann(self, relation_dict, entity_dict, sents_offset_, text):
        words_offset_ = []
        sents = []
        for sent_offset in sents_offset_:
            sent_start = sent_offset[0]
            sent_end = sent_offset[1]
            sent = text[sent_start:sent_end]
            words_offset_generator = span_tokenizer(sent)
            words_offset = [word_offset for word_offset in words_offset_generator]
            words_offset_.append([[offset[0] + sent_start, offset[1] + sent_start] for offset in words_offset])
            word_tokens = [text[(offset[0] + sent_start):(offset[1] + sent_start)] for offset in words_offset]
            sents.append(dict({'tokens': word_tokens, 'entities': [], 'relations': []}))
        if len(entity_dict) != 0:
            for enn in entity_dict.keys():
                sent_id = find_interval_index([int(entity_dict[enn]['entity_start']), int(entity_dict[enn]['entity_end'])], sents_offset_)
                entity_dict[enn]['sent_id'] = sent_id
                start, end = find_span_index([int(entity_dict[enn]['entity_start']), int(entity_dict[enn]['entity_end'])], sent_id, words_offset_)
                entity_dict[enn]['start'] = start
                entity_dict[enn]['end'] = end
            for i in range(len(sents_offset_)):
                offset = 0
                for enn in entity_dict.keys():
                    if entity_dict[enn]['sent_id'] == i:
                        entity_dict[enn]['entity_id_in_sentence'] = offset
                        offset = offset + 1
                        sents[i]['entities'].append(entity_dict[enn])
        if len(relation_dict) != 0:
            for r_ann in relation_dict.keys():
                e1_sent_id = entity_dict[relation_dict[r_ann]['head_id']]['sent_id']
                e2_sent_id = entity_dict[relation_dict[r_ann]['tail_id']]['sent_id']
                if e1_sent_id == e2_sent_id:
                    relation_dict[r_ann]['head'] = entity_dict[relation_dict[r_ann]['head_id']]['entity_id_in_sentence']
                    relation_dict[r_ann]['tail'] = entity_dict[relation_dict[r_ann]['tail_id']]['entity_id_in_sentence']
                    sents[e1_sent_id]['relations'].append(relation_dict[r_ann])
        # print(list(enumerate(sents)))
        return sents

    def obtain_annotations(self):
        annotation, text = self.read_data()
        entity_ann, relation_ann = order_ann(annotation)
        entity_dict = self.correct_entity_ann(entity_ann, text)
        relation_dict = self.relation_ann(relation_ann)
        sents_offset_generator = sentence_tokenizer.span_tokenize(text)
        # print(list(enumerate(sents_offset_generator)))
        sents_offset_ = [sent_offset for sent_offset in sents_offset_generator]
        sents = self.sents_ann(relation_dict, entity_dict, sents_offset_, text)
        return sents

    def to_json(self):
        name = self.pmid
        sents = self.obtain_annotations()
        for i, sent in enumerate(sents):
            sents[i]['orig_id'] = str(name) + '_' + str(i)
        with open(os.path.join(self.root_path, str(name) + '.json'), 'w') as fout:
            json.dump(sents, fout)

