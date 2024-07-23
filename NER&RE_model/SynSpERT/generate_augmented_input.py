import spacy
import scispacy
import json
from spacy.tokens import Doc
from more_itertools import locate


# pip install scispacy
#pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_sci_sm-0.5.0.tar.gz
#####################################
#### Customized tokenizer        ####
#####################################
# nlp = spacy.load('en_core_web_sm')
nlp = spacy.load("en_core_sci_sm")



def custom_tokenizer(text):
    tokens = text.split(" ")
    return Doc(nlp.vocab, tokens)
    # global tokens_dict
    # if text in tokens_dict:
    #    return Doc(nlp.vocab, tokens_dict[text])
    # else:
    #    VaueError("No tokenization for input text: ", text)

nlp.tokenizer = custom_tokenizer
#####################################


class JsonInputAugmenter():
    def __init__(self):
        date = '0217_1'  # Date or version identifier for the dataset
        basepath = r'your/base/path/'  # Set your base path here
        
        # Paths to input data files
        self.input_dataset_paths = [
            basepath + 'md_train_KG_' + date + '.json',
            basepath + 'md_test_KG_' + date + '.json',
            basepath + 'md_KG_all_' + date + '.json'
        ]
        
        # Paths to output data files
        self.output_dataset_paths = [
            basepath + 'md_train_KG_' + date + '_agu.json',
            basepath + 'md_test_KG_' + date + '_agu.json',
            basepath + 'md_KG_all_' + date + '_agu.json'
        ]

    def augment_docs_in_datasets(self):
        for ipath, opath  in zip(self.input_dataset_paths, self.output_dataset_paths):
            self._augment_docs(ipath, opath)
            #self._datasets[dataset_label] = dataset

    def _augment_docs(self, ipath, opath):
        global tokens_dict
        documents = json.load(open(ipath))
        print(documents)
        augmented_documents = []
        nmultiroot=0
        for document in documents:
            jtokens = document['tokens']
            jrelations = document['relations']
            jentities = document['entities']
            jorig_id = document['orig_id']
            jtext = document['sents']
            print(jorig_id)

            lower_jtokens = jtokens #[t.lower() for t in jtokens]
            print(lower_jtokens)
            text = ' '.join(lower_jtokens)
            print(text)
            #text = str.lower(text)
    
            #tokens_dict = {text: jtokens} #put the text in token_dict
            tokens = nlp(text)            #get annotated tokens
            print(tokens)
            jtags = [token.tag_ for token in tokens]
            print(jtags)
            #self.taglist =self.taglist + jtags
            jdeps = [token.dep_ for token in tokens]
            print(jdeps)
            #"verb_indicator", "dep_head"
            #root = jdeps.index("ROOT") + 1 #as tokens are numbered from 1 by CoreNLP convention
            vpos = list(locate(jdeps, lambda x: x == 'ROOT'))
            if (len(vpos) != 1):
                flag = 1
                nmultiroot += 1
                print("*** Full sentence:", text)
                for i in vpos:
                    print("ROOT [", i, "]: ", jtokens[i], ", pos tag: ", jtags[i], ", dep: ", jdeps[i])
            else:
                flag = 0
            verb_indicator = [0] * len(jdeps)
            for i in vpos:
                verb_indicator[i] = 1
            jdep_heads = []
            for i, token in enumerate(tokens):
              if token.head == token:
                 token_idx = 0
              else:
                 token_idx = token.head.i - tokens[0].i + 1
              jdep_heads.append(token_idx)
            if (flag==1):
            d = {"tokens": jtokens, "pos_tags": jtags, "dep_label": jdeps, "verb_indicator": verb_indicator, "dep_head": jdep_heads, "entities": jentities, "relations": jrelations, "orig_id": jorig_id, "sents":jtext}
            augmented_documents.append(d)
        print("===============  #docs with multiroot = ", nmultiroot)
        with open(opath, "w") as ofile:
            json.dump(augmented_documents, ofile)


if __name__ == "__main__":
    augmenter = JsonInputAugmenter()
    augmenter.augment_docs_in_datasets()
    #print(list(set(augmenter.taglist)))