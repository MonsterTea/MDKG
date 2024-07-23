'''
from transformers import (BertConfig, BertTokenizer, RobertaTokenizer,
                                  BertForSequenceClassification)

from model import (SyntaxBertForSequenceClassification, SyntaxBertForTokenClassification,
                   SyntaxBertConfig, GNNClassifier,
                   SyntaxRobertaForTokenClassification, SyntaxRobertaConfig)


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'syntax_bert_seq': (SyntaxBertConfig, SyntaxBertForSequenceClassification, BertTokenizer),
    'syntax_bert_tok': (SyntaxBertConfig, SyntaxBertForTokenClassification, BertTokenizer),
    'gcn': (SyntaxBertConfig, GNNClassifier, BertTokenizer),
    'syntax_roberta_tok': (SyntaxRobertaConfig, SyntaxRobertaForTokenClassification, RobertaTokenizer),
    'syn_spert': (SyntaxBertConfig, SyntaxBertModel, BertTokenizer) 
}
'''
from transformers import (BertConfig, BertTokenizer)
#from syn_models.syntax_bert import (SyntaxBertConfig, SyntaxBertModel)
from spert.models import SynSpERTConfig
from spert.models import SynSpERT


MODEL_CLASSES = {
    'syn_spert': (SynSpERTConfig, SynSpERT, BertTokenizer) 
}


def read_config_file(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]   
    config = config_class.from_pretrained(args.config_path)
#                                          finetuning_task=args.task_name)
    return config
