import argparse
import sys

#DKS: This is a single arg parser that is invoked for any input script (train/eval).
#DKS: In future, we might have 2 parsers, one for train and another for eval.


#def _add_all_args(arg_parser):
def get_argparser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")

    arg_parser.add_argument('--config_path', type=str, help="Path to config file")

    # Input
    arg_parser.add_argument('--types_path', type=str, help="Path to type specifications")

    # Preprocessing
    arg_parser.add_argument('--tokenizer_path', type=str, help="Path to tokenizer")
    arg_parser.add_argument('--max_span_size', type=int, default=10, help="Maximum size of spans")
    arg_parser.add_argument('--lowercase', action='store_true', default=False,
                            help="If true, input is lowercased during preprocessing")
    arg_parser.add_argument('--sampling_processes', type=int, default=4,
                            help="Number of sampling processes. 0 = no multiprocessing for sampling")
    arg_parser.add_argument('--sampling_limit', type=int, default=100, help="Maximum number of sample batches in queue")

    # Logging
    arg_parser.add_argument('--label', type=str, help="Label of run. Used as the directory name of logs/models")
    arg_parser.add_argument('--log_path', type=str, help="Path do directory where training/evaluation logs are stored")
    arg_parser.add_argument('--store_predictions', action='store_true', default=False,
                            help="If true, store predictions on disc (in log directory)")
    arg_parser.add_argument('--store_examples', action='store_true', default=True,
                            help="If true, store evaluation examples on disc (in log directory)")
    arg_parser.add_argument('--example_count', type=int, default=None,
                            help="Count of evaluation example to store (if store_examples == True)")
    arg_parser.add_argument('--debug', action='store_true', default=False, help="Debugging mode on/off")

    # Model / Training / Evaluation
    arg_parser.add_argument('--model_path', type=str, help="Path to directory that contains model checkpoints")
    arg_parser.add_argument('--model_type', type=str, default="spert", help="Type of model")
    arg_parser.add_argument('--cpu', action='store_true', default=False,
                            help="If true, train/evaluate on CPU even if a CUDA device is available")
    arg_parser.add_argument('--eval_batch_size', type=int, default=1, help="Evaluation batch size")
    arg_parser.add_argument('--max_pairs', type=int, default=1000,
                            help="Maximum entity pairs to process during training/evaluation")
    arg_parser.add_argument('--rel_filter_threshold', type=float, default=0.4, help="Filter threshold for relations")
    arg_parser.add_argument('--size_embedding', type=int, default=25, help="Dimensionality of size embedding")
    arg_parser.add_argument('--prop_drop', type=float, default=0.1, help="Probability of dropout used in SpERT")
    arg_parser.add_argument('--freeze_transformer', action='store_true', default=False, help="Freeze BERT weights")
    arg_parser.add_argument('--no_overlapping', action='store_true', default=False,
                            help="If true, do not evaluate on overlapping entities "
                                 "and relations with overlapping entities")

    # Misc
    arg_parser.add_argument('--seed', type=int, default=None, help="Seed")
    arg_parser.add_argument('--cache_path', type=str, default=None,
                            help="Path to cache transformer models (for HuggingFace transformers library)")


#def train_argparser():
#    arg_parser = argparse.ArgumentParser()

    # Input
    arg_parser.add_argument('--train_path', type=str, help="Path to train dataset")
    arg_parser.add_argument('--valid_path', type=str, help="Path to validation dataset")

    # Logging
    arg_parser.add_argument('--save_path', type=str, help="Path to directory where model checkpoints are stored")
    arg_parser.add_argument('--init_eval', action='store_true', default=False,
                            help="If true, evaluate validation set before training")
    arg_parser.add_argument('--save_optimizer', action='store_true', default=False,
                            help="Save optimizer alongside model")
    arg_parser.add_argument('--train_log_iter', type=int, default=1, help="Log training process every x iterations")
    arg_parser.add_argument('--final_eval', action='store_true', default=False,
                            help="Evaluate the model only after training, not at every epoch")

    # Model / Training
    arg_parser.add_argument('--train_batch_size', type=int, default=1, help="Training batch size")
    arg_parser.add_argument('--epochs', type=int, default=20, help="Number of epochs")
    arg_parser.add_argument('--neg_entity_count', type=int, default=100,
                            help="Number of negative entity samples per document (sentence)")
    arg_parser.add_argument('--neg_relation_count', type=int, default=100,
                            help="Number of negative relation samples per document (sentence)")
    arg_parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    arg_parser.add_argument('--lr_warmup', type=float, default=0.1,
                            help="Proportion of total train iterations to warmup in linear increase/decrease schedule")
    arg_parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay to apply")
    arg_parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm")
    arg_parser.add_argument('--noise_lambda', type=float, default=0.15, help="noise_lambda for noisy tune")

#    _add_common_args(arg_parser)

#    return arg_parser


#def eval_argparser():
#    arg_parser = argparse.ArgumentParser()

    # Input
    arg_parser.add_argument('--dataset_path', type=str, help="Path to dataset")
    
    
    arg_parser.add_argument('--pos_embedding', type=int, default=25, help="Dimensionality of pos embedding")

    ##########################################################################
    arg_parser.add_argument('--use_pos', action='store_true', default=False, 
                            help="Whether pos embedding of tokens to be used")
    arg_parser.add_argument('--use_entity_clf', type=str, default="none", 
                            help="Type of entity classifier output to be used for relation classification")


#    _add_common_args(arg_parser)

    return arg_parser
