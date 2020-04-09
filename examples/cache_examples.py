import argparse
import sys
import csv
import os
import math
import logging
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)


import transformers
from sentence_transformers.util import batch_to_device
from sentence_transformers.readers import InputExample
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models,  SentenceMultiDataset
from torch.utils.data.distributed import DistributedSampler


class PatentDataReader:
    """
    Reads in the Patent dataset. Each line contains two patent texts (s1_col_idx, s2_col_idx) and one label (score_col_idx)
    """

    def __init__(self, dataset_folder, s1_col_idx=-3, s2_col_idx=-2, score_col_idx=-1, delimiter="\t",
                 quoting=csv.QUOTE_NONE, normalize_scores=True, min_score=0, max_score=1):
        self.dataset_folder = dataset_folder
        self.score_col_idx = score_col_idx
        self.s1_col_idx = s1_col_idx
        self.s2_col_idx = s2_col_idx
        self.delimiter = delimiter
        self.quoting = quoting
        self.normalize_scores = normalize_scores
        self.min_score = min_score
        self.max_score = max_score

    def get_examples(self, filename, max_examples=0):
        """
        filename specified which data split to use (train.csv, dev.csv, test.csv).
        """
        data = csv.reader(open(os.path.join(self.dataset_folder, filename), encoding="utf-8"),
                          delimiter=self.delimiter, quoting=self.quoting)
        examples = []
        for idx, row in tqdm(enumerate(data),total=2000000):

            try:
                score = float(row[self.score_col_idx])
            except:
                print(row[self.score_col_idx])
                continue

            if self.normalize_scores:  # Normalize to a 0...1 value
                score = (score - self.min_score) / (self.max_score - self.min_score)



            s1 = row[self.s1_col_idx]
            s2 = row[self.s2_col_idx]
            examples.append(InputExample(guid=filename + str(id), texts=[s1, s2], label=score))

            if max_examples > 0 and len(examples) >= max_examples:
                break

        return examples




def main():
    parser = set_parser()
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("arguments are parsed")

    patent_reader = PatentDataReader(args.data_dir, normalize_scores=True)
    # Use BERT for mapping tokens to embeddings
    #word_embedding_model = models.BERT('bert-base-cased', max_seq_length=510)

    # Apply mean pooling to get one fixed sized sentence vector
    #pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
    #                               pooling_mode_mean_tokens=True,
    #                               pooling_mode_cls_token=False,
    #                               pooling_mode_max_tokens=False)


    #model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_data = load_and_cache_examples(args, patent_reader, None)
    logger.info("caching finished")




def load_and_cache_examples(args, sts_reader, model, evaluate=False):
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            "patent",
            args.part_no
        ),
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        train_data = torch.load(cached_features_file)
    elif args.part_no != "":
        logger.info("Creating features from part %s", args.part_no)
        train_data = SentenceMultiDataset(sts_reader.get_examples('part{}.tsv'.format(args.part_no), max_examples=args.max_example), model, thread_count=args.n_threads)
        logger.info("Data size size is %s", str(len(train_data)))
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(train_data, cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        train_data = SentenceMultiDataset(sts_reader.get_examples('train.tsv', max_examples=args.max_example), model, thread_count=args.n_threads)
        logger.info("Data size size is %s", str(len(train_data)))


        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(train_data, cached_features_file)
            
    
    return train_data


def set_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=510,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_example", default=0, type=int, help="Number of example to be trained on.",
    )


    parser.add_argument('--n_threads', default=4, type=int, help='maximum number of threads for single process')
    parser.add_argument('--part_no', default="", type=str, help='part number for partitioned caching process')
    return parser



if __name__ == '__main__':
    main()
