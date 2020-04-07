import argparse
import sys

if not '/home/caskurlu/patentBert/sentence-transformers' in sys.path:
      sys.path += ['/home/caskurlu/patentBert/sentence-transformers']

from sentence_transformers.readers import InputExample
import csv
import os
import pickle
import logging
class PatentDataReader:
    """
    Reads in the Patent dataset. Each line contains two patent texts (s1_col_idx, s2_col_idx) and one label (score_col_idx)
    """

    def __init__(self, dataset_folder, s1_col_idx=7, s2_col_idx=8, score_col_idx=-1, delimiter="\t",
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
        for id, row in enumerate(data):

            try:
                score = float(row[self.score_col_idx])
            except:
                print(row[self.score_col_idx])
                continue

            if self.normalize_scores:  # Normalize to a 0...1 value
                score = (score - self.min_score) / (self.max_score - self.min_score)

            s1 = row[self.s1_col_idx].lower()
            s2 = row[self.s2_col_idx].lower()
            examples.append(InputExample(guid=filename + str(id), texts=[s1, s2], label=score))

            if max_examples > 0 and len(examples) >= max_examples:
                break

        return examples
from sentence_transformers import SentenceTransformer, LoggingHandler



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    #parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir. Should contain the .tsv files (or other data files) for the task.",     )
    parser.add_argument("--model_dir", default=None, type=str, required=True, help="The model dir")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output dir")
    parser.add_argument("--version", default=None, type=str, required=True, help="version of the model dir")
    parser.add_argument("--source_file", default=None, type=str, required=True, help="Path to the input data.",     )

    args = parser.parse_args()

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    model = SentenceTransformer(args.model_dir)
    '''
    pno2abstract = pickle.load(open('{}/pno2abstract.dict'.format(args.data_dir), 'rb'))

    # model._first_module().max_seq_length = 256
    pnos = []
    texts = []
    items = list(pno2abstract.items())
    for pno,text in items:
        pnos.append(pno)
        texts.append(text)

    embeddings = model.encode(texts)
    dct = {}
    for idx in range(len(pnos)):
        dct[pnos[idx]] = embeddings[idx]

    pickle.dump(dct, open('{}/{}_abstract_embeddings.dict'.format(args.output_dir, args.version), 'wb'))
    '''
    pno2desc = pickle.load(open(args.source_file, 'rb'))

    # model._first_module().max_seq_length = 256
    pnos = []
    texts = []
    for pno in pno2desc.keys():
        pnos.append(pno)
        texts.append(pno2desc[pno])

    embeddings = model.encode(texts)
    dct = {}
    for idx in range(len(pnos)):
        dct[pnos[idx]] = embeddings[idx]

    pickle.dump(dct, open('{}/pno2vec_{}.dict'.format(args.output_dir, args.version), 'wb'))

if __name__ == '__main__':
    main()
