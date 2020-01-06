from sentence_transformers.readers import InputExample
import csv
import os


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

            s1 = row[self.s1_col_idx]
            s2 = row[self.s2_col_idx]
            examples.append(InputExample(guid=filename + str(id), texts=[s1, s2], label=score))

            if max_examples > 0 and len(examples) >= max_examples:
                break

        return examples