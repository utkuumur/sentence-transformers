import sys
import csv
import os
import math
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange



scratch_folder = sys.argv[1]
source_folder = sys.argv[2]

if not '{}/sentence-transformers'.format(source_folder) in sys.path:
  sys.path += ['{}/sentence-transformers'.format(source_folder)]
if not source_folder in sys.path:
  sys.path += [source_folder]



import transformers
from sentence_transformers.util import batch_to_device
from sentence_transformers.readers import InputExample
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses, models



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



#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset

model_save_path = "{}/output/training_patent_bert".format(scratch_folder)
sts_reader = PatentDataReader('{}/data'.format(source_folder), normalize_scores=True)

# Use BERT for mapping tokens to embeddings
word_embedding_model = models.BERT('bert-base-cased', max_seq_length=510)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                              pooling_mode_mean_tokens=True,
                              pooling_mode_cls_token=False,
                              pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# model = SentenceTransformer('{}/output/training_patent_bert'.format(source_folder))
# In[ ]:

train_batch_size = 32
num_epochs = 10
test_data = SentencesDataset(examples=sts_reader.get_examples("dev.tsv",max_examples=40), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=train_batch_size)
# evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
# model.evaluate(evaluator)

# Convert the dataset to a DataLoader ready for training
print("Read STSbenchmark train dataset")
train_data = SentencesDataset(sts_reader.get_examples('train.tsv', max_examples=17714), model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_data)*num_epochs/train_batch_size*0.1) #10% of train data for warm-up
print("Warmup-steps: {}".format(warmup_steps))

train_objectives=[(train_dataloader, train_loss)]
evaluator=None
epochs=num_epochs
evaluation_steps=1000
output_path=model_save_path
gradient_accumulation_steps = 4
optimizer_class = transformers.AdamW
optimizer_params = {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False}
max_grad_norm = 1
fp16 = False
local_rank = -1
save_epoch = True

device = model.device

dataloaders = [dataloader for dataloader, _ in train_objectives]

# Use smart batching
for dataloader in dataloaders:
    dataloader.collate_fn = model.smart_batching_collate




loss_models = [nn.DataParallel(loss) for _, loss in train_objectives]
logging.info('number of models is {} '.format(len(loss_models)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for loss_model in loss_models:
    loss_model.to(device)


min_batch_size = min([len(dataloader) for dataloader in dataloaders])
num_train_steps = int(min_batch_size * epochs)

# Prepare optimizers
optimizers = []
schedulers = []
for loss_model in loss_models:
    param_optimizer = list(loss_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    if local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
    scheduler = model._get_scheduler(optimizer, scheduler='WarmupLinear', warmup_steps=warmup_steps, t_total=t_total)

    optimizers.append(optimizer)
    schedulers.append(scheduler)


global_step = 0
data_iterators = [iter(dataloader) for dataloader in dataloaders]

# model = MyDataParallel(model)
# model.to(device)

num_train_objectives = len(train_objectives)
for epoch in trange(epochs, desc="Epoch"):
    training_steps = 0

    for loss_model in loss_models:
        loss_model.zero_grad()
        loss_model.train()

    for step in trange(num_train_objectives * min_batch_size, desc="Iteration"):
        idx = step % num_train_objectives

        loss_model = loss_models[idx]
        optimizer = optimizers[idx]
        scheduler = schedulers[idx]
        data_iterator = data_iterators[idx]

        try:
            data = next(data_iterator)
        except StopIteration:
            logging.info("Restart data_iterator")
            data_iterator = iter(dataloaders[idx])
            data_iterators[idx] = data_iterator
            data = next(data_iterator)

        features, labels = batch_to_device(data, device)
        loss_value = loss_model(features, labels)


        loss_value.mean().backward()
        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)


        training_steps += 1

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1


    if save_epoch:
        model.save(output_path + "_" + str(epoch))

    # self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1)



model.save(model_save_path)

import pickle
pno2abst = pickle.load(open('{}/pno2abstract.dict'.format(scratch_folder),'rb'))
items = list(pno2abst.items())
print(len(items))

pnos = []
abst_texts = []

for item in items:
    pnos.append(item[0])
    abst_texts.append(item[1])


for i in range(4, 4+num_epochs):
    model = SentenceTransformer(model_save_path+"_"+str(i))
    print('max len:',model._first_module().max_seq_length)
    sen_emb_abst = model.encode(abst_texts)
    dct = {}
    for idx, pno in enumerate(pnos):
        dct[pno] = sen_emb_abst[idx]

    file = open('{}/embeddings/pno_abst_emb_v{}.dict'.format(scratch_folder, i),'wb')
    pickle.dump(dct, file)

