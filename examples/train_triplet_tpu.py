import argparse
import sys
import csv
import os
import math
import logging
import datetime
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.utils.utils import SampleGenerator
import torch.nn as nn
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)
import transformers
from sentence_transformers.util import batch_to_device
from sentence_transformers.readers import InputExample, TripletReader
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models#,  SentenceMultiDataset
from torch.utils.data.distributed import DistributedSampler





def train(args, train_dataset, model, train_loss):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    #train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = SampleGenerator(train_dataset, sample_count =args.train_batch_size)

    train_objectives = [(train_dataloader, train_loss)]
    epochs = args.epochs
    output_path = args.output_dir
    optimizer_class = transformers.AdamW
    optimizer_params = {'lr': args.learning_rate, 'eps': 1e-6, 'correct_bias': False}
    max_grad_norm = 1
    # local_rank = -1
    save_epoch = True

    dataloaders = [dataloader for dataloader, _ in train_objectives]
    # Use smart batching
    for dataloader in dataloaders:
        dataloader.collate_fn = model.smart_batching_collate

    loss_models = [loss for _, loss in train_objectives]
    logging.info('number of models is {} '.format(len(loss_models)))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for loss_model in loss_models:
        loss_model.to(args.device)

    model.best_score = -9999

    min_batch_size = min([len(dataloader) for dataloader in dataloaders])
    num_train_steps = int(min_batch_size * epochs)
    warmup_steps = math.ceil(len(train_dataset) * args.epochs / args.train_batch_size * 0.1)  # 10% of train data for warm-up
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
        if args.local_rank != -1:
            t_total = t_total // args.world_size

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler = model._get_scheduler(optimizer, scheduler='WarmupLinear', warmup_steps=warmup_steps,
                                         t_total=t_total)

        optimizers.append(optimizer)
        schedulers.append(scheduler)




    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", epochs)
    logger.info("  Instantaneous batch size per TPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Total optimization steps = %d", t_total)



    global_step = 0
    data_iterators = [iter(dataloader) for dataloader in dataloaders]
    num_train_objectives = len(train_objectives)
    # set_seeds(1,args)
    tr_loss = 0.0
    for epoch in trange(epochs, desc="Epoch", disable=args.local_rank not in [-1, 0]):
        training_steps = 0

        for loss_model in loss_models:
            loss_model.zero_grad()
            loss_model.train()

        for step in trange(num_train_objectives * min_batch_size, desc="Iteration", disable=args.local_rank not in [-1, 0]):
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

            features, labels = batch_to_device(data, args.device)
            loss_value = loss_model(features, labels)
            
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)

            training_steps += 1
            tr_loss += loss_value.item()

            #optimizer.step()
            xm.optimizer_step(optimizer, barrier=True)
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

           

        if args.local_rank in [-1, 0] and save_epoch:
            model.save(output_path + "_" + str(epoch))

    return tr_loss / global_step


def main():
    parser = set_parser()
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.info("arguments are parsed")
    args.world_size = args.gpus * args.nodes

    #patent_reader = PatentDataReader(args.data_dir, normalize_scores=True)
    patent_reader = TripletReader(args.data_dir,s1_col_idx=3, s2_col_idx=4, s3_col_idx=5, has_header=True)
    # Use BERT for mapping tokens to embeddings
    logger.warning("Loading Bert Model")
    word_embedding_model = models.BERT('bert-base-uncased', max_seq_length=510)
    logger.warning("Model is loaded")
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)


    if args.use_tpu:
        logger.warning("TPU training")
        device = xm.xla_device()
        args.n_gpu = 1
    elif args.local_rank == -1:
        logger.warning("Non dist training")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        logger.warning("Dist training local rank %s", args.local_rank)
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=10))
        args.n_gpu = 1
    args.device = device

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    train_loss = losses.TripletLoss(model=model)
    model.to(args.device)
    train_loss.to(args.device)


    # Training
    if args.do_train:
        logger.warning("Read Patent Training dataset")
        train_data = load_and_cache_examples(args, patent_reader, model)
        logger.warning("Training dataset is loaded")
        # train_data = SentencesDataset(patent_reader.get_examples('train.tsv', max_examples=17714), model)
        tr_loss = train(args, train_data, model, train_loss)
        logger.info(" average loss = %s", tr_loss)




def load_and_cache_examples(args, sts_reader, model, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            "patent",
        ),
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        train_data = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        #train_data = SentenceMultiDataset(sts_reader.get_examples('train.tsv', max_examples=args.max_example), model, thread_count=args.n_threads)
        train_data = SentencesDataset(sts_reader.get_examples('train.tsv', max_examples=args.max_example), model)
        logger.info("Data size size is %s", str(len(train_data)))

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(train_data, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

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
        "--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation.",
    )

    parser.add_argument(
        "--max_example", default=0, type=int, help="Number of example to be trained on.",
    )
    parser.add_argument("--use_tpu", action="store_true", help="Use tpu")
    parser.add_argument("--learning_rate", default=4e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=2, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--n_threads', default=4, type=int, help='maximum number of threads for single process')

    parser.add_argument('--save_steps', default=10000, type=int)

    return parser



if __name__ == '__main__':
    main()
