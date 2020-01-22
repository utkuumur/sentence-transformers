from logging import getLogger
import os
import sys
import torch
import socket
import signal
import subprocess
import argparse

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Translate sentences")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of sentences per batch")

    # model / output paths
    parser.add_argument("--model_path", type=str, default="", help="Model path")
    parser.add_argument("--output_path", type=str, default="", help="Output path")

    # parser.add_argument("--max_vocab", type=int, default=-1, help="Maximum vocabulary size (-1 to disable)")
    # parser.add_argument("--min_count", type=int, default=0, help="Minimum vocabulary count")

    # source language / target language
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")


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
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=2, type=int,
                        metavar='N',
                        help='number of total epochs to run')


    return parser

parser = get_parser()
params = parser.parse_args()

params.is_slurm_job = 'SLURM_JOB_ID' in os.environ
print("SLURM job: %s" % str(params.is_slurm_job))
print("local rank: %s" % str(params.local_rank))
if params.is_slurm_job:

        # assert params.local_rank == -1   # on the cluster, this is handled by SLURM

        SLURM_VARIABLES = [
            'SLURM_JOB_ID',
            'SLURM_JOB_NODELIST', 'SLURM_JOB_NUM_NODES', 'SLURM_NTASKS', 'SLURM_TASKS_PER_NODE',
            'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU',
            'SLURM_NODEID', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_TASK_PID'
        ]

        PREFIX = "%i - " % int(os.environ['SLURM_PROCID'])
        for name in SLURM_VARIABLES:
            value = os.environ.get(name, None)
            print(PREFIX + "%s: %s" % (name, str(value)))

        # # job ID
        # params.job_id = os.environ['SLURM_JOB_ID']

        # number of nodes / node ID
        params.n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        params.node_id = int(os.environ['SLURM_NODEID'])

        # local rank on the current node / global rank
        params.global_rank = params.local_rank
        params.local_rank = int(os.environ['SLURM_LOCALID']) % 4

        # number of processes / GPUs per node
        params.world_size = int(os.environ['SLURM_NTASKS'])
        params.n_gpu_per_node = params.world_size // params.n_nodes

        # define master address and master port
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        params.master_addr = hostnames.split()[0].decode('utf-8')
        # print("Master add")
        assert 10001 <= params.master_port <= 20000 or params.world_size == 1
        print(PREFIX + "Master address: %s" % params.master_addr)
        print(PREFIX + "Master port   : %i" % params.master_port)

        # set environment variables for 'env://'
        os.environ['MASTER_ADDR'] = params.master_addr
        os.environ['MASTER_PORT'] = str(params.master_port)
        os.environ['WORLD_SIZE'] = str(params.world_size)
        os.environ['RANK'] = str(params.global_rank)


# sanity checks
assert params.n_nodes >= 1
assert 0 <= params.node_id < params.n_nodes
assert 0 <= params.local_rank <= params.global_rank < params.world_size
assert params.world_size == params.n_nodes * params.n_gpu_per_node

params.is_master = params.node_id == 0 and params.local_rank == 0
params.multi_node = params.n_nodes > 1
params.multi_gpu = params.world_size > 1

# summary
print("\n\n")
PREFIX = "%i - " % params.global_rank
print(PREFIX + "Number of nodes: %i" % params.n_nodes)
print(PREFIX + "Node ID        : %i" % params.node_id)
print(PREFIX + "Local rank     : %i" % params.local_rank)
print(PREFIX + "Global rank    : %i" % params.global_rank)
print(PREFIX + "World size     : %i" % params.world_size)
print(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
print(PREFIX + "Master         : %s" % str(params.is_master))
print(PREFIX + "Multi-node     : %s" % str(params.multi_node))
print(PREFIX + "Multi-GPU      : %s" % str(params.multi_gpu))
print(PREFIX + "Hostname       : %s" % socket.gethostname())
