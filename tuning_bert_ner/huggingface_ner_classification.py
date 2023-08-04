# Databricks notebook source
# MAGIC %md 
# MAGIC ### TO DO 
# MAGIC - put dataset under DELTA / UC - use Spark HF Datasets 
# MAGIC - places them under HF prquet loader or read with Ray 
# MAGIC
# MAGIC - add time execution benchmark 
# MAGIC
# MAGIC
# MAGIC - pandas udf  -> map_batches code for pre-trained models inference 
# MAGIC - serving -> on GPU A10 

# COMMAND ----------

# MAGIC %pip install ray[default,rllib,tune]==2.3.1 mlflow==2.3 seqeval

# COMMAND ----------

try: 
  shutdown_ray_cluster()
except:
  print("No Ray cluster is initiated")

# COMMAND ----------

import datasets
import evaluate
import numpy as np
from datasets import ClassLabel, load_dataset,load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

# COMMAND ----------


# Ensure the correct number of cores are mentioned as per the cluster selected -> per node ? 
# g5.2 - 1GPU	24GPURAM	8CPU	32GB
# g5.4 - 1GPU 24 GRAM 16CPU 64GB 
#setup_ray_cluster(num_worker_nodes=MAX_NUM_WORKER_NODES,num_cpus_per_node=8)

#GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
task = "cola"
batch_size = 64
use_gpu = True  # set this to False to run on CPUs
num_workers = 4  # set this to number of GPUs/CPUs you want to use, Configure based on the total gpus across the worker node

model_checkpoint = "bert-large-cased" #"distilbert-base-uncased"
#model_checkpoint = "distilbert-base-uncased"
pretrained_model_name_or_path = model_checkpoint #"EleutherAI/pythia-12b"

num_cpu_cores_per_worker = 8 # total cpu's present in each node
num_gpu_per_worker = 1 # total gpu's present in each node
resource_per_worker_int = num_cpu_cores_per_worker/num_gpu_per_worker 
max_length = 1024
local_output_dir = '/tmp/run/details'
gradient_checkpointing = True
DEFAULT_SEED = 42
seed = DEFAULT_SEED 

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
experiment_location = f"/Users/{username}/hf_multi-gpu"


# COMMAND ----------

from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster, MAX_NUM_WORKER_NODES
import ray

RAY_LOG_DIR = "/dbfs/tmp/ray/logs/"
# Start the ray cluster
# This is not required if oyu do not use DeepSpeed
setup_ray_cluster(
  num_worker_nodes=MAX_NUM_WORKER_NODES,
  num_cpus_per_node=num_cpu_cores_per_worker,
  num_gpus_per_node=num_gpu_per_worker,
  collect_log_to_path=RAY_LOG_DIR#"/dbfs/path/to/ray_collected_logs"
  )


# COMMAND ----------

# MAGIC %md
# MAGIC # Fine-tune a 🤗 Transformers model

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook is based on [an official 🤗 notebook - "How to fine-tune a model on text classification"](https://github.com/huggingface/notebooks/blob/6ca682955173cc9d36ffa431ddda505a048cbe80/examples/text_classification.ipynb). The main aim of this notebook is to show the process of conversion from vanilla 🤗 to [Ray AIR](https://docs.ray.io/en/latest/ray-air/getting-started.html) 🤗 without changing the training logic unless necessary.
# MAGIC
# MAGIC In this notebook, we will:
# MAGIC 1. [Set up Ray](#setup)
# MAGIC 2. [Load the dataset](#load)
# MAGIC 3. [Preprocess the dataset with Ray AIR](#preprocess)
# MAGIC 4. [Run the training with Ray AIR](#train)
# MAGIC 5. [Predict on test data with Ray AIR](#predict)
# MAGIC 6. [Optionally, share the model with the community](#share)

# COMMAND ----------

# MAGIC %md
# MAGIC Uncomment and run the following line in order to install all the necessary dependencies (this notebook is being tested with `transformers==4.19.1`):

# COMMAND ----------

import ray
ray.init(address='auto')

# COMMAND ----------

print(ray.cluster_resources())

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we will see how to fine-tune one of the [🤗 Transformers](https://github.com/huggingface/transformers) model to a token classification task of the [GLUE Benchmark](https://gluebenchmark.com/). We will be running the training using [Ray AIR](https://docs.ray.io/en/latest/ray-air/getting-started.html).
# MAGIC
# MAGIC You can change those two variables to control whether the training (which we will get to later) uses CPUs or GPUs, and how many workers should be spawned. Each worker will claim one CPU or GPU. Make sure not to request more resources than the resources present!
# MAGIC
# MAGIC By default, we will run the training with one GPU worker.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-tuning a model on a token classification task

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading the dataset <a name="load"></a>

# COMMAND ----------

# MAGIC %md
# MAGIC We will use the [🤗 Datasets](https://github.com/huggingface/datasets) library to download the data and get the metric we need to use for evaluation (to compare our model to the benchmark). This can be easily done with the functions `load_dataset` and `load_metric`.
# MAGIC
# MAGIC Apart from `mnli-mm` being a special code, we can directly pass our task name to those functions.
# MAGIC
# MAGIC As Ray AIR doesn't provide integrations for 🤗 Datasets yet, we will simply run the normal 🤗 Datasets code to load the dataset from the Hub.

# COMMAND ----------

from datasets import load_dataset
raw_datasets = load_dataset("conll2003")
ray_datasets = ray.data.from_huggingface(raw_datasets)

# COMMAND ----------

# MAGIC %sql
# MAGIC use CATALOG pj ;
# MAGIC CREATE DATABASE if not exists llm ;

# COMMAND ----------

# # saving data to the Delta to show 
# train_df = datasets["train"].to_pandas()
# val_df = datasets["validation"].to_pandas()
# test_df = datasets["test"].to_pandas()

# display(spark.createDataFrame(train_df))
# spark.createDataFrame(train_df).write.format("delta").saveAsTable("pj.llm.training_wikiann_ner")


# COMMAND ----------

# MAGIC %md
# MAGIC The `dataset` object itself is [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict), which contains one key for the training, validation, and test set (with more keys for the mismatched validation and test set in the special case of `mnli`).

# COMMAND ----------

# MAGIC %md
# MAGIC We will also need the metric. In order to avoid serialization errors, we will load the metric inside the training workers later. Therefore, now we will just define the function we will use.

# COMMAND ----------

# MAGIC %md
# MAGIC The metric is an instance of [`datasets.Metric`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Metric).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing the data with Ray AIR <a name="preprocess"></a>

# COMMAND ----------

# MAGIC %md
# MAGIC Before we can feed those texts to our model, we need to preprocess them. This is done by a 🤗 Transformers' `Tokenizer`, which will (as the name indicates) tokenize the inputs (including converting the tokens to their corresponding IDs in the pretrained vocabulary) and put it in a format the model expects, as well as generate the other inputs that model requires.
# MAGIC
# MAGIC To do all of this, we instantiate our tokenizer with the `AutoTokenizer.from_pretrained` method, which will ensure that:
# MAGIC
# MAGIC - we get a tokenizer that corresponds to the model architecture we want to use,
# MAGIC - we download the vocabulary used when pretraining this specific checkpoint.

# COMMAND ----------


# task_to_keys = {
#     "cola": ("sentence", None),
#     "mnli": ("premise", "hypothesis"),
#     "mnli-mm": ("premise", "hypothesis"),
#     "mrpc": ("sentence1", "sentence2"),
#     "qnli": ("question", "sentence"),
#     "qqp": ("question1", "question2"),
#     "rte": ("sentence1", "sentence2"),
#     "sst2": ("sentence", None),
#     "stsb": ("sentence1", "sentence2"),
#     "wnli": ("sentence1", "sentence2"),
# }


# import ray.data

# ray_datasets = ray.data.from_huggingface(datasets)

# import pandas as pd
# from ray.data.preprocessors import BatchMapper

# def preprocess_function(examples: pd.DataFrame):
#     # if we only have one column, we are inferring.
#     # no need to tokenize in that case. 
#     if len(examples.columns) == 1:
#         return examples
#     examples = examples.to_dict("list")
#     sentence1_key, sentence2_key = task_to_keys[task]
#     if sentence2_key is None:
#         ret = tokenizer(examples[sentence1_key], truncation=True)
#     else:
#         ret = tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)
#     # Add back the original columns
#     ret = {**examples, **ret}
#     return pd.DataFrame.from_dict(ret)

# COMMAND ----------

# MAGIC %md
# MAGIC We pass along `use_fast=True` to the call above to use one of the fast tokenizers (backed by Rust) from the 🤗 Tokenizers library. Those fast tokenizers are available for almost all models, but if you got an error with the previous call, remove that argument.

# COMMAND ----------

# MAGIC %md
# MAGIC To preprocess our dataset, we will thus need the names of the columns containing the sentence(s). The following dictionary keeps track of the correspondence task to column names:

# COMMAND ----------

# MAGIC %md
# MAGIC For Ray AIR, instead of using 🤗 Dataset objects directly, we will convert them to [Ray Datasets](https://docs.ray.io/en/latest/data/dataset.html). Both are backed by Arrow tables, so the conversion is straightforward. We will use the built-in `ray.data.from_huggingface` function.

# COMMAND ----------

# MAGIC %md
# MAGIC We can then write the function that will preprocess our samples. We just feed them to the `tokenizer` with the argument `truncation=True`. This will ensure that an input longer than what the model selected can handle will be truncated to the maximum length accepted by the model.
# MAGIC
# MAGIC We use a `BatchMapper` to create a Ray AIR preprocessor that will map the function to the dataset in a distributed fashion. It will run during training and prediction.

# COMMAND ----------

set_seed(DEFAULT_SEED)

# COMMAND ----------

training_args =dict()
training_args['do_train'] = True
training_args['text_column_name'] = None
training_args['label_column_name'] = None
training_args['pad_to_max_length'] = True
training_args['max_seq_length'] = None
training_args['label_all_tokens'] = False 
training_args['fp16'] = True
training_args['return_entity_level_metrics'] = False
training_args['task_name'] = 'ner'

# COMMAND ----------

if training_args['do_train']:
    column_names = raw_datasets["train"].column_names
    features = raw_datasets["train"].features
else:
    column_names = raw_datasets["validation"].column_names
    features = raw_datasets["validation"].features

if training_args['text_column_name'] is not None:
    text_column_name = data_args.text_column_name
elif "tokens" in column_names:
    text_column_name = "tokens"
else:
    text_column_name = column_names[0]

if training_args['label_column_name'] is not None:
    label_column_name = data_args.label_column_name
elif f"{training_args['task_name']}_tags" in column_names:
    label_column_name = f"{training_args['task_name']}_tags"
else:
    label_column_name = column_names[1]

# In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
# unique labels.
def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list

# If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
# Otherwise, we have to get the list of labels manually.
labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
if labels_are_int:
    label_list = features[label_column_name].feature.names
    label_to_id = {i: i for i in range(len(label_list))}
else:
    label_list = get_label_list(raw_datasets["train"][label_column_name])
    label_to_id = {l: i for i, l in enumerate(label_list)}

num_labels = len(label_list)
print(label_list)

# COMMAND ----------

# Intialize model and tokenizer and config to set the correct labels and ID's
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    use_fast=True,)
config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path,
    num_labels=num_labels,
    finetuning_task=training_args['task_name'])

model = AutoModelForTokenClassification.from_pretrained(
    pretrained_model_name_or_path,
    config=config)

# Tokenizer check: this script requires a fast tokenizer.
if not isinstance(tokenizer, PreTrainedTokenizerFast):
    raise ValueError(
        "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
        " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
        " this requirement"
    )

# COMMAND ----------

# Model has labels -> use them.
if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
    print('test')
    if sorted(model.config.label2id.keys()) == sorted(label_list):
        # Reorganize `label_list` to match the ordering of the model.
        if labels_are_int:
            label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
            label_list = [model.config.id2label[i] for i in range(num_labels)]
        else:
            label_list = [model.config.id2label[i] for i in range(num_labels)]
            label_to_id = {l: i for i, l in enumerate(label_list)}
    else:
        logger.warning(
            "Your model seems to have been trained with labels, but they don't match the dataset: ",
            f"model labels: {sorted(model.config.label2id.keys())}, dataset labels:"
            f" {sorted(label_list)}.\nIgnoring the model labels as a result.",
        )
del model

# COMMAND ----------


import pandas as pd
import ray.data
from ray.data.preprocessors import BatchMapper


ray_datasets = ray.data.from_huggingface(raw_datasets)
# Preprocessing the dataset
# Padding strategy
padding = "max_length" if training_args['pad_to_max_length'] else False

# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=padding,
        truncation=True,
        max_length= training_args['max_seq_length'],
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                if training_args['label_all_tokens']:
                    label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

#model = BertModel.from_pretrained("bert-base-multilingual-cased")
#tokenizer = BertTokenizer.from_pretrained(model_checkpoint, use_fast=True)


def preprocess_func(text: pd.DataFrame):
    dataset = Dataset.from_pandas(text)
    dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        desc="Running tokenizer on train dataset")
    return dataset.to_pandas()

batch_encoder = BatchMapper(preprocess_func, batch_format="pandas")

# COMMAND ----------

# train = raw_datasets['train'].map(
#     tokenize_and_align_labels,
#     batched=True,
#     desc="Running tokenizer on train dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fine-tuning the model with Ray AIR <a name="train"></a>

# COMMAND ----------

import mlflow
from utils import get_or_create_experiment

# get or create an experiment
get_or_create_experiment(experiment_location)


# COMMAND ----------

# MAGIC %md
# MAGIC Now that our data is ready, we can download the pretrained model and fine-tune it.
# MAGIC
# MAGIC Since all our tasks are about sentence classification, we use the `AutoModelForSequenceClassification` class.
# MAGIC
# MAGIC We will not go into details about each specific component of the training (see the [original notebook](https://github.com/huggingface/notebooks/blob/6ca682955173cc9d36ffa431ddda505a048cbe80/examples/text_classification.ipynb) for that). The tokenizer is the same as we have used to encoded the dataset before.
# MAGIC
# MAGIC The main difference when using the Ray AIR is that we need to create our 🤗 Transformers `Trainer` inside a function (`trainer_init_per_worker`) and return it. That function will be passed to the `HuggingFaceTrainer` and will run on every Ray worker. The training will then proceed by the means of PyTorch DDP.
# MAGIC
# MAGIC Make sure that you initialize the model, metric, and tokenizer inside that function. Otherwise, you may run into serialization errors.
# MAGIC
# MAGIC Furthermore, `push_to_hub=True` is not yet supported. Ray will, however, checkpoint the model at every epoch, allowing you to push it to hub manually. We will do that after the training.
# MAGIC
# MAGIC If you wish to use thrid party logging libraries, such as MLflow or Weights&Biases, do not set them in `TrainingArguments` (they will be automatically disabled) - instead, you should pass Ray AIR callbacks to `HuggingFaceTrainer`'s `run_config`. In this example, we will use MLflow.

# COMMAND ----------

# num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
# metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
# model_name = model_checkpoint.split("/")[-1]
# validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
# name = f"{model_name}-finetuned-{task}"

# COMMAND ----------

#!mkdir /local_disk0/tmp/ray/

# COMMAND ----------

import os
import re
import json
import logging
import subprocess

import numpy as np
import torch

from pathlib import Path
from functools import partial
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import ray
from ray.air import session
import ray.util.scheduling_strategies
from ray.train.huggingface import HuggingFaceTrainer
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.air.config import ScalingConfig

from ray.data.preprocessors import Chain
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster, MAX_NUM_WORKER_NODES


from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from datasets import load_dataset,load_from_disk

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)



# COMMAND ----------






def trainer_init_per_worker(train_dataset, eval_dataset = None, **config):

    set_seed(seed)
    # Use the actual number of CPUs assigned by Ray
    os.environ["OMP_NUM_THREADS"] = str(
        session.get_trial_resources().bundles[-1].get("CPU", 1)
    )
    # Enable tf32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Is CUDA available: {torch.cuda.is_available()}")

    # Metrics
    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if training_args['return_entity_level_metrics']:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
    print("Loading model")        
    tokenizer = AutoTokenizer.from_pretrained(
                                pretrained_model_name_or_path,
                                use_fast=True,)
    
    
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args['fp16'] else None)
    
    model_config = AutoConfig.from_pretrained(
                            pretrained_model_name_or_path,
                            num_labels=num_labels,
                            finetuning_task=training_args['task_name'])

    model = AutoModelForTokenClassification.from_pretrained(
                            pretrained_model_name_or_path,
                            config=model_config)
    
    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = dict(enumerate(label_list))

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)
    
    epochs = config.get("epochs", 10)
    lr = config.get("learning_rate", 2e-5)
    per_device_train_batch_size = config.get("per_device_train_batch_size")
    per_device_eval_batch_size = config.get("per_device_eval_batch_size")
    logging_steps = config.get("logging_steps")
    save_strategy= config.get("save_strategy")
    evaluation_strategy = config.get("evaluation_strategy")
    save_steps = config.get("save_steps")
    eval_steps = config.get("eval_steps") 
    #warmup_steps = config.get("warmup_steps")
    disable_tqdm=config.get("disable_tqdm")
    #remove_unused_columns=config.get("remove_unused_columns")


    args = TrainingArguments(
        output_dir=pretrained_model_name_or_path,
        learning_rate=lr,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=epochs,
        #weight_decay=config.get("weight_decay", 0.01),
        disable_tqdm=True,  # declutter the output a little
        #gradient_checkpointing=gradient_checkpointing,
        logging_strategy=evaluation_strategy,
        logging_steps=logging_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        #remove_unused_columns=remove_unused_columns,
        #warmup_steps=warmup_steps,
        fp16=True, # change to true if using v100 or T4
        bf16=False,# change to false if using v100 or T4 (supported starting from A10)
        push_to_hub=False,
        no_cuda=False,  # you need to explicitly set no_cuda if you want CPUs
        load_best_model_at_end=False,

    )



    print("Model loaded")
    print("Train data size: %d", len(train_dataset))
    print("Test data size: %d", len(eval_dataset))

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("Starting training")
    return trainer

# COMMAND ----------

# MAGIC %md
# MAGIC With our `trainer_init_per_worker` complete, we can now instantiate the `HuggingFaceTrainer`. Aside from the function, we set the `scaling_config`, controlling the amount of workers and resources used, and the `datasets` we will use for training and evaluation.
# MAGIC
# MAGIC We specify the `MLflowLoggerCallback` inside the `run_config`, and pass the preprocessor we have defined earlier as an argument. The preprocessor will be included with the returned `Checkpoint`, meaning it will also be applied during inference.

# COMMAND ----------

!mkdir -p  /dbfs/Users/puneet.jain@databricks.com/hf/job/

# COMMAND ----------

from ray.train.huggingface import HuggingFaceTrainer
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback

tags = dict(
  local_dir = f"/dbfs/Users/{username}/hf/job/",
  base_model_dir = pretrained_model_name_or_path,
  n_gpus = str(num_workers),
  num_cpu_cores_per_worker = str(num_cpu_cores_per_worker -2),
  num_gpu_per_worker = str(num_gpu_per_worker),  
  max_length = str(max_length),
  username = username)


trainer = HuggingFaceTrainer(
    trainer_init_per_worker=trainer_init_per_worker,
    trainer_init_config={
        "lr" : 1e-6, # per device
        "per_device_train_batch_size" : 5,
        "per_device_eval_batch_size" : 5,
        "save_strategy" : "epoch",
        "evaluation_strategy" : "epoch",
        "logging_steps" : 500,
        "save_steps" : 500,
        "eval_steps" : 500,
        "warmup_steps" : 25,
        "disable_tqdm" : True,
        "remove_unused_columns" :False,
        "epochs": 2
        },
    scaling_config=ScalingConfig(
                    num_workers=num_workers, 
                    use_gpu=use_gpu,
                    resources_per_worker={"GPU": 1}
                    ),
    run_config=RunConfig(
        local_dir = f"/local_disk0/Users/{username}/hf/job/",
        callbacks=[
          MLflowLoggerCallback(
                    experiment_name=experiment_location,
                    tags=tags, 
                    save_artifact=True#True will not work with dbfs -> check with what this can work (local_disk0?)
                    )
                  ],
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="eval_loss",
            checkpoint_score_order="min",
            ),
        ),
    datasets={
        "train": ray_datasets["train"],
        "evaluation": ray_datasets['validation'],
        },
 
    preprocessor=batch_encoder,
)


# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we call the `fit` method to start training with Ray AIR. We will save the `Result` object to a variable so we can access metrics and checkpoints.

# COMMAND ----------

# DBTITLE 1,fvhi
result = trainer.fit()

# COMMAND ----------

result

# COMMAND ----------

# MAGIC %md
# MAGIC You can use the returned `Result` object to access metrics and the Ray AIR `Checkpoint` associated with the last iteration.

# COMMAND ----------

result.checkpoint

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tune hyperparameters with Ray AIR <a name="predict"></a>

# COMMAND ----------

# MAGIC %md
# MAGIC If we would like to tune any hyperparameters of the model, we can do so by simply passing our `HuggingFaceTrainer` into a `Tuner` and defining the search space.
# MAGIC
# MAGIC We can also take advantage of the advanced search algorithms and schedulers provided by Ray Tune. In this example, we will use an `ASHAScheduler` to aggresively terminate underperforming trials.

# COMMAND ----------

from ray import tune
from ray.tune import Tuner
from ray.tune.schedulers.async_hyperband import ASHAScheduler

tune_epochs = 4

tuner = Tuner(
    trainer,
    param_space={
        "trainer_init_config": {
            "learning_rate": tune.grid_search([2e-5, 2e-4, 2e-3, 2e-2]),
            "epochs": tune_epochs,
        }
    },
    tune_config=tune.TuneConfig(
        metric="eval_loss",
        mode="min",
        scheduler=ASHAScheduler(
            max_t=tune_epochs,
        )
    ),
    run_config=RunConfig(
        checkpoint_config=CheckpointConfig(num_to_keep=1, checkpoint_score_attribute="eval_loss", checkpoint_score_order="min")
    ),
)

# COMMAND ----------

tune_results = tuner.fit()

# COMMAND ----------

# MAGIC %md
# MAGIC We can view the results of the tuning run as a dataframe, and obtain the best result.

# COMMAND ----------

tune_results.get_dataframe().sort_values("eval_loss")

# COMMAND ----------

best_result = tune_results.get_best_result()

# COMMAND ----------

best_result.checkpoint

# COMMAND ----------

result.checkpoint

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predict on test data with Ray AIR <a name="predict"></a>

# COMMAND ----------

# MAGIC %md
# MAGIC You can now use the checkpoint to run prediction with `HuggingFacePredictor`, which wraps around [🤗 Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines). In order to distribute prediction, we use `BatchPredictor`. While this is not necessary for the very small example we are using (you could use `HuggingFacePredictor` directly), it will scale well to a large dataset.

# COMMAND ----------



# COMMAND ----------

from ray.train.huggingface import HuggingFacePredictor
from ray.train.batch_predictor import BatchPredictor
import pandas as pd

predictor = BatchPredictor.from_checkpoint(
    checkpoint=result.checkpoint, #best_result.checkpoint, 
    predictor_cls=HuggingFacePredictor,
    task="ner",
    device=0 if use_gpu else -1,  # -1 is CPU, otherwise device index
)
prediction = predictor.predict(ray_datasets["test"].map_batches(lambda x: x[["sentence"]]), num_gpus_per_worker=int(use_gpu))
prediction.show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Share the model <a name="share"></a>

# COMMAND ----------

# MAGIC %md
# MAGIC To be able to share your model with the community, there are a few more steps to follow.
# MAGIC
# MAGIC We have conducted the training on the Ray cluster, but share the model from the local environment - this will allow us to easily authenticate.
# MAGIC
# MAGIC First you have to store your authentication token from the Hugging Face website (sign up [here](https://huggingface.co/join) if you haven't already!) then execute the following cell and input your username and password:

# COMMAND ----------

from huggingface_hub import notebook_login

notebook_login()

# COMMAND ----------

# MAGIC %md
# MAGIC Then you need to install Git-LFS. Uncomment the following instructions:

# COMMAND ----------

# !apt install git-lfs

# COMMAND ----------

# MAGIC %md
# MAGIC Now, load the model and tokenizer locally, and recreate the 🤗 Transformers `Trainer`:

# COMMAND ----------

from ray.train.huggingface import HuggingFaceCheckpoint

checkpoint = HuggingFaceCheckpoint.from_checkpoint(result.checkpoint)
hf_trainer = checkpoint.get_model(model=AutoModelForSequenceClassification)

# COMMAND ----------

torch.cuda.current_device()

# COMMAND ----------

import torch
import pandas as pd
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from transformers.pipelines import pipeline
from ray.train.huggingface  import HuggingFacePredictor

model_checkpoint = "YOUR_MODEL_CHEKCPOINT"
tokenizer_checkpoint = "YOUR_MODEL_CHEKCPOINT"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

model_config = AutoConfig.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
predictor = HuggingFacePredictor(
               pipeline=pipeline(
               task="ner", model=model, tokenizer=tokenizer, 
                device  = torch.cuda.current_device()))


prompts = pd.DataFrame(
      ["Puneet and Anastasia are flying to London", 
       "My name is Clara and I live in Berkeley, California"], columns=["sentences"])
predictions = predictor.predict(prompts)
predictions

# COMMAND ----------

syntaxer = pipeline(task = 'ner')
syntaxer("My name is Clara and I live in Berkeley, California")

# COMMAND ----------



# COMMAND ----------

syntaxer = pipeline(
               task="ner", model=model, tokenizer=tokenizer, 
                device  = torch.cuda.current_device())
syntaxer("Tanya and puneet setting in a tree kissing and reached Nirvana")

# COMMAND ----------

print(predictions.loc[1][0])

# COMMAND ----------

#Define the components of the model in a dictionary
import mlflow 

task = "text-classification"
architecture = f"{model_checkpoint}-finetuned-cola"
#model = transformers.AutoModelForSequenceClassification.from_pretrained(architecture)
#tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)

transformers_model = {"model": hf_trainer, "tokenizer": tokenizer}

#Log the model components
with mlflow.start_run(run_name="logging_fine_tunned_bert_cased"):
    model_info = mlflow.transformers.log_model(
        transformers_model=transformers_model,
        artifact_path="text_classifier",
        task=task,
    )


# COMMAND ----------

# Load the components as a pipeline
loaded_pipeline = mlflow.transformers.load_model(
    model_info.model_uri, return_type="pipeline"
)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC You can now upload the result of the training to the Hub, just execute this instruction:

# COMMAND ----------

# hf_trainer.push_to_hub()

# COMMAND ----------

# MAGIC %md
# MAGIC You can now share this model with all your friends, family, and favorite pets: they can all load it with the identifier `"your-username/the-name-you-picked"` so for instance:
# MAGIC
# MAGIC ```python
# MAGIC from transformers import AutoModelForSequenceClassification
# MAGIC
# MAGIC model = AutoModelForSequenceClassification.from_pretrained("sgugger/my-awesome-model")
# MAGIC ```

# COMMAND ----------


