# Databricks notebook source
# MAGIC %md 
# MAGIC Uncomment and run the following line in order to install all the necessary dependencies (this notebook is being tested with `transformers==4.19.1`):

# COMMAND ----------



# COMMAND ----------

# MAGIC %pip install ray[default,rllib,tune]==2.5.0 mlflow==2.3 protobuf==3.19.0

# COMMAND ----------

# MAGIC %md
# MAGIC # Fine-tune a 🤗 Transformers model
# MAGIC
# MAGIC
# MAGIC In this notebook, we will see how to fine-tune one of the [🤗 Transformers](https://github.com/huggingface/transformers) model to a text classification task of the [GLUE Benchmark](https://gluebenchmark.com/). We will be running the training using [Ray AIR](https://docs.ray.io/en/latest/ray-air/getting-started.html).
# MAGIC
# MAGIC You can change those two variables to control whether the training (which we will get to later) uses CPUs or GPUs, and how many workers should be spawned. Each worker will claim one CPU or GPU. Make sure not to request more resources than the resources present!
# MAGIC
# MAGIC By default, we will run the training with one GPU worker.
# MAGIC
# MAGIC
# MAGIC This notebook is built to run on any of the classifcation tasks, with any model checkpoint from the [Model Hub](https://huggingface.co/models) as long as that model has a version with a classification head. Depending on your model and the GPU you are using, you might need to adjust the batch size to avoid out-of-memory errors. Set those three parameters, then the rest of the notebook should run smoothly.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Configurations 
# MAGIC
# MAGIC Move them into JSON or YML 

# COMMAND ----------

from utils import get_cluster_spec

task = "cola" # Taken a task from the HF Bebchmarking 
batch_size = 64
use_gpu = True  # set this to False to run on CPUs
#num_workers = 4  # set this to number of GPUs/CPUs you want to use, Configure based on the total gpus across the worker node

model_checkpoint = "bert-base-multilingual-cased"
pretrained_model_name_or_path = model_checkpoint #"EleutherAI/pythia-12b"

max_length = 1024
gradient_checkpointing = True
DEFAULT_SEED = 42
seed = DEFAULT_SEED 

num_labels=2
metric_name = 'matthews_correlation'
validation_key = 'validation'

  
catalog_schema = "ap.llm"
#username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
username = "databricks_LLMchampion"
#experiment_location = f"/Users/{username}/hf_multi-gpu"
experiment_location = f"/Shared/ray_demo_mlflow/{username}/hf_multi-gpu"

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook is based on [an official 🤗 notebook - "How to fine-tune a model on text classification"](https://github.com/huggingface/notebooks/blob/6ca682955173cc9d36ffa431ddda505a048cbe80/examples/text_classification.ipynb). The main aim of this notebook is to show the process of conversion from vanilla 🤗 to [Ray AIR](https://docs.ray.io/en/latest/ray-air/getting-started.html) 🤗 without changing the training logic unless necessary.
# MAGIC
# MAGIC In this notebook, we will:
# MAGIC 1. [Set up Ray Spark on the Databricks Platform](#setup)
# MAGIC 2. [Load the dataset from Delta into the Hugging Face data loader ](#load)
# MAGIC 3. [Preprocess the dataset with Ray AIR](#preprocess)
# MAGIC 4. [Run the training(fine tunning) with Ray AIR](#train)
# MAGIC 5. [Track and log your LLM with MLFlow ](# Logging with MlFLow)
# MAGIC 6. [Predict on test data with Ray AIR](#predict)
# MAGIC 7. [Serve your model with Real-Time Endpoint on Databricks ](#serve)
# MAGIC 8. [Optionally, share the model with the community](#share)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### LIbraries General 

# COMMAND ----------

import os
import re
import json
import logging
import subprocess
import requests

import numpy as np
import pandas as pd
import mlflow

from pathlib import Path
from functools import partial
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, field
from dataclasses import asdict 

from databricks.sdk.runtime import *

import torch
import evaluate

from huggingface_hub import snapshot_download
from datasets import load_dataset, load_from_disk, load_metric, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    BertTokenizer, 
    BertModel,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed,
    default_data_collator,
    DataCollatorWithPadding
)

from datasets import Dataset,DatasetDict


# COMMAND ----------

# MAGIC %md 
# MAGIC #### Custom Libraries 

# COMMAND ----------

from utils import get_or_create_experiment, get_cluster_spec
# get or create experiment
get_or_create_experiment(f"/Shared/ray_demo_mlflow/{username}/hf_multi-gpu")

num_cpu_cores_per_worker, num_gpu_per_worker =  get_cluster_spec() # total cpu's present in each node # total gpu's present in each node
resource_per_worker_int = num_cpu_cores_per_worker/num_gpu_per_worker -2

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Libraries Ray 

# COMMAND ----------

import ray
import ray.data
from ray.data.preprocessors import BatchMapper

from ray.air import session
import ray.util.scheduling_strategies
from ray.train.huggingface import HuggingFaceTrainer
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.air.config import ScalingConfig

from ray.data.preprocessors import Chain
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster, MAX_NUM_WORKER_NODES

from ray.train.huggingface import HuggingFaceTrainer
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback


# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the dataset

# COMMAND ----------

# MAGIC %md
# MAGIC We will use the [🤗 Datasets](https://github.com/huggingface/datasets) library to download the data and get the metric we need to use for evaluation (to compare our model to the benchmark). This can be easily done with the functions `load_dataset` and `load_metric`.
# MAGIC
# MAGIC As Ray AIR doesn't provide integrations for 🤗 Datasets yet,
# MAGIC You have a few options:
# MAGIC - simply run the normal 🤗 Datasets code to load the dataset from the Hub.
# MAGIC - use [Spark HF integration](https://huggingface.co/docs/datasets/use_with_spark#caching) - we are going to use that one 
# MAGIC
# MAGIC
# MAGIC The `dataset` object itself is [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict), which may contain one key for the training, validation, and test set  - in our case we already splitted the data.

# COMMAND ----------

from utils import read_delta_hf

train_dataset = read_delta_hf(table_name = f"{catalog_schema}.training_glue_cola")
val_dataset = read_delta_hf(table_name = f"{catalog_schema}.validation_glue_cola")
test_dataset = read_delta_hf(table_name = f"{catalog_schema}.test_glue_cola")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Load metrics for the dataset 

# COMMAND ----------

# MAGIC %md
# MAGIC We will also need the metric. In order to avoid serialization errors, we will load the metric inside the training workers later. Therefore, now we will just define the function we will use.
# MAGIC
# MAGIC The metric is an instance of [`datasets.Metric`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Metric).

# COMMAND ----------

def load_metric_fn():
  return evaluate.load('glue', task)

metric = load_metric_fn()
metric

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessing the data with Ray AIR 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Setup the Ray Cluster 

# COMMAND ----------


try: 
  shutdown_ray_cluster()
except:
  print("No Ray cluster is initiated")

RAY_LOG_DIR = "/tmp/ray/logs"
# Start the ray cluster
setup_ray_cluster(
  num_worker_nodes=MAX_NUM_WORKER_NODES,
  num_cpus_per_node=int(num_cpu_cores_per_worker),
  num_gpus_per_node=int(num_gpu_per_worker),
  collect_log_to_path=RAY_LOG_DIR #ray_collected_logs
  )

ray.init(address='auto')

num_workers = (ray.cluster_resources()["CPU"]/num_cpu_cores_per_worker)

print(ray.cluster_resources())

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Preprocess your input data if necessary 

# COMMAND ----------

# MAGIC %md
# MAGIC Before we can feed those texts to our model, we need to preprocess them. This is done by a 🤗 Transformers' `Tokenizer`, which will (as the name indicates) tokenize the inputs (including converting the tokens to their corresponding IDs in the pretrained vocabulary) and put it in a format the model expects, as well as generate the other inputs that model requires.
# MAGIC
# MAGIC To do all of this, we instantiate our tokenizer with the `AutoTokenizer.from_pretrained` method, which will ensure that:
# MAGIC
# MAGIC - we get a tokenizer that corresponds to the model architecture we want to use,
# MAGIC - we download the vocabulary used when pretraining this specific checkpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC We pass along `use_fast=True` to the call above to use one of the fast tokenizers (backed by Rust) from the 🤗 Tokenizers library. Those fast tokenizers are available for almost all models, but if you got an error with the previous call, remove that argument. <br>
# MAGIC
# MAGIC For Ray AIR, instead of using 🤗 Dataset objects directly, we will convert them to [Ray Datasets](https://docs.ray.io/en/latest/data/dataset.html). Both are backed by Arrow tables, so the conversion is straightforward. We will use the built-in `ray.data.from_huggingface` function.<br>
# MAGIC
# MAGIC We can then write the function that will preprocess our samples. We just feed them to the `tokenizer` with the argument `truncation=True`. This will ensure that an input longer than what the model selected can handle will be truncated to the maximum length accepted by the model.
# MAGIC
# MAGIC We use a `BatchMapper` to create a Ray AIR preprocessor that will map the function to the dataset in a distributed fashion. It will run during training and prediction.
# MAGIC

# COMMAND ----------


def preprocess_func(text: pd.DataFrame):
  if len(text.columns) == 1:
        return text
  text = text.to_dict("list")
  ret = tokenizer(text["sentence"], padding='max_length' ,truncation=True) 
  ret = {**text, **ret}
  return pd.DataFrame.from_dict(ret)


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
batch_encoder = BatchMapper(preprocess_func, batch_format="pandas")


# COMMAND ----------

ray_train_datasets = ray.data.from_huggingface(train_dataset)
ray_val_datasets = ray.data.from_huggingface(val_dataset)
ray_test_datasets = ray.data.from_huggingface(test_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-tuning the model with Ray AIR 

# COMMAND ----------

# MAGIC %md
# MAGIC Now that our data is ready, we can download the pretrained model and fine-tune it.
# MAGIC
# MAGIC Since we focus on the sentence classification, we use the `AutoModelForSequenceClassification` class.
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

# from transformers import HfArgumentParser
# parser = HfArgumentParser(ScriptArgumentsTrainer())
# script_args = parser.parse_known_args()[0]

# COMMAND ----------

from training_arguments import ScriptArgumentsTrainer, ScriptArgumentsTags
config = asdict(ScriptArgumentsTrainer())

def load_metric_fn():
  return evaluate.load("glue", "cola")

def trainer_init_per_worker(train_dataset, eval_dataset = None, **config):

    set_seed(seed)
    # Use the actual number of CPUs assigned by Ray
    os.environ["OMP_NUM_THREADS"] = str(
        session.get_trial_resources().bundles[-1].get("CPU", 1)
    )
    # Enable tf32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    metric = load_metric_fn()

    def compute_metrics(p: EvalPrediction):
      preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
      preds =  np.argmax(preds, axis=1)
      result = metric.compute(predictions=preds, references=p.label_ids)
      if len(result) > 1:
          result["combined_score"] = np.mean(list(result.values())).item()
      return result

    print("Loading Training Arguments") 
    args = TrainingArguments(
            output_dir=config.get("output_dir", "model_bert_finetune"), #for now it has to be 1 name no /  
            learning_rate=config.get("learning_rate", 2e-5),
            per_device_train_batch_size=config.get("per_device_train_batch_size"),
            per_device_eval_batch_size=config.get("per_device_eval_batch_size"),
            num_train_epochs=config.get("epochs", 10),
            disable_tqdm=config.get("disable_tqdm"),
            logging_strategy=config.get("evaluation_strategy"),
            logging_steps=config.get("logging_steps"),
            evaluation_strategy=config.get("evaluation_strategy"),
            eval_steps=config.get("eval_steps"),
            save_strategy=config.get("save_strategy"),
            save_steps=config.get("save_steps"),
            fp16=config.get("fp16"), # change to true if using v100 or T4
            bf16=config.get("bf16",True),# change to false if using v100 or T4 (supported starting from A10)
            push_to_hub=config.get("push_to_hub"),
            no_cuda=config.get("no_cuda"),  # you need to explicitly set no_cuda if you want CPUs
        )

    print("Loading model and it's tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    print("Model loaded")
    print("Train data size: %d", len(train_dataset))
    print("Test data size: %d", len(eval_dataset))
    
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
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

!ls /dbfs/Users/databricks_LLMchampion/hf/

# COMMAND ----------

!ls /local_disk0/tmp/

# COMMAND ----------

TAGS = asdict(ScriptArgumentsTags())

trainer = HuggingFaceTrainer(
    trainer_init_per_worker=trainer_init_per_worker,
    # use this one if you want to change your default configs from the HF Config 
    trainer_init_config={
        "lr" : 1e-4, # per device
        "per_device_train_batch_size" : 32,
        "per_device_eval_batch_size" : 8,
        "save_strategy" : "epoch",
        "evaluation_strategy" : "epoch",
        "warmup_steps" : 25,
        "disable_tqdm" : True,
        "remove_unused_columns" :False,
        "epochs": 10
        },
    scaling_config=ScalingConfig(
                    num_workers=2, 
                    use_gpu=use_gpu,
                    resources_per_worker={"GPU": 1, "CPU": resource_per_worker_int}
                    ),
    run_config=RunConfig(
        local_dir = f"/local_disk0/Users/{username}/hf/", # /local_disk0/Users/{username}/hf/; f"/dbfs/Users/{username}/hf/"
        callbacks=[
          MLflowLoggerCallback(
                    experiment_name=experiment_location,
                    tags=TAGS, 
                    save_artifact=True#True will not work with dbfs -> use local_disk0
                    )
                  ],
        #sync_config=sync_config,  
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="eval_loss",
            checkpoint_score_order="min",
            ),
      verbose = 0
        ),
    datasets={
        "train": ray_train_datasets,
        "evaluation": ray_val_datasets,
        },
 
    preprocessor=batch_encoder,
)


# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we call the `fit` method to start training with Ray AIR. We will save the `Result` object to a variable so we can access metrics and checkpoints.

# COMMAND ----------

result = trainer.fit()

# COMMAND ----------

result

# COMMAND ----------

!ls /local_disk0/Users/databricks_LLMchampion/hf/HuggingFaceTrainer_2023-06-21_16-58-01/HuggingFaceTrainer_c7f40_00000_0_2023-06-21_16-58-01/

# COMMAND ----------

# MAGIC %md
# MAGIC You can use the returned `Result` object to access metrics and the Ray AIR `Checkpoint` associated with the last iteration.

# COMMAND ----------

result.metrics_dataframe.head()

# COMMAND ----------

!ls /dbfs/Users/databricks_LLMchampion/hf/HuggingFaceTrainer_2023-06-16_15-33-06/HuggingFaceTrainer_1723b_00000_0_2023-06-16_15-33-06/checkpoint_000004

# COMMAND ----------

!ls /dbfs/Users/databricks_LLMchampion/hf/HuggingFaceTrainer_2023-06-16_15-33-06/HuggingFaceTrainer_1723b_00000_0_2023-06-16_15-33-06/

# COMMAND ----------

result.checkpoint

# COMMAND ----------

result.metrics

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Tune hyperparameters with Ray AIR 

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
        num_samples=1,
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

best_result = tune_results.get_best_result().checkpoint
best_result

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Load model into MlFLow 
# MAGIC
# MAGIC We recommend you log your model while training, but it may happen you forgot, or simply you like to log your model post training. 
# MAGIC Here we are presenting how to log model from the existing checkpoint into MlFLow.
# MAGIC
# MAGIC To log your model while training, you can have a look at this mocking code: 
# MAGIC ```
# MAGIC import mlflow
# MAGIC import torch
# MAGIC from transformers import pipeline, DefaultDataCollator, EarlyStoppingCallback
# MAGIC
# MAGIC with mlflow.start_run(run_name="hugging_face") as run:
# MAGIC
# MAGIC     train_results = trainer.train()
# MAGIC     model = trainer.state.best_model_checkpoint
# MAGIC     #Build our final hugging face pipeline
# MAGIC     classifier = pipeline("pipeline-task-name", model=model, tokenizer = model_def, device_map='auto')
# MAGIC     #log the model to MLFlow
# MAGIC     reqs = mlflow.transformers.get_default_pip_requirements(model)
# MAGIC     mlflow.transformers.log_model(artifact_path="model", transformers_model=classifier, pip_requirements=reqs)
# MAGIC     mlflow.log_metrics(train_results.metrics)
# MAGIC
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC Now, load the model and tokenizer locally, and recreate the 🤗 Transformers `Trainer`:

# COMMAND ----------

from ray.train.huggingface import HuggingFaceCheckpoint

checkpoint_path = "/dbfs/Users/databricks_LLMchampion/hf/HuggingFaceTrainer_2023-06-16_15-33-06/HuggingFaceTrainer_1723b_00000_0_2023-06-16_15-33-06/checkpoint_000004"

checkpoint = HuggingFaceCheckpoint.from_directory(checkpoint_path)
fineT_model = checkpoint.get_model(model=AutoModelForSequenceClassification)
tokenizer = checkpoint.get_tokenizer(tokenizer=AutoTokenizer)

# COMMAND ----------

# MAGIC %md 
# MAGIC You can also use this command if your results are still in the memory: 
# MAGIC `HuggingFaceCheckpoint.from_checkpoint(results.checkpoint)`

# COMMAND ----------

import mlflow 
import transformers

with mlflow.start_run(run_name="Sentiment BERT Fine Tuned") as run:
  #Build our final hugging face pipeline
  task = "text-classification"
  architecture = f"{model_checkpoint}-finetuned-glue"
  sentiment_pipeline = transformers.pipeline(
      task=task,
      model=fineT_model,
      tokenizer=tokenizer,
      framework="pt",
      torch_dtype=torch.bfloat16, 
      device_map="auto"
  )
  #log the model to MLFlow
  reqs = mlflow.transformers.get_default_pip_requirements(fineT_model)
  mlflow.transformers.log_model(artifact_path="model", transformers_model=sentiment_pipeline, pip_requirements=reqs)


# COMMAND ----------

# MAGIC %md 
# MAGIC Register your model and move it into a Staging or Production Phase 

# COMMAND ----------

#Save the model in the registry & move it to Production
model_name_registry = "ray_demo_hf_clf"
model_registered = mlflow.register_model("runs:/"+run.info.run_id+"/model", model_name_registry)
client = mlflow.tracking.MlflowClient()
print("registering model version "+model_registered.version+" as production model")
#Move the model as Production
client.transition_model_version_stage(name = model_name_registry, version = model_registered.version, stage = "Staging", archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md 
# MAGIC Score your model with the mlflow 

# COMMAND ----------

import torch
#Make sure to legerage the GPU when available
model_uri = "models:/ray_demo_hf_clf/Staging"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
pipeline = mlflow.transformers.load_model(model_uri, device=device.index)
pipeline.predict(train_dataset['sentence'])

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Examples how to score your model directly from MLFLow 
# MAGIC
# MAGIC
# MAGIC ```
# MAGIC logged_model = 'runs:/PLACE_YOUR_RUNID_HERE/model'
# MAGIC # Load model as a PyFuncModel.
# MAGIC loaded_model = mlflow.pyfunc.load_model(logged_model)
# MAGIC # Predict on a Pandas DataFrame
# MAGIC loaded_model.predict(data_mock)
# MAGIC
# MAGIC ```
# MAGIC
# MAGIC Or with the SparkUDF (it's not gonna USE GPU)
# MAGIC
# MAGIC ```
# MAGIC from pyspark.sql.functions import struct, col
# MAGIC # Load model as a Spark UDF. Override result_type if the model does not return double values.
# MAGIC loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)
# MAGIC display(df.withColumn('predictions', loaded_model(struct(*map(col, df.columns)))))
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predict with Ray AIR 
# MAGIC
# MAGIC
# MAGIC
# MAGIC You can now use the checkpoint to run prediction with `HuggingFacePredictor`, which wraps around [🤗 Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines). In order to distribute prediction, we use `BatchPredictor`. While this is not necessary for the very small example we are using (you could use `HuggingFacePredictor` directly), it will scale well to a large dataset.

# COMMAND ----------

from ray.train.huggingface import HuggingFaceCheckpoint
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
import os
requirements_path = ModelsArtifactRepository("models:/ray_demo_hf_clf/Staging").download_artifacts(artifact_path="pipeline") # download model requirement from remote registry
print("Your Path for the Artifact is: ", requirements_path)

# COMMAND ----------

from ray.train.huggingface import HuggingFacePredictor
from ray.train.batch_predictor import BatchPredictor

ray_train_datasets = ray.data.from_huggingface(train_dataset)

checkpoint = HuggingFaceCheckpoint.from_directory(requirements_path)
predictor = BatchPredictor.from_checkpoint(checkpoint=checkpoint, 
                                           predictor_cls=HuggingFacePredictor,
                                           task="text-classification",
                                           device=0 if use_gpu else -1,
                                           num_gpus_per_worker=1
                                           )
prediction = predictor.predict(ray_train_datasets.map_batches(lambda x: x[["sentence"]]), num_gpus_per_worker=int(use_gpu))

# COMMAND ----------

prediction.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Serve your model 

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Before MLFlow 2.4+ for model Serving 
# MAGIC As of the moment, model serving yet does not work with the new MLFLow Transformer Flavor - in order to serve your model you would need to wrap it under the PyFunc. 
# MAGIC This will soon be changed and you can directly use mlflow.transformers.log_model - we are going to update that part when it's available.

# COMMAND ----------

import mlflow
import torch
import pandas as pd
from transformers import pipeline

pipeline_artifact_name = "pipeline"
pipeline_output_dir = '/local_disk0/model_artifacts/sentiment-analysis'
sentiment_pipeline.save_pretrained(pipeline_output_dir)

class SentimentAnalysisPipelineModel(mlflow.pyfunc.PythonModel):
  from transformers import pipeline
  from mlflow.models import ModelSignature
  from mlflow.types import DataType, Schema, ColSpec

  signature = ModelSignature(inputs=Schema([ColSpec(name="sentence", type=DataType.string)]),
                             outputs=Schema([ColSpec(name="label", type=DataType.string), ColSpec(name="score", type=DataType.double)]))

 
  def load_context(self, context):
    # model = PreTrainedModel.from_pretrained(context.artifacts[pipeline_artifact_name])
    # tokenizer = PreTrainedTokenizer.from_pretrained(context.artifacts[pipeline_artifact_name])
    self.pipeline = pipeline("sentiment-analysis", context.artifacts[pipeline_artifact_name])
    
  def predict(self, context, model_input): 
    text = model_input[model_input.columns[0]].to_list()
    pipe = self.pipeline(text)
    result = [{"label": prediction['label'], "score": prediction['score']} for prediction in pipe]
    return result

# Add Pip Requirements Env list : 
pip_reqs = mlflow.transformers.get_default_pip_requirements(fineT_model)  
# register the model with mlflow
with mlflow.start_run(run_name="Sentiment BERT Fine Tuned"):
  pymodel = SentimentAnalysisPipelineModel()
  model_info = mlflow.pyfunc.log_model(
    artifact_path="model", 
    python_model=pymodel, 
    signature=pymodel.signature, 
    artifacts={pipeline_artifact_name: pipeline_output_dir},
    pip_requirements=pip_reqs
    )


# COMMAND ----------

# MAGIC %md 
# MAGIC If you want to verify that your pipeline works before logging it into mlflow, you can use this code to test your Class 
# MAGIC
# MAGIC ```
# MAGIC import pandas as pd
# MAGIC from mlflow.pyfunc import PythonModel, PythonModelContext
# MAGIC from transformers import pipeline
# MAGIC
# MAGIC # test the model works locally 
# MAGIC data_mock = pd.DataFrame(train_dataset.to_pandas().iloc[:10,:]["sentence"])
# MAGIC
# MAGIC pymodel = SentimentAnalysisPipelineModel()
# MAGIC context = PythonModelContext({pipeline_artifact_name: pipeline_output_dir})
# MAGIC pymodel.load_context(context)
# MAGIC
# MAGIC pymodel.predict(context, pd.DataFrame(data_mock))
# MAGIC
# MAGIC ```

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'YOUR MODEL SERVING ENDPOINT URL'
  headers = {'Authorization': f'Bearer YOUR PAT TOKEN', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  print(data_json)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

score_model(data_mock)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Share the model (optional)
# MAGIC
# MAGIC To be able to share your model with the community, there are a few more steps to follow.
# MAGIC
# MAGIC We have conducted the training on the Ray cluster, but share the model from the local environment - this will allow us to easily authenticate.
# MAGIC
# MAGIC First you have to store your authentication token from the Hugging Face website (sign up [here](https://huggingface.co/join) if you haven't already!) then execute the following cell and input your username and password.
# MAGIC
# MAGIC ```
# MAGIC from huggingface_hub import notebook_login
# MAGIC notebook_login()
# MAGIC ```
# MAGIC Then you need to install Git-LFS. Uncomment the following instructions: `!apt install git-lfs`
# MAGIC
# MAGIC You can now upload the result of the training to the Hub, just execute this instruction:`hf_trainer.push_to_hub()`
# MAGIC
# MAGIC You can now share this model with all your friends, family, and favorite pets: they can all load it with the identifier `"your-username/the-name-you-picked"` so for instance:
# MAGIC
# MAGIC ```python
# MAGIC from transformers import AutoModelForSequenceClassification
# MAGIC
# MAGIC model = AutoModelForSequenceClassification.from_pretrained("sgugger/my-awesome-model")
# MAGIC ```
# MAGIC
