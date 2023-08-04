import mlflow
import requests
from databricks.sdk.runtime import *

from datasets import Dataset
from datasets import DatasetDict

def get_or_create_experiment(experiment_location: str) -> None:
 
  if not mlflow.get_experiment_by_name(experiment_location):
    print(f"Experiment does not exist. Creating experiment {experiment_location}")
    
    mlflow.create_experiment(experiment_location)

  print(f"Setting the experiment {experiment_location}")  
  mlflow.set_experiment(experiment_location)


def get_cluster_spec():
  '''
  Function to get the num_cpus and num_gpus of a Databricks cluster
  '''
  API_URL = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
  TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
 
  response = requests.get(
    API_URL + '/api/2.0/clusters/list-node-types',
    headers={"Authorization": "Bearer " + TOKEN},
  )
 
  if response.status_code == 200:
    details = [(node_details['num_cores'] , node_details['num_gpus'])  for node_details in response.json()['node_types'] \
    if node_details['node_type_id'] == spark.conf.get('spark.databricks.workerNodeTypeId') ][0]
    print('num_cpus_per_node:',details[0] )
    print('num_gpus_per_node:',details[1])
    return details[0] , details[1]
  else:
    print("Error: %s: %s" % (response.json()["error_code"], response.json()["message"]))
    return None, None

def read_delta_hf(table_name, cache_dir_name = "/dbfs/tmp/run/hf"):
  return (Dataset.from_spark(
              spark.read.table(table_name), 
              cache_dir=cache_dir_name
              ).remove_columns(column_names=["idx"])
              )
