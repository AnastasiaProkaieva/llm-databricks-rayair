# Define and parse arguments.
from dataclasses import dataclass, field
from typing import Optional


from transformers import TrainingArguments

def get_args(config):
  return TrainingArguments(
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

@dataclass 
class ScriptArgumentsTags:
  local_dir: str = field(default='/tmp/ray/hf/job')
  base_model_dir: str  = field(default="bert-base-multilingual-cased", metadata={"help":"XXX"})
  n_gpus: str = field(default="1", metadata={"help":"XXX"})
  num_cpu_cores_per_worker: str = field(default="16", metadata={"help":"XXX"})
  num_gpu_per_worker: str = field(default="2", metadata={"help":"XXX"})
  max_length: str = field(default="512", metadata={"help":"XXX"})
  user_name: str = field(default="databricks_LLMchampion", metadata={"help":"XXX"})


@dataclass
class ScriptArgumentsTrainer:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    output_dir: Optional[str] = field(default='model_name_finetune')
    per_device_train_batch_size: Optional[int] = field(default=32)
    per_device_eval_batch_size: Optional[int] = field(default=8)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-6)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
  
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training. Сhange to true if using v100 or T4"},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training, supported starting from A10"},
    )

    model_name: Optional[str] = field(
        default="bert-base-multilingual-cased",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    # dataset_name: Optional[str] = field(
    #     default="glue",
    #     metadata={"help": "The preference dataset to use."},
    # )

    evaluation_strategy: Optional[str] = field(
        default="epoch", 
        metadata={"help": "Can be also 'steps' "}
    )

    save_strategy: Optional[str] = field(default="no")

    save_steps: int = field(
        default=200, 
        metadata={"help": "Save checkpoint every X updates steps."}
    )

    logging_steps: int = field(
        default=50, 
        metadata={"help": "Log every X updates steps."}
    )

    eval_steps: Optional[str] = field(
        default=50, 
        metadata={"help": "Log every X updates steps."}
    )

    max_seq_length: Optional[int] = field(default=512)
  
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
  
    max_steps: int = field(
        default=1000, 
        metadata={"help": "How many optimizer update steps to take"}
    )
    warmup_ratio: float = field(
        default=0.03, 
        metadata={"help": "Fraction of steps to do a warmup for"}
    )

    disable_tqdm : bool = field(
        default=True,
        metadata={"help": "Enabling the widget to track your progress"},
    )
    push_to_hub: str = field(default=False)
    no_cuda: str = field(
        default=False, 
        metadata={"help": "You need to explicitly set no_cuda if you want CPUs"},
    )
    load_best_model_at_end : str = field(default=False)

    # lr_scheduler_type: str = field(
    # default="constant",
    # metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    # )
    # remove_unused_columns
    # warmup_steps
