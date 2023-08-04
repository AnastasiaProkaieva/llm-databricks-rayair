# Support Repository for the Distributed Fine Tuning of Large Languages Models on Databricks Lakehouse with Ray AI Runtime

Following the previous series of blog posts published by Databricks (Running DeepSpeed on Databricks and FineTunning with HF) around LLMs and recent announcements about open-sourced LLMs such as DollyV2, Falcon40B, MPT7B, etc. for commercial usage - have exploded the interest in training your own LLM. There are a plethora of tools that can help you to accelerate your data preprocessing, data loading, model training, scoring and finally serving, but they are still spread across various platforms and people are struggling to understand why model X fit perfectly fine but model Z  is causing out of memory and what to do with that. We also often hear that FineTuning takes a significant amount of time, and people want to accelerate training but also inference, and they want to do that seamlessly and coherently without changing their code significantly while moving from one model to another. 

Here we continue this series of blog posts around scaling fine-tuning and scoring the LLM models. This time we are going to use Ray AI Runtime(AIR) for the series of blog posts about how to tune various Large Language Models(LLM’s) on the Databricks Lakehouse Platform from BERT to MPT7B or even LLAMA70B. 

In the first few parts of the series, we are going to fine-tune a few models: the BERT Large model for classical multi-label classification use cases and another one for the Multilingual Named Entity Recognition (Token Classification) model both from the Hugging Face 🤗 Hub on the Databricks Lakhouse Platform, - simply staying “relatively” small models(up to a billion parameters). Our last part will go deeper into scaling LLMs with more than 12B and higher parameters. 

We have tried to make our example model’s family agnostic, and we will specify within the blog when and where you would require to customize your code accordingly. 


Main contributors:
- Anastasia Prokaieva, SSA Databricks
- Puneet Jain, SSA Databricks

If you find any bugs let us know by raising an issue! 
