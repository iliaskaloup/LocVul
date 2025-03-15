# LocVul - Vulnerability Detection - Line-level Localization of Vulnerabilities
Software vulnerability detection on real-world data. We propose an LLM-based Vulnerability Detection methodology to the line-level of granularity, thereby achieving vulnerability localization.

## Overview of the overall methodology
The high-level overview of the overall methodology that we followed in our study is illustrated in the figure below:
![alt text](https://github.com/iliaskaloup/LocVul/blob/main/LocVul_overview.png?raw=true)

### Replication Package of our research work entitled "LocVul: Line-level Vulnerability Localization based on a Sequence-to-Sequence approach"
This repository is the replication package developed during the conduction of our publication on which we examined the accuracy and cost-effectiveness of a Sequence-to-Sequence approach for extracting vulnerable lines out of functions predicted as vulnerable.

To replicate the analysis and reproduce the results:

~~~
git clone https://github.com/iliaskaloup/LocVul.git
~~~

and navigate to the cloned repository.

Inside the LocVul folder in the main branch, there is a yaml file:
• torchenv.yml file, which is the python-conda virtual environment (venv) that we used.

There are 6 python scripts in the root directory and 3 folders:

• data_mining.py: It downloads the dataset and saves it as dataset.csv in the folder "data".

• vulnDet_pipeline.py: It fine-tunes CodeBERT for function-level vulnerability predictions and then it employs Self-Attention method to localize the vulnerable lines, evaluating this explainability-based approach

• Seq2Seq_vulnDet.py: It fine-tunes CodeT5 for line-level vulnerability detection

• seq2seq_eval.py: It executes both CodeBERT and CodeT5 models to find vulnerable functions and the vulnerable lines inside them, evaluating the line-level performance of the Seq2Seq approach

• statistical_test.py: It performs the Wilcoxon signed-rank statistical test to identify statistical significance between the results of the two approaches

• visualize.py: It produces the bar charts that are presented in the paper to compare LocVul with the Self-Attention approach


• jupyter/ contains the jupyter equivalents of the python scripts

• results/ contains the results of the analysis for all the used evaluation metrics per seed

• preliminary_work/ contains several scripts developed in the beginning of this research and are not actual part of the study described in the paper


### Dataset

To construct the dataset.csv run:
~~~
python data_mining.py
~~~

### Experiment Replication
To train the function-level model (CodeBERT) and evaluate the Self-Attention approach (training and inference) run:
~~~
python vulnDet_pipeline.py –seed=9 –FINE_TUNE=”yes” –model_variation=”microsoft/codebert-base” --checkpoint_dir=”./checkpoints” --sampling=”no” --REMOVE_MISSING_LINE_LABELS=”yes” --EXPLAINER="ATTENTION" --EXPLAIN_ONLY_TP=”no” --sort_by_lines=”yes”
~~~

To evaluate the Self-Attention approach (inference only) run:
~~~
python vulnDet_pipeline.py –seed=9 –FINE_TUNE=”no” –model_variation=”microsoft/codebert-base” --checkpoint_dir=”./checkpoints” --sampling=”no” --REMOVE_MISSING_LINE_LABELS=”yes” --EXPLAINER="ATTENTION" --EXPLAIN_ONLY_TP=”no” --sort_by_lines=”yes”
~~~

To train the line-level model (CodeT5) run:
~~~
python Seq2Seq_vulnDet.py –seed=9 –FINE_TUNE=”yes” –model_variation=”Salesforce/codet5-base” – checkpoint_dir=”./checkpoints_seq2seq”
~~~

To evaluate the Sequence-to-Sequence approach run:
~~~
python seq2seq_eval.py --seed=9 --model_variation="microsoft/codebert-base" --model_variation_seq2seq="Salesforce/codet5-base" --checkpoint_dir="./checkpoints" --checkpoint_dir_seq2seq="./checkpoints_seq2seq" --sampling=”no” --REMOVE_MISSING_LINE_LABELS="yes" --ONLY_TP="no" --sort_by_lines="yes" --SIMILARITY_REPLACEMENT="yes"
~~~

To evaluate the Sequence-to-Sequence approach without the similar line replacement mechanism that handles hallucinations run:
~~~
python seq2seq_eval.py --seed=9 --model_variation="microsoft/codebert-base" --model_variation_seq2seq="Salesforce/codet5-base" --checkpoint_dir="./checkpoints" --checkpoint_dir_seq2seq="./checkpoints_seq2seq" --sampling=”no” --REMOVE_MISSING_LINE_LABELS="yes" --ONLY_TP="no" --sort_by_lines="yes" --SIMILARITY_REPLACEMENT="no"
~~~

### Acknowledgements

Special thanks to HuggingFace for providing the transformers libary.

Special thanks to the paper entitled "A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries" for providing the Big-Vul dataset, which was utilized as the basis for our analysis. If you use Big-Vul, please cite:
~~~
J. Fan, Y. Li, S. Wang and T. N. Nguyen, "A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries," 2020 IEEE/ACM 17th International Conference on Mining Software Repositories (MSR), Seoul, Korea, Republic of, 2020, pp. 508-512, doi: 10.1145/3379597.3387501
~~~

### Licence

[MIT License](https://github.com/iliaskaloup/vulnDetection_realScenario/blob/main/LICENSE)

### Citation
To cite this paper:
~~~
lias Kalouptsoglou, Miltiadis Siavvas, Apostolos Ampatzoglou, Dionysios Kehagias, and Alexander Chatzigeorgiou. 2025. LocVul: Line-level Vulnerability Localization based on a Sequence-to-Sequence approach. ACM Trans. Softw. Eng. Methodol.
~~~
