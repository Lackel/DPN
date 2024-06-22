
# Decoupled Prototypical Network (DPN)
Data and code for paper titled [Generalized Category Discovery with Decoupled Prototypical Network](https://arxiv.org/abs/2211.15115) (AAAI 2023 paper)

*Generalized Category Discovery (GCD)* aims to recognize both known and novel categories from a set of unlabeled data, based on another dataset labeled with only known categories. In this paper, we present a novel model called Decoupled Prototypical Network (DPN). By formulating a bipartite matching problem for category prototypes, DPN can not only decouple known and novel categories to achieve different training objectives effectively, but also align known categories in labeled and unlabeled data to transfer category-specific knowledge explicitly and capture high-level semantics.


## Contents
[1. Data](#data)

[2. Model](#model)

[3. Requirements](#requirements)

[4. Running](#running)

[5. Results](#results)

[6. Thanks](#thanks)

[7. Citation](#citation)

## Data
We performed experiments on three public datasets: [clinc](https://aclanthology.org/D19-1131/), [banking](https://aclanthology.org/2020.nlp4convai-1.5/) and [stackoverflow](https://aclanthology.org/W15-1509/), which have been included in our repository in the data folder ' ./data '.

## Model
Our model mainly contains five steps: Representation Learning, Prototype Learning, Alignment and Decoupling, Semantic-aware Prototypical Learning and EMA Updating.
<div align=center>
<img src="./figures/model.png"/>
</div>

## Requirements
* python==3.8
* pytorch==1.11.0
* transformers==4.19.2
* scipy==1.8.0
* numpy==1.21.6
* scikit-learn==1.1.1
* pytorch-pretrained-bert==0.6.2

## Running
Pre-training, training and testing our model through the bash scripts:
```
sh run.sh
```
You can also add or change parameters in run.sh (More parameters are listed in init_parameter.py)

## Results
<div align=center>
<img src="./figures/visual.png"/>
</div>
It should be noted that the experimental results may be different because of the randomness of clustering when testing even though we fixed the random seeds. So you can run evaluation multiple times to eliminate the effects of clustering randomness.

## Thanks
Some code references the following repositories:
* [DeepAligned](https://github.com/thuiar/DeepAligned-Clustering)

## Citation
If our paper or code is helpful to you, please consider citing our paper:
```
@inproceedings{an2023generalized,
  title={Generalized category discovery with decoupled prototypical network},
  author={An, Wenbin and Tian, Feng and Zheng, Qinghua and Ding, Wei and Wang, QianYing and Chen, Ping},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={11},
  pages={12527--12535},
  year={2023}
}
```
