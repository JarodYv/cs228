# cs228 - 概率图模型

本项目中的学习笔记是[斯坦福大学CS288](https://cs228.stanford.edu/)-概率图模型[^1]课程的简明导论。

本课程从基础知识开始介绍概率图形模型，最后从第一性原理出发解释[变分自动编码器](conclusion/variational_autoencoder.md)，这是一个重要的概率模型，也是深度学习中最具影响力的前沿成果之一。

## 目录

* [预备](preliminaries/README.md)
    * [简介](preliminaries/introduction.md)
    * [概率论回顾](preliminaries/review_of_probability_theory.md)
    * [真实世界应用案例](preliminaries/examples_of_real-world_applications.md)
* [表示](representation/README.md)
    * [贝叶斯网络](representation/bayesian_networks.md)
    * [马尔可夫随机场](representation/markov_random_fields.md)
* [推理](inference/README.md)
    * [变量消除](inference/variable_elimination.md)
    * [信念传播](inference/belief_propagation.md)
    * [最大后验(MAP)推理](inference/MAP_inference.md)
    * [基于抽样的推理](inference/sampling-based_inference.md)
    * [变分推理](inference/variational_inference.md)
* [学习](learning/README.md)
    * [有向模型学习](learning/learning_in_directed_models.md)
    * [无向模型学习](learning/learning_in_undirected_models.md)
    * [潜变量模型学习](learning/learning_in_latent_variable_models.md)
    * [贝叶斯学习](learning/bayesian_learning.md)
    * [结构学习](learning/structure_learning.md)
* [总结](conclusion/README.md)
    * [变分编码器](conclusion/variational_autoencoder.md) 

[^1]:概率图模型是机器学习的一个分支，研究如何用事物的概率描述和归因真实世界。
