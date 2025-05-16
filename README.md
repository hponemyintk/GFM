# RelGT: Relational Graph Transformer

Source code for the paper **[Relational Graph Transformer](https://arxiv.org/abs/2505.xxxxx)** 
>by [Vijay Prakash Dwivedi](http://vijaydwivedi.com.np), [Sri Jaladi](https://www.linkedin.com/in/srijaladi), [Yangyi Shen](https://www.linkedin.com/in/yangyi-shen-232514264/), [Federico Lopez](https://fedelopez77.github.io), [Charilaos I. Kanatsoulis](https://sites.google.com/site/harikanats/), [Rishi Puri](https://www.linkedin.com/in/rishi-puri-1726b1147/), [Matthias Fey](https://rusty1s.github.io/#/), [Jure Leskovec](https://cs.stanford.edu/people/jure/)

## Abstract


Relational Deep Learning (RDL) is a promising approach for building state-of-the-art predictive models on multi-table relational data by representing it as a heterogeneous temporal graph. However, commonly used Graph Neural Network models suffer from fundamental limitations in capturing complex structural patterns and long-range dependencies that are inherent in relational data. While Graph Transformers have emerged as powerful alternatives to GNNs on general graphs, applying them to relational entity graphs presents unique challenges: (i) Traditional positional encodings fail to generalize to massive, heterogeneous graphs; (ii) existing architectures cannot model the temporal dynamics and schema constraints of relational data; (iii) existing tokenization schemes lose critical structural information. Here we introduce the Relational Graph Transformer (RelGT), the first graph transformer architecture designed specifically for relational tables. RelGT employs a novel multi-element tokenization strategy that decomposes each node into five components (features, type, hop distance, time, and local structure), enabling efficient encoding of heterogeneity, temporality, and topology without expensive precomputation. Our architecture combines local attention over sampled subgraphs with global attention to learnable centroids, incorporating both local and database-wide representations. Across 21 tasks from the RelBench benchmark, RelGT consistently matches or outperforms GNN baselines by up to 18%, establishing Graph Transformers as a powerful architecture for Relational Deep Learning.


### Code releasing soon
Email: ```vdwivedi@cs.stanford.edu```
