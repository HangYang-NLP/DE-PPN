# Parallel Prediction for Document-level Event Extraction


Table of Contents
=================
  * [Overview](#overview)
  * [Data](#data)
  * [Reference](#reference)

## Overview
Code for the paper ["Document-level Event Extraction via Parallel Prediction Networks"](https://aclanthology.org/2021.acl-long.492/).

<p align="center">
  <img src="./overview.png" alt="Photo" style="width="100%;"/>
</p>

## Dependencies 
- pytorch=1.6 
- transformers=3.1.0
- tensorboardX
- numpy
- tqdm

## Datasets
- ChFinAnn (Access from [https://github.com/dolphin-zs/Doc2EDAG/blob/master/Data.zip]. Data preprocessing, sentence-level extraction and evaluation metrics following Doc2EDAG[https://github.com/dolphin-zs/Doc2EDAG]).

## Reference
```
@inproceedings{yang-etal-2021-document,
    title = "Document-level Event Extraction via Parallel Prediction Networks",
    author = "Yang, Hang  and
      Sui, Dianbo  and
      Chen, Yubo  and
      Liu, Kang  and
      Zhao, Jun  and
      Wang, Taifeng",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.492",
    doi = "10.18653/v1/2021.acl-long.492",
    pages = "6298--6308",
    abstract = "Document-level event extraction (DEE) is indispensable when events are described throughout a document. We argue that sentence-level extractors are ill-suited to the DEE task where event arguments always scatter across sentences and multiple events may co-exist in a document. It is a challenging task because it requires a holistic understanding of the document and an aggregated ability to assemble arguments across multiple sentences. In this paper, we propose an end-to-end model, which can extract structured events from a document in a parallel manner. Specifically, we first introduce a document-level encoder to obtain the document-aware representations. Then, a multi-granularity non-autoregressive decoder is used to generate events in parallel. Finally, to train the entire model, a matching loss function is proposed, which can bootstrap a global optimization. The empirical results on the widely used DEE dataset show that our approach significantly outperforms current state-of-the-art methods in the challenging DEE task. Code will be available at https://github.com/HangYang-NLP/DE-PPN.",
}
```