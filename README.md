# Zero-Inflated Exponential Family Embeddings 

## Introduction

This repo implements the embedding models in the 2017 ICML 
[paper](http://proceedings.mlr.press/v70/liu17a/liu17a.pdf) "Zero-Inflated Exponential Family Embeddings" 

Zero-Inflated Exponential Family Embedding (ZIE) model uses zero-inflated distributions to fit the conditional probability 
in the embedding model. As a result, ZIE model automatically downweights zeros in the data. Please see the details in the 
[paper](http://proceedings.mlr.press/v70/liu17a/liu17a.pdf). 

## Running the code

`python demo.py`

Note: this repo does not contain any data -- it only use some random data to show how to use the code. The code requires 
`numpy`, `scipy`, and `tensorflow`.

## Misc

If you have any questions, please contact the Li-Ping Liu (liping.liulp at gmail).

If you have used the code in your work, please cite: 

@inproceedings{zie17,
  title =    {Zero-Inflated Exponential Family Embeddings},
  author =   {Li-Ping Liu and David M. Blei},
  booktitle ={Proceedings of the 34th International Conference on Machine Learning},
  pages =    {2140--2148},
  year =     {2017},
  editor =   {Doina Precup and Yee Whye Teh},
  volume =   {70},
  series =   {Proceedings of Machine Learning Research},
  address =  {International Convention Centre, Sydney, Australia},
  month =    {06--11 Aug},
  publisher ={PMLR},
}
