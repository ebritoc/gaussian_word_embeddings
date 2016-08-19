# gaussian_word_embeddings

C implementation of [Luke Vilnis and Andrew McCallum
<i>Word Representations via Gaussian Embedding</i>, in ICLR 2015](http://arxiv.org/abs/1412.6623)
where each word is represented as a multivariate Gaussian distribution.

### Installing
A GCC compiler is required for the installation. The code is compiled by running 'make'.

### Learning 
Embeddings can be learned by executing
'./learn -train <FILE> [OPTIONS]'
where file is the training corpus. 

Example:
'./learn -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -binary 0 -iter 3'

### Closest words
The 40 closest embeddings to a query word can be displayed with
'./distance <FILE>'
where FILE contains word projections in the binary format.

### Specificity evaluation
By executing
'./distance <FILE>'
where FILE contains word projections in the binary format, the top 100 nearest words are displayed, sorted by descending variance.

### Reformatting of embeddings
Word embeddings in binary format can be converted to readable (text) format with:
'./binary2text <FILE>'
where FILE contains word projections in the binary format. It is also possible to drop the header and/of the covariance matrix and separate the means and the covariance matrices into different files.
