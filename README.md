# Periodization

Code for performing automatic periodization of historical text corpora.

## Requirements
* Python libraries: numpy, gensim

## Instructions
Run:
```python
python periodize.py TEXT_DIR FILE_LIST [--workers workers] [--processes processes] [--ext ext]
```
where:
* `TEXT_DIR`: directory with input texts, where all the texts in each initial time period are contained in one file. 
* `FILE_LIST`: a list of ids (in order) and their corresponding file names.
* `workers`: number of workers for training each word embedding model (default: 1).
* `processes`: number of word embedding models to train in parallel (default: 1).
* `ext`: extension of files to include in the periodization (default: txt).

The program will output the merged clusters and distances, in order of best merges found. 


### Acknowledgements 
This project uses a slightly modified version of this [port](https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf) of the word embedding alignment code from [HistWords](https://github.com/williamleif/histwords). 
