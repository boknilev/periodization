# -*- coding: utf-8 -*-
"""
Periodize a collection of texts arranged by time

Usage: periodize.py TEXT_DIR FILE_LIST [--ext ext] [--verbose] [--workers workers] [--processes processes]

Arguments:
  TEXT_DIR        Folder with text files
  FILE_LIST       File containing ids and text file base names
  
Options:
  -h, --help             show this help message
  --ext ext              File extension [Default: txt]
  -v --verbose           Print a lot of logging information
  --workers workers      Number of threads for training each word embedding model [Default: 1]
  --processes processes  Number of processes to train models in parallel 
"""
  

"""
Algorithm:
1. Given list of texts in consecutive periods
2. Compute a word embedding model for every text
3. Compute distance between each two consecutive periods
4. Merge texts from most similar two periods and recompute word embedding model for merged corpus
5. Repeat from 3 until there's only one time period
"""

from docopt import docopt
import numpy as np
import subprocess, os
from multiprocessing import Pool
from gensim.models import word2vec
from gensim_word2vec_procrustes_align import smart_procrustes_align_gensim


def periodize(text_dir, ids_files, ext='txt', workers=1, processes=1):
    """
    Periodize a historical corpus of texts 
    
    text_dir: folder containing text files
    ids_files: list of (id, f) pairs, where ids are in chronical order and  
             each file f contains a text from one period in consecutive order
    
    return clustering
    """
    
    print 'periodizing'
    cur_ids_files = ids_files[:]
    print 'training base models'
    cur_models = train_base_models(text_dir, ids_files, workers, processes)

    merged, merged_dists = [], []
    while len(cur_ids_files) > 1:
        print 'len(cur_ids_files):', len(cur_ids_files)
        cur_ids_files, cur_merged, cur_merged_dist, cur_models = find_best_merge(text_dir, cur_ids_files, cur_models, ext, workers, processes)
        merged.append(cur_merged)
        merged_dists.append(cur_merged_dist)
    
    # TODO: convert merged to clustering
    clustering = merged
    return clustering, merged_dists


def train_base_models(text_dir, ids_files, workers=1, processes=1):
    """
    Train models for the base periods


    ids_files: list of (id, f) pairs, where ids are in chronical order and  
             each file f contains a text from one period in consecutive order
    workers: number of of workers for training a single model
    processes: number of processes to train models in parallel 

    return emb_models: list of trained word embedding models
    """

    if processes > 1:
        pool_size = min(len(ids_files), processes)
        print 'starting pool of size ' + str(pool_size)
        pool = Pool(pool_size, maxtasksperchild=1)
        pool_args = [[os.path.join(text_dir, id_file[1]), workers] for id_file in ids_files]
        try:
            emb_models = pool.map(train_emb_model_wrapper, pool_args)
        except KeyboardInterrupt:
            print "Caught KeyboardInterrupt, terminating workers"
            pool.terminate()
            pool.join()
        else:
            pool.close()
            pool.join()
    else:    
        emb_models = [train_emb_model(os.path.join(text_dir, id_file[1]), workers=workers) for id_file in ids_files]

    return emb_models


def find_best_merge(text_dir, ids_files, emb_models, ext='txt', workers=1, processes=1):
    """
    Find the best pair to merge
    
    ids_files: list of (id, f) pairs, where ids are in chronical order and  
             each file f contains a text from one period in consecutive order
    emb_models: list of word embedding models corresponding to the files in ids_files
    ext: file extension
    workers: number of of workers for training a single model
    processes: number of processes to train models in parallel 
    
    return (merged_ids_files, merged_id, merged_emb_models) tuple, where: 
            merged_ids_files is a list of (id, f) pairs after merging the best pair
            merged_id is the id1-id2 id of the merged pair
            merged_emb_models is a list of models after training model on the merged best pair
    """
    
    print 'finding best merge'
    print 'ids_files:', ids_files
    
    # compute distances
    dists = [compute_distance(emb_models[i], emb_models[i+1]) for i in xrange(len(emb_models)-1)]
    
    # find best pair
    best = np.argmin(dists)
    
    # merge best pair
    merged_id = ids_files[best][0] + '-' + ids_files[best+1][0]
    merged_file = merged_id + '.' + ext
    print 'merged file:', merged_file
    file1 = os.path.join(text_dir, ids_files[best][1])
    file2 = os.path.join(text_dir, ids_files[best+1][1])
    file12 = os.path.join(text_dir, merged_file)
    #print file12
    #print merged_file
    subprocess.Popen(['cat ' + file1 + ' ' + file2 + ' > ' + file12], shell=True).wait()
    merged_model = train_emb_model(file12, workers=workers)
    merged_dist = dists[best]

    # plug merged file into file list
    merged_ids_files, merged_emb_models = [], []
    for i in xrange(best):
        merged_ids_files.append(ids_files[i])
        merged_emb_models.append(emb_models[i])
    merged_ids_files.append((merged_id, merged_file))
    merged_emb_models.append(merged_model)
    for i in xrange(best + 2, len(ids_files)):
        merged_ids_files.append(ids_files[i])
        merged_emb_models.append(emb_models[i])
    
    return merged_ids_files, merged_id, merged_dist, merged_emb_models
    

def train_emb_model_wrapper(args):
    return train_emb_model(*args)

    
def train_emb_model(filename, workers=1):
    """ Train word embedding model
    
    filename: name of text file
    
    return emb_model: a gensim word embedding model
    """
    
    print 'training embedding model on file:', filename
    
    sentences = word2vec.LineSentence(filename)
    emb_model = word2vec.Word2Vec(sentences, size=100, sg=1, workers=workers)
    emb_model.init_sims()
    return emb_model
    

def compute_distance(emb_model1, emb_model2):
    """ Compute distance between two word embedding models
    
    emb_model1: a trained gensim Word2Vec model
    emb_model2: a trained gensim Word2Vec model
    
    return distance
    """
    
    print 'computing distance'
#    print emb_model1
#    print emb_model2
#    print emb_model1
    
    aligned_emb_model2 = smart_procrustes_align_gensim(emb_model1, emb_model2)
    distance = np.linalg.norm(emb_model1.wv.syn0norm - aligned_emb_model2.wv.syn0norm, ord='fro')
    print 'distance:', distance
    return distance


def visualize_clustering(clustering, dists):
    
    print 'clustering:'
    print clustering
    print 'dists:'
    print dists


def run(text_dir, file_list, ext='txt', workers=1, processes=1):
    
    ids_files = []
    with open(file_list) as f:
        for line in f:
            if len(line.strip().split()) == 2:
                ids_files.append(line.strip().split())
    
    clustering, dists = periodize(text_dir, ids_files, ext=ext, workers=workers, processes=processes)
    visualize_clustering(clustering, dists)    
   
   
if __name__ == '__main__':
    args = docopt(__doc__)
    print args
    
    if args['--verbose']:
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)    

    ext = args['--ext'] if args['--ext'] else 'txt'
    workers = int(args['--workers']) if args['--workers'] else 1
    processes = int(args['--processes']) if args['--processes'] else 1
    
    run(args['TEXT_DIR'], args['FILE_LIST'], ext=ext, workers=workers, processes=processes)
    
    
    
    
    
    
    
