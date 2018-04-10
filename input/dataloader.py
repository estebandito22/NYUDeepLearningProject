import os

import torch
import torch.utils.data as data

#from input.vocab import Vocab
#from input.glove import Glove
#from input.dataset import Dataset
from vocab import Vocab
from glove import Glove
from dataset import Dataset


#train loader
def get_train_data(files, glove_file, glove_embdim, batch_size=1, shuffle=True):
    vocab = Vocab(files)
    vocab.create()
    vocab.add_begend_vocab()

    glove = Glove(glove_file, glove_embdim)
    glove.create(vocab)

    dataset = Dataset(files)
    dataset.set_pad_indices(vocab)
    dataset.create(vocab)
    dataset.add_glove_vecs(glove)

    dataloader = data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn,  shuffle=shuffle)
    return dataloader, vocab, glove, dataset.__len__()

#validation loader
def get_val_data(files, vocab, glove, batch_size=1):
    dataset = Dataset(files)
    dataset.set_pad_indices(vocab)
    dataset.create(vocab)
    dataset.add_glove_vecs(glove)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return dataloader

#test loader
def get_test_data(files, vocab, glove, batch_size=1):
    dataset = Dataset(files)
    dataset.set_mode('test')
    dataset.set_pad_indices(vocab)
    dataset.create(vocab)
    dataset.add_glove_vecs(glove)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return dataloader


if __name__=='__main__':
    file_names = [('MSRVTT/captions.json', 'MSRVTT/trainvideo.json', 'MSRVTT/Frames')]
    cur_dir = os.getcwd()
    files = [[os.path.join(cur_dir,filetype) for filetype in file] for file in file_names]
    glove_dir = '../../SQUAD/glove/'
    glove_filename = 'glove.6B.50d.txt'
    glove_filepath = os.path.join(glove_dir, glove_filename)
    glove_embdim = 50

    train_dataloader, vocab, glove = get_train_data(files, glove_filepath, glove_embdim, 2)

    '''for batch in train_dataloader:
    	print(batch[1])
    	print(batch[3])
    	print(batch[5])'''

    file_names = [('MSRVTT/captions.json', 'MSRVTT/testvideo.json', 'MSRVTT/Frames')]
    cur_dir = os.getcwd()
    files = [[os.path.join(cur_dir,filetype) for filetype in file] for file in file_names]
    test_dataloader = get_test_data(files, vocab, glove, 2)

    for batch in test_dataloader:
    	print(batch[1])
    	print(batch[3])
    	print(batch[5])


