import os
import time

import torch
import torch.utils.data as data

try:
    from input.vocab import Vocab
    from input.glove import Glove
    from input.dataset import Dataset
    from input.pixel import Pixel
except:
    from vocab import Vocab
    from glove import Glove
    from dataset import Dataset
    from pixel import Pixel

#train loader
def get_train_data(files, pklpath, glove_file, glove_embdim, batch_size=1, shuffle=True, num_workers=0, pretrained=False, pklexist=False, data_parallel= True, frame_trunc_length=45, spatial=False):
    
    start_time = time.time()
    vocab = Vocab(files)
    vocab.add_begend_vocab()
    vocab.create()
    vocab_time = time.time()

    glove = Glove(glove_file, glove_embdim)
    glove.create(vocab)
    glove_time = time.time()

    if pretrained:
        pixel = Pixel(files, pklpath)
        if not pklexist:
            pixel.create()
            pixel.save()
        else: pixel.load()		
    pixel_time = time.time()

    dataset = Dataset(files)
    dataset.set_flags(mode='train', data_parallel=data_parallel, frame_trunc_length=frame_trunc_length, pretrained=pretrained, spatial=spatial)
    dataset.set_pad_indices(vocab)
    dataset.create(vocab)
    dataset.add_glove_vecs(glove)
    if pretrained: dataset.add_video_vecs(pixel)
    dataset_time = time.time()

    print('Vocab : {0}, Glove : {1}, Pixel : {2}, Dataset : {3}'.format(vocab_time-start_time, glove_time-vocab_time, pixel_time-glove_time, dataset_time-pixel_time))
    dataloader = data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn,  shuffle=shuffle, num_workers=num_workers)
    return dataloader, vocab, glove, dataset.__len__()

#validation loader
def get_val_data(files, pklpath, vocab, glove, batch_size=1, num_workers=0, pretrained=False, pklexist=False, data_parallel= True, frame_trunc_length=45, spatial=False):
    
    if pretrained:
        pixel = Pixel(files, pklpath)
        if not pklexist:
            pixel.create()
            pixel.save()
        else: pixel.load()
   
    dataset = Dataset(files)
    dataset.set_flags(mode='test', data_parallel=data_parallel, frame_trunc_length=frame_trunc_length, pretrained=pretrained, spatial=spatial)
    dataset.set_pad_indices(vocab)
    dataset.create(vocab)
    dataset.add_glove_vecs(glove)
    if pretrained: dataset.add_video_vecs(pixel)

    dataloader = data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, num_workers=num_workers)
    return dataloader

#test loader
def get_test_data(files, pklpath, vocab, glove, batch_size=1, num_workers=0, pretrained=False, data_parallel= True, frame_trunc_length=45):

    if pretrained:
        pixel = Pixel(files, pklpath)
        if not pklexist:
            pixel.create()
            pixel.save()
        else: pixel.load()

    dataset = Dataset(files)
    dataset.set_flags(mode='test', data_parallel=data_parallel, frame_trunc_length=frame_trunc_length, pretrained=pretrained)
    dataset.set_pad_indices(vocab)
    dataset.create(vocab)
    dataset.add_glove_vecs(glove)
    if pretrained: dataset.add_video_vecs(pixel)
    
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
