# Interpoetry

This repository contains the original implementation of the unsupervised poem translation models presented in  
[Interpoetry: Generating Classical Chinese Poems from Vernacular Chinese](link here) (EMNLP 2019). 

A fancy demonstration could be found [here](https://pengshancai.github.io/interpoetry/).

Thanks to FacebookResearch for opensourcing [Phrase-Based & Neural Unsupervised Machine Translation](https://github.com/facebookresearch/UnsupervisedMT) project.


## Dependencies

* Python 3
* [NumPy](http://www.numpy.org/)
* [PyTorch](http://pytorch.org/) (currently tested on version 1.0.1)
* [TQDM](https://github.com/tqdm/tqdm) (4.31.1, for preprocess.py only)

## Download / preprocess data

Quickroutes are provided to save time, you could only download [processed data](https://umass-my.sharepoint.com/:f:/g/personal/zhichaoyang_umass_edu/EtxFUYdWDKtFnEK7Jt1WNnYBn7CR1iwZS5Zd66A1Q84j5Q?e=10VSHv) and unzip in interpoetry folder. Then continue to [Train](https://github.com/whaleloops/interpoetry#train) section. However, if you are interested in detailed steps, please continue reading.

### Vernaculars
Training data are collected from 281 sanwens and fictions written by more than 40 famous Chinese authors (鲁迅, 金庸, 毕淑敏, 余秋雨, 张小娴, 温世仁 etc.). The dataset includes more than 500K sentences.

### Poems
Classical poem data for training are collected from [here](https://github.com/chinese-poetry/chinese-poetry).

### Parallel data (poems and thier translation)
TODO

### Preprocess

Download all raw data [here](https://umass-my.sharepoint.com/:f:/g/personal/zhichaoyang_umass_edu/EjuB2kReBAdBmyojACZ7cYcBdxGvsrcWV3cjhBYkKORrwg?e=ntnXUt)

Run with following commands to generate preprocessed sanwen data.
```
python preprocess.py data/vocab.txt data/sanwen/sanwen sanwen abc nopmpad 7
```

Run with following commands to generate preprocessed poems data.
```
python preprocess.py data/vocab.txt data/jueju7_out abc juejue pmpad 7 
```

Commands to process parallel data are similar. Replace second param to the actual file you would like to process.


## Train

Given binarized monolingual (poem and poem) training data, parallel evaluation (poem and its tranlation) data, you can train the model using the following command:

```
python main.py 

## main parameters
--exp_name test 

## network architecture and parameters sharing
--transformer True 
--n_enc_layers 4 --n_dec_layers 4 --share_enc 2 --share_dec 2 
--share_lang_emb True --share_output_emb True 

## datasets location, denoising auto-encoder parameters, and back-translation directions
--langs 'pm,sw' 
--n_mono -1                                 # number of monolingual sentences (-1 for everything)
--mono_dataset $MONO_DATASET                # monolingual dataset
--para_dataset $PARA_DATASET                # parallel dataset
--mono_directions 'pm,sw' --max_len '70,110' --word_shuffle 2 --word_dropout 0.05 --word_blank 0.1 
--pivo_directions 'sw-pm-sw,pm-sw-pm' 

## pretrained embeddings
--pretrained_emb $PRETRAINED 
--pretrained_out False 

## dynamic loss coefficients
--lambda_xe_mono '0:1,100000:0.1,300000:0' --lambda_xe_otfd 1 

## CPU on-the-fly generation
--otf_num_processes 8 --otf_sync_params_every 1000 

## optimization and training steps
--enc_optimizer adam,lr=0.00006 
--epoch_size 210000 
--batch_size 32 
--emb_dim 768 
--max_epoch 35 

## saving models and evaluation length (could set eval_length to -1 to eval every sentence, but took really long time)
--save_periodic True --eval_length 960 

## pad parmas
--pad_weight 0.1 
--do_bos True 

## rl parmas
--use_rl True 
--reward_gamma_ap 0.0 
--reward_gamma_ar 0.4 
--reward_type_ar punish 
--reward_thresh_ar 0.85 
--rl_start_epoch 0 

## With
MONO_DATASET='pm:./data/data_pad/jueju7_out.tr.pth,./data/data_pad/jueju7_out.vl.pth,,./data/data_pad/poem_jueju7_para.pm.pth;sw:./data/data_pad/sanwen.tr.pth,./data/data_pad/sanwen.vl.pth,./data/data_pad/sanwen.te.pth,./data/data_pad/poem_jueju7_para.sw.pth' 
PARA_DATASET='pm-sw:,,./data/data_pad/poem_jueju7_para.XX.pth'
PRETRAINED='./data/word_embeddings_weight.pt'

```

## Evaluation

If you would like to run evaluations only, append these three lines to training params above.

```
python main.py 

...(training params like shown above)...

--eval_only True 
--model_path ./dumped/test/4909294/periodic-20.pth  # path of model to load from
--dump_path ./dumped/test/eval_result               # result files to save to

```

## Citation

Please cite the following if you find this repo useful.

### Interpoetry: Generating Classical Chinese Poems from Vernacular Chinese

```
@inproceedings{yangcai2019interpoetry,
  title={Interpoetry: Generating Classical Chinese Poems from Vernacular Chinese},
  author={Yang, Zhichao and Cai, Pengshan and ... TODO},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2019}
}
```

## License

See the [LICENSE](LICENSE.md) file for more details.
