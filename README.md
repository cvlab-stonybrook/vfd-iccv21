# Variational_Transfer_Fewshot

Implementation for Variational Transfer Learning for Fine-grained Few-shot Visual Recognition

## Environment
- Python3
- Pytorch 1.4.0

## Datasets

### CUB
- Change directoy to filelists/CUB
- Download the dataset from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
- Put it as `./CUB_200_2011`
- Run `python ./write_CUB_filelist.py`  

### NAB
- Change directoy to filelists/NAB
- Download the dataset from http://dl.allaboutbirds.org/nabirds
- Put it as `./nabirds`
- Run `python ./write_NAB_filelist.py`  

### Stanford Dogs
- Change directoy to filelists/DOG
- Download the dataset from http://vision.stanford.edu/aditya86/ImageNetDogs/
- Put it as `./stanforddogs`
- Run `python ./write_DOG_filelist.py`  


## Running Experiments

### Run training phase:
```bash
python train_vae.py
```

### Run testing phase:
```bash
python finetune_sample.py
```




