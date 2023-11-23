# SpeechTripleNet
The implementation of paper "SpeechTripleNet: End-to-End Disentangled Speech Representation Learning for Content, Timbre and Prosody"

### [Demo](https://speechtriplenet.github.io/) | [Paper](https://www1.se.cuhk.edu.hk/~hccl/publications/pub/mmfp3442-lu-CC-BY.pdf)

### Environment setup
```bash
conda env create -f environment.yml
conda activate speechtriplenet-env
```

### Quick try of speech editing with pretrained model
```bash
jupyter notebook speech_editing.ipynb
```

### Data feature extraction for training
```bash
python preprocess.py --config ./configs/VCTK/preprocess.yaml
```

### Training
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --mdl CCDPJ -p ./configs/VCTK/preprocess.yaml -t ./configs/VCTK/train.yaml -m ./configs/VCTK/model.yaml
```

### Inference
See ```speech_editing.ipynb```.