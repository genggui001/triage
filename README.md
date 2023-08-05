# Triage
The source code of Triage.

## Requirements
### GPU Requirements
* We have successfully run this code on 3090, V100 and A100 respectively.

### Python Requirements
```
pip install nvidia-pyindex==1.0.9
conda env create -f tf1.15.yml
```

## Pretrained Language Model
We conducted experiments on three language models RoBERTa, NEZHA, and RoFormer respectively. You can download the pretrained language models from the links below.
* [chinese_roformer-char_L-12_H-768_A-12](https://github.com/ZhuiyiTechnology/roformer)

The directory of the model in **triage.py** should be changed, for example:
```Python
config_path = './pretrain_weights/chinese_roformer-char_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './pretrain_weights/chinese_roformer-char_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './pretrain_weights/chinese_roformer-char_L-12_H-768_A-12/vocab.txt'
```
