# CNNs for Text Classification in PyTorch

A minimal PyTorch implementation of Convolutional Neural Networks (CNNs) for text classification.

Supported features:
- Character and/or word embeddings in the input layer
- Mini-batch training with CUDA

## Usage

Training data should be formatted as below:
```
sentence \t label
sentence \t label
...
```

To prepare data:
```
python prepare.py training_data
```

To train:
```
python train.py model char_to_idx word_to_idx tag_to_idx training_data.csv (validation_data) num_epoch
```

To predict:
```
python predict.py model.epochN char_to_idx word_to_idx tag_to_idx test_data
```

To evaluate:
```
python evaluate.py model.epochN char_to_idx word_to_idx tag_to_idx test_data
```

## References

Yoon Kim. 2014. [Convolutional Neural Networks for Sentence Classification.](https://arxiv.org/abs/1408.5882) arXiv:1408.5882.

Yunlun Yang, Yunhai Tong, Shulei Ma, Zhi-Hong Deng. 2016. [A Position Encoding Convolutional Neural Network Based on Dependency Tree for Relation Classification.](https://aclweb.org/anthology/D16-1007) In EMNLP.
