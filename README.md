# CNNs for Text Classification in PyTorch

A minimal PyTorch implementation of Convolutional Neural Networks (CNNs) for text classification.

Supported features:
- Lookup, character embedding and/or self-attentive encoding in the input layer
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
python3 prepare.py training_data
```

To train:
```
python3 train.py model char_to_idx word_to_idx tag_to_idx training_data.csv (validation_data) num_epoch
```

To predict:
```
python3 predict.py model.epochN char_to_idx word_to_idx tag_to_idx test_data
```

To evaluate:
```
python3 evaluate.py model.epochN char_to_idx word_to_idx tag_to_idx test_data
```

## References

Yoon Kim. 2014. [Convolutional Neural Networks for Sentence Classification.](https://arxiv.org/abs/1408.5882) arXiv:1408.5882.

Yunlun Yang, Yunhai Tong, Shulei Ma, Zhi-Hong Deng. 2016. [A Position Encoding Convolutional Neural Network Based on Dependency Tree for Relation Classification.](https://aclweb.org/anthology/D16-1007) In EMNLP.

Xiang Zhang, Junbo Zhao, Yann LeCun. 2015. [Character-level Convolutional Networks for Text Classification.](https://arxiv.org/abs/1509.01626) arXiv:1509.01626.
