# ETNet: Error Transition Network for Arbitrary Style Transfer

This repository contains the code (in [TensorFlow](https://www.tensorflow.org/)) for the paper:

[__ETNet: Error Transition Network for Arbitrary Style Transfer__](https://arxiv.org/pdf/1910.12056.pdf)
<br>
Chunjin song*, [Zhijie Wu*](https://zhijiew94.github.io/), [Yang Zhou](https://zhouyangvcc.github.io/), [Minglun Gong](http://www.cs.mun.ca/~gong/), [Hui Huang](https://vcc.tech/~huihuang) (* equal contribution, in alphabetic order)
<br>
NeurIPS 2019

## Introduction

This repository contains an official implementation for [ETNet: Error Transition Network for Arbitrary Style Transfer](https://arxiv.org/pdf/1910.12056.pdf). To improve the stylization results, we introduce an iterative error-correction mechanism to break the stylization process into multiple refinements with the Laplacian pyramid strategy. Given an insufficiently stylized image in a refinement, we compute what is wrong with the current estimate and then transit the error information to the whole image. The simplicity and motivation lie in the following aspect: the detected errors evaluate the residuals to the ground truth, which thus can guide the refinement effectively. Based on this motivation, ETNet can achieve stylization by presenting more adaptive style patterns and preserving high-level content structure better.

If you have any questions, please feel free to contact Zhijie Wu (wzj.micker@gmail.com).


[teaser]: ./docs/figures/teaser.jpg
![teaser]

## Stylization Matrix
[image_results]: ./docs/figures/matrices.jpg
![image_results]

## Refinements for existing methods
[refined_results]: ./docs/figures/refined_detail.jpg
![refined_results]

## Comparison with existing methods

[comparison_results]: ./docs/figures/comparison.jpg
![comparison_results]

- ETNet synthesizes visually pleasing results with concrete multi-scale style patterns (e.g. color distribution, brush strokes and some other salient patterns in style images).
- Both [AvatarNet](https://arxiv.org/abs/1805.03857) and [AAMS](https://arxiv.org/abs/1901.05127) often blur content structures. [WCT](https://arxiv.org/abs/1705.08086) distorts the brush strokes and some other salient style patterns, while [AdaIN](https://arxiv.org/abs/1703.06868) cannot even keep the color distribution.


## Dependencies
- Python 3.6
- CUDA 9.0
- Cudnn 7.1.2
- [TensorFlow](https://www.tensorflow.org/) (version >= 1.12, but just tested on TensorFlow 1.12).
- Keras 2.0.0
- Numpy
- scipy
- pyyaml

## Download
- The training of our style transfer network requires pretrained [VGG](https://arxiv.org/abs/1409.1556) networks. Both the trained model of ETNet and pre-trained VGG model can be downloaded through the [Google Drive](https://drive.google.com/file/d/1vqxi63GHIA_eLmXpRo02fknROIrJ4yF4/view?usp=sharing). Note that the pre-trained VGG model is used as the encoding layers of our model. Please place all the downloaded models to appropriate directories as shown below.
- [Places365](http://places2.csail.mit.edu/download.html) dataset is applied for the content image dataset.
- [WikiArt](https://www.kaggle.com/c/painter-by-numbers/data) dataset is used as the style image dataset.

## Usage

### Train Model

In order to train a model to transfer styles, we should set some essential information in the `train.yaml`.
```
GPU_ID: [0,1,2,3]  # GPU ids used for training
...
Incremental: 2  # Count of layers in current network
layers_num: 3
...
dir_checkpoint: [[0, models/layer1/checkpoints],[1, models/layer2/checkpoints]] # Directory of pretrained models: [[layer_0, model_0],[layer_1, model_1],...]
vgg_weights: models/vgg_normalised.t7   

data: # Directory for input training images
  dir_content: image/input/content
  dir_style: image/input/style

dir_out: image/output/train # Root directory for results
output: # Subdirectories for results
  dir_log: logs
  dir_config: configs
  dir_sample: samples
  dir_checkpoint: checkpoints
```
Generally, the `train.yaml` is written in `json` format and contain the following options.
- `GPU_ID`: a list for GPU ID. We use four GPUs (e.g. [0, 1, 2, 3]) for training the network at third level by default.
- `Incremental`: the layer index for training, starting from 0.
- `layers_num`: the network count in current model.
- `weight`: the weight to balance the effect of `content` and `styles` of perceptual loss. When we train the networks at different levels, we assign different values to `style`. For a model with three different levels, we set `styles` as `2`, `5`, `8`  at the training stages for the networks at first, second and third level respectively.
- `dir_checkpoint`: a path list for pretrained networks. Each element in the list is also another list, which aims to indicate the layer index and corresponding model path.
- `vgg_weights`: the directory of pretrained VGG models.
- `data`: the directory to place all the training data. Specifically, `dir_content` and `dir_style` are the two directories to store content and style images correspondingly. They can be `image/input/content` for multiple content images and `image/input/style` for style images.
- `dir_out`: the root directory for output results. During training, the model would output the initial configuration information and other intermediate information about logs, stylized samples, checkpoints into `dir_config`, `dir_log`, `dir_sample` and `dir_checkpoint`.


Please note that, when training networks at different levels, we should update `Incremental`, `layers_num`, `weight` and `dir_checkpoint` correspondingly to adjust the balance between content and style, preserve the same training batch size and indicate which network to be trained. For the network at first level, these keys should be set up as:
```
GPU_ID: [0]

Incremental: 0  
layers_num: 1  

weight:  
  content: 1.0
  style:  2.0

dir_checkpoint: None 
```
But for finetuning this network, the `dir_checkpoint` should be set as:
```
dir_checkpoint: [[0, models/layer1/checkpoints]] # models/layer1/checkpoints is path of a pretrained model
```
For the network at second level, we update the configuration as:
```
GPU_ID: [0,1]

Incremental: 1  
layers_num: 2  

weight:  
  content: 1.0
  style:  5.0

dir_checkpoint: [[0, models/layer1/checkpoints]]
```
Similarly, we can add another list to the `dir_checkpoint` to include an extra pretrained model for further finetuning.

As for the network at third level, the file can be configured as:
```
GPU_ID: [0,1,2,3]
Incremental: 2 
layers_num: 3 

weight:
  content: 1.0
  style:  8.0

dir_checkpoint: [[0, models/layer1/checkpoints],[1, models/layer2/checkpoints]]
```

After the `train.yaml` has been set well, then we directly run `train.py` to start model training as:
```
python train.py
```


### Test Model

Before transfering styles into content structures with the trained model, we should configurate `test.yaml` as:
```
GPU_ID: 0 # the gpu for testing

layers_num: 3 

checkpoint_dir: [[0, models/layer1/checkpoints],[1, models/layer2/checkpoints],[2, models/layer3/checkpoints]]
vgg_weights: models/vgg_normalised.t7


data: # Directory for testing images
  dir_content: image/input/content
  dir_style: image/input/style

dir_out: image/output/test # Root directory for output results
output: # Subdirectories for outputs
  dir_result: results
```
This file (`test.yaml`) has the following options:
- `GPU_ID`: indicate which gpu should be used for testing.
- `layers_num`: the level number within a model.
- `checkpoint_dir`: the path list to place the pretrained model, which will be used for generation. More information can be refered in the `training` part.
- `vgg_weights`: the path for pretrained VGG models.
- `data`: the directory of the input images. Specifically, `dir_content` and `dir_style` are used to indicate the pathes of content and style images respectively.
- `dir_out`: the output directory for evaluation results. It can be `image/output/test` for multiple synthesis images.

For the model with three different levels, after setting `test.yaml` as shown above, we can start the testing by running `stylize.py`, such as
```
python stylize.py
```

## Citation

If you use our code/model/data, please cite our paper:

```
@misc{song2019etnet,
    title={ETNet: Error Transition Network for Arbitrary Style Transfer},
    author={Chunjin Song and Zhijie Wu and Yang Zhou and Minglun Gong and Hui Huang},
    year={2019},
    eprint={1910.12056},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## License
MIT License
