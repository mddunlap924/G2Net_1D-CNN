# G2Net Gravitational Wave Detection – 1D-CNN



This is code I developed for the [Kaggle G2Net Gravitational Wave Competition](https://www.kaggle.com/c/g2net-gravitational-wave-detection/overview). Thanks to teammates [Johnny Lee](https://www.kaggle.com/wuliaokaola), [Alex](https://www.kaggle.com/lihuajing), and [Coolz](https://www.kaggle.com/cooolz) we were able to finish in 10th position out of 1,219 teams (i.e., top 1%) through creative problem solving and advanced solutions.

The objective of this competition was to determine if a linear sine-sweep signal, also referred to as a chirp waveform, was either present or not in a data instance. The challenge is that the chirp waveform would have a very low signal-to-noise ratio (SNR) and frequency content could change between data instances. SNR is one, if not the most, informative measures of signal detectability. Fundamentally, this competition was a multi-channel signal detection challenge.

In this repository an one-dimensional convolutional neural network (1D-CNN) is used to assign a probability from 0-1 for the detection of the chirp waveform. We ensemble this 1D-CNN with other two-dimensional (2D) spectrogram image classification techniques to boost our score. I provide a 2D CQT transform approach [here](https://github.com/mddunlap924/G2Net_Spectrogram-Classification) which uses image classification network architecture(s).

# Requirements

The required packages are provided in the [requirements.txt](https://github.com/mddunlap924/G2Net_1D-CNN/blob/main/requirements.txt)

The data is ~77GB and can be found on [Kaggle](https://www.kaggle.com/c/g2net-gravitational-wave-detection/data). 

# 1D-CNN

In this section a description of the modeling approach and steps required to execute the code are provided. [PyTorch Lightning][1] was used for this project. 

1. The [execute_models_bash]() is used to execute a model and its configuration. Run commands in the terminal to execute. This file can be modified to execute over multiple configuration files. This approach is helpful for experimenting with various hyperparameters, training on multiple data folds,  and allowing you to work on other tasks. At the time of writing PyTorch Lightning had an issue with [releasing RAM][2] and this was a suitable workaround.

   ```
   sh execute_models_bash
   ```

2. [main_1D.py]() contains the high-level workflow and structure for calling commands such as data loading, k-fold splitting, data preprocessing, training, logging results, and inference. This file provides the overall workflow. 

3. [pl_model_1d.py]() contains methods and classes for tasks such as data normalization, waveform augmentations, data loaders, data modules, 1D-CNN model description, and checkpoint locations.

4. [helper_functions_1d.py]() contains methods and classes for tasks such as logging data with [Weights & Biases][3], signal processing and filtering techniques with [GWpy][4] such as data [spectral whitening][5], loading configuration parameters, and measuring descriptive statistics on the datasets.

The 1D-CNN architecture has six 1D CNN layers thats feed into three dense layers. Average pooling is used between 1D CNN layers, SiLU activation is used throughout, and dropout is used to help regularize in the dense layers.

Through multiple experiments is was found that polarity inversion was a beneficial augmentation technique. [Audiomentations][6] was used for testing various one-dimensional data augmentation techniques.

Users are encouraged to modify the files as they see fit to best work with their applications. 

[1]: https://www.pytorchlightning.ai/	"PyTorch Lightning"
[2]: https://github.com/PyTorchLightning/pytorch-lightning/issues/2010	"RAM not correctly released when training a pl module multiple times #2010"
[3]: http://www.xsgeo.com/course/spec.htm	"Weights & Biases"
[4]: https://gwpy.github.io/	"GWpy"
[5]: http://www.xsgeo.com/course/spec.htm	"Spectral Whitening in Practice"
[6]: https://github.com/iver56/audiomentations	"Audiomentations"

