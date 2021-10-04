# G2Net Gravitational Wave Detection – 1D-CNN



This is code I developed for the [Kaggle G2Net Gravitational Wave Competition](https://www.kaggle.com/c/g2net-gravitational-wave-detection/overview). Thanks to teammates [Johnny Lee](https://www.kaggle.com/wuliaokaola), [Alex](https://www.kaggle.com/lihuajin), and [Coolz](https://www.kaggle.com/cooolz) we were able to finish in 10th position out of 1,219 teams (i.e., top 1%) through creative problem solving and advanced solutions.

The objective of this competition was to determine if a linear sine-sweep signal, also referred to as a chirp waveform, was either present or not in a data instance. The challenge is that the chirp waveform would have a very low signal-to-noise ratio (SNR) and frequency content could change between data instances. SNR is a one, if not the most, informative measures of signal detectability. Fundamentally, this competition was a multi-channel signal detection challenge.

In this repository an one-dimensional convolutional neural network (1D-CNN) is used to assign a probability from 0-1 for the detection of the chirp waveform. We ensemble this 1D-CNN with other two-dimensional (2D) spectrogram image classification techniques to boost our score. I provide a 2D CQT transform approach at [here](https://github.com/mddunlap924/G2Net_Spectrogram-Classification) which use image classification network architectures.

# Requirements

The required packages are provided in the [requirements.txt](https://github.com/mddunlap924/G2Net_1D-CNN/blob/main/requirements.txt)
