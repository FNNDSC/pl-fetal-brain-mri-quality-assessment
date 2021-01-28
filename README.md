# Automatic Fetal Brain Quality Assessment

Iván Legorreta <ilegorreta@outlook.com>

## Abstract

The aim of this project was to develop a Quality Assessment tool for fetal brain MRIs,
which is able to score each volume through a deep learning regression model.
Developed using Python3 and Keras/Tensorflow framework.

Our network architecture consists of a non-linear configuration, known as Residual Network (ResNet) architecture: 
![Resnet Architecture Diagram](https://github.com/ilegorreta/Automatic-Fetal-Brain-Quality-Assessment-Tool/blob/main/resnet_architecture_diagram.png)

Given that we are dealing with an unbalanced distribution regarding input dataset,
we applied different weights to each input class to compensate for the imbalance in the training sample.

This repository contains the tool to be used for predications and downstream research.
For model training and validation, see
https://github.com/ilegorreta/Automatic-Fetal-Brain-Quality-Assessment-Tool

## Development

```bash
DOCKER_BUILDKIT=1 docker build -t fnndsc/pl-fetal-quality-assessment .
```

<details>
<summary>What's BuildKit?</summary>
Our <code>Dockerfile</code> leverages advanced features of Docker.

<ul>
<li>https://github.com/moby/moby/issues/15717#issuecomment-493854811</li>
<li>https://docs.docker.com/engine/reference/builder/#buildkit</li>
</ul>
</details>
