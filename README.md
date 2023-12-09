# Neural Ordinary Differential Scene Understanding (NODIS)
PyTorch implementation for the ECCV 2020 paper [NODIS: Neural Ordinary Differential Scene Understanding](https://arxiv.org/abs/2001.04735v3). Here's a preview of how the model works: ([Short Video](https://youtu.be/4VLnOpeIzjs), [Long Video](https://youtu.be/kgMRG8LxkH0)).

![Screenshot](/docs/teaser_eccv.png)

## Setup
1. Ensure python & pytorch is installed. Based on python 3.6 and pytorch 0.4.1.
2. Compile: Run `make` in the main directory
3. Download Neural ODE module [from here](https://github.com/rtqichen/torchdiffeq/tree/master/torchdiffeq)
4. For fair comparison, the pretrained object detector checkpoint is provided [here](https://drive.google.com/open?id=1xXIcROgv-u1Yq7ILIyWAndVBQxvP3jUD), and save it in *checkpoints/vgdet/*

## Training
Train the model using command `python train_rels.py …` More details are found in the README file within the repo.

## Evaluation
Evaluate the trained model by running command `python eval_rels.py …` NODIS pretrained models can be found [here](https://drive.google.com/open?id=…).

## Contribution
Fork the repository, make some changes, and open a pull request.If any doubt arises, feel free to reach out.