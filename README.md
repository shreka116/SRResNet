# SRResNet

This is the impelmentation of SRResNet, a part of SRGAN "Ledig, Christian, et al. "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network." arXiv preprint arXiv:1609.04802 (2016).".


Modification made from the paper:

1) residual blocks 	      -->  preactivation residual blocks

2) first and last conv. layer -->  9x9 conv. layer


Implementation was done in Torch7 supporting cuda/cudnn backend. (PASCAL Titan X)


###To Train Model
'th main.lua -nEpochs 350 -batchSize 16 -learningRate 1e-4'
options in training the model can be changed by user's preference and can be found in 'opts.lua'.


###To Test Model
'th main.lua -testOnly true -resume checkpoints -testData Set5'
options in testing the model can be found in 'opts.lua'.


