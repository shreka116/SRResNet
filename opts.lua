--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--


local M = {}

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 Unsupervised Optical Flow via CNN Training script')
--   cmd:text('See https://github.com/facebook/fb.resnet.torch/blob/master/TRAINING.md for examples')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
--   cmd:option('-data',       '',         'Path to dataset')
   cmd:option('-dataset',    'ILSVRC2015_5000', 'Options: ILSVRC2015_5000')
   cmd:option('-testData',         'Set14'    , 'Options: Set5 | Set14 | BSD100')
   cmd:option('-manualSeed',         0,         'Manually set RNG seed')
   cmd:option('-nGPU',               1,         'Number of GPUs to use by default')
   cmd:option('-backend',         'cudnn',      'Options: cudnn | cunn')
   cmd:option('-cudnn',          'fastest',     'Options: fastest | default | deterministic')
   cmd:option('-genData',        '../dataset',  'Path to dataset')
   ------------- Data options ------------------------
   cmd:option('-nThreads',           2,         'number of data loading threads')
   ------------- Training options --------------------
   cmd:option('-nEpochs',            0,         'Number of total epochs to run')
   cmd:option('-epochNumber',        1,         'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',         16,         'mini-batch size (1 = pure stochastic)')
   cmd:option('-testOnly',         'false',     'Run on validation set only')
--    cmd:option('-tenCrop',         'false', 'Ten-crop testing')
   ------------- Checkpointing options ---------------
   cmd:option('-save',          'checkpoints',  'Directory in which to save checkpoints')
   cmd:option('-resume',           'none',      'Resume from the latest checkpoint in this directory')
   cmd:option('-retrain',          'none',      'Resume from the latest checkpoint in this directory')

   ---------- Optimization options ----------------------
   cmd:option('-learningRate',      1e-4,       'initial learning rate')
   cmd:option('-beta_1',            0.9,        'first parameter of Adam optimizer')
   cmd:option('-beta_2',           0.999,       'second parameter of Adam optimizer')
   ---------- Model options ----------------------------------
   cmd:option('-networkType',     'SRResNet',   'Options: SRResNet')
   cmd:option('-retrain',         'none',       'Path to model to retrain with')
   cmd:option('-optimState',      'none',       'Path to an optimState to reload from')
   ---------- Hyper parameters  ------------------------------
   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.testOnly = opt.testOnly ~= 'false'
--    opt.tenCrop = opt.tenCrop ~= 'false'
--    opt.shareGradInput = opt.shareGradInput ~= 'false'
--    opt.optnet = opt.optnet ~= 'false'
--    opt.resetClassifier = opt.resetClassifier ~= 'false'

   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
   end

   if opt.dataset == 'ILSVRC2015_5000' then
      opt.nEpochs = opt.nEpochs == 0 and 200 or opt.nEpochs
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   return opt
end

return M
