--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--
require 'gnuplot'
require 'torch'
require 'paths'
require 'optim'
require 'nn'

local models        = require 'models/init'
local DataLoader    = require 'dataloader'
local opts          = require 'opts'
local Trainer       = require 'train'
local checkpoints   = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
cutorch.setDevice(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

--------TO DO -----------TO DO----------------
-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)


-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)


--------TO DO -----------TO DO----------------
if opt.testOnly then
   local loss, PSNR = trainer:test(0, valLoader)
   print(string.format(' * Results loss: %1.4f  PSNR: %1.3f', loss, PSNR))
   return
end
---------------------------------------------

--------TO DO -----------TO DO----------------
local PSNR, SSIM = 0.0,0.0
local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local Losses     = checkpoint and torch.load('checkpoints/Losses_' .. startEpoch-1 .. '.t7') or {trainLosses = {}, testLosses = {}}

for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainLoss  = trainer:train(epoch, trainLoader)
   Losses.trainLosses[#Losses.trainLosses + 1] = trainLoss
   gnuplot.pngfigure('losses/trainLoss.png')
   gnuplot.plot({torch.range(1, #Losses.trainLosses), torch.Tensor(Losses.trainLosses), '-'})
   gnuplot.plotflush()

--    -- Run model on validation set
--    local testLoss, PSNR = trainer:test(epoch, valLoader)
--    Losses.testLosses[#Losses.testLosses + 1] = testLoss
--    gnuplot.pngfigure('losses/testLoss.png')
--    gnuplot.plot({torch.range(1, #Losses.testLosses), torch.Tensor(Losses.testLosses), '-'})
--    gnuplot.plotflush()

   if (epoch%10 == 0) then
        checkpoints.save(epoch, model, trainer.optimState, opt)
        torch.save('checkpoints/Losses_' .. epoch .. '.t7', Losses)

   end

    -- print(string.format(' * Finished Epoch [%d/%d]  PSNR: %1.3f',epoch, opt.nEpochs, PSNR))

end

---------------------------------------------
