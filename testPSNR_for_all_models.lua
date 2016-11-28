require 'gnuplot'
require 'sys'
require 'cutorch'

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
--------TO DO -----------TO DO----------------
-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

local PSNRs = {}
for i=1,250/10 do
   local recents     = torch.load('checkpoints/latest.t7')
   recents.epoch     = i*10
   recents.optimFile = "optimState_" .. i*10 .. '.t7'
   recents.modelFile = "model_" .. i*10 .. ".t7"
   torch.save('checkpoints/latest.t7', recents)
   -- Load previous checkpoint, if it exists
   local checkpoint, optimState = checkpoints.latest(opt)

   -- Create model
   local model, criterion = models.setup(opt, checkpoint)

   -- The trainer handles the training loop and evaluation on validation set
   local trainer = Trainer(model, criterion, opt, optimState)


   if opt.testOnly then

      local loss, PSNR = trainer:test(0, valLoader)

      print(string.format(' * Results loss: %1.4f  PSNR: %1.3f', loss, PSNR))
      PSNRs[#PSNRs + 1]= PSNR

   end

--   sys.execute('th main.lua -nGPU 1 -batchSize 1 -testOnly true')
   gnuplot.pngfigure('losses/testing/testPSNR.png')
   gnuplot.plot({ torch.range(1, #PSNRs), torch.Tensor(PSNRs), '-' })
   gnuplot.plotflush()
   
end

