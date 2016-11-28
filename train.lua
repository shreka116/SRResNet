--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'

local M = {}
local Trainer = torch.class('SRResNet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model           = model
   self.criterion       = criterion
   self.optimState      = optimState or {
      learningRate      = opt.learningRate,
      learningRateDecay = 0.0,
      beta1             = opt.beta_1,
      beta2             = opt.beta_2,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)
   local timer              = torch.Timer()
   local dataTimer          = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize  = dataloader:size()
   local lossSum    = 0.0
   local N          = 0
  
   print('=============================')
   print(self.optimState)
   print(self.model)
   print('=============================')
   
   print('=> Training epoch # ' .. epoch)
   self.model:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)
      
      -- begin train actual SRResNet
      local output              = self.model:forward(self.input):float()
      local batchSize           = output:size(1)
      
      local MSE_loss            = self.criterion:forward(self.model.output, self.gt_input)
      
      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.gt_input)
      self.model:backward(self.input , self.criterion.gradInput)

      optim.adam(feval, self.params, self.optimState)

      lossSum = lossSum + MSE_loss
	
      N = n

      if (n%100) == 0 then
          print(string.format('Gradient min: %1.4f \t max:  %1.4f \t norm: %1.4f', torch.min(self.gradParams:float()), torch.max(self.gradParams:float()), torch.norm(self.gradParams:float())))
        
          image.save('losses/input_img.png', self.input[{ {1},{},{},{} }]:reshape(3,self.input[{ {1},{},{},{} }]:size(3),self.input[{ {1},{},{},{} }]:size(4)))
          image.save('losses/estimated_img.png', self.model.output[{ {1},{},{},{} }]:reshape(3,self.model.output[{ {1},{},{},{} }]:size(3),self.model.output[{ {1},{},{},{} }]:size(4)))
          image.save('losses/gt_img.png', self.gt_input[{ {1},{},{},{} }]:reshape(3,self.gt_input[{ {1},{},{},{} }]:size(3),self.gt_input[{ {1},{},{},{} }]:size(4)))

      end

	if (n%10) == 0 then
        --   image.save('losses/input_img.png', self.input[{ {1},{},{},{} }]:reshape(3,self.input[{ {1},{},{},{} }]:size(3),self.input[{ {1},{},{},{} }]:size(4)))
        --   image.save('losses/estimated_img.png', self.model.output[{ {1},{},{},{} }]:reshape(3,self.model.output[{ {1},{},{},{} }]:size(3),self.model.output[{ {1},{},{},{} }]:size(4)))
        --   image.save('losses/gt_img.png', self.gt_input[{ {1},{},{},{} }]:reshape(3,self.gt_input[{ {1},{},{},{} }]:size(3),self.gt_input[{ {1},{},{},{} }]:size(4)))

   	     print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Loss %1.8f'):format(
             epoch, n, trainSize, timer:time().real, dataTime, MSE_loss))
   	end

   	   -- check that the storage didn't get changed due to an unfortunate getParameters call
 	    assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

    return lossSum / N
end

function Trainer:test(epoch, dataloader)

   local timer = torch.Timer()
   local size = dataloader:size()
   local avgPSNR    = 0.0
   local N          = 0
   local lossSum    = 0.0

   self.model:evaluate()
   for n, sample in dataloader:run() do
      -- Copy input and target to the GPU
      self:copyInputs(sample)

      -- begin train actual SRResNet
      local output              = self.model:forward(self.input):float()
      local batchSize           = output:size(1)
      
      local MSE_loss            = self.criterion:forward(self.gt_input, self.model.output)

      lossSum = lossSum + MSE_loss

      N = n

      local PSNR = evaluatePSNR(self.model.output, self.gt_input)
      avgPSNR = avgPSNR + PSNR

      for ii = 1, batchSize do
          image.save('losses/testing/estimated_img_' .. n  ..'.png', self.model.output[{ {ii},{},{},{} }]:reshape(3,self.model.output[{ {ii},{},{},{} }]:size(3),self.model.output[{ {ii},{},{},{} }]:size(4)))
          image.save('losses/testing/gt_img' .. n .. '.png', self.gt_input[{ {ii},{},{},{} }]:reshape(3,self.gt_input[{ {ii},{},{},{} }]:size(3),self.gt_input[{ {ii},{},{},{} }]:size(4)))
      end
      collectgarbage()
      print((' | Test: [%d][%d/%d]    Time %.3f  loss %1.4f PSNR %1.3f'):format( epoch, n, size, timer:time().real, MSE_loss, PSNR))



      timer:reset()
   end
   self.model:training()

   return lossSum/N, avgPSNR/N
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.gt_input = self.gt_input or torch.CudaTensor()

   self.input:resize(sample.input:size()):copy(sample.input)
   self.gt_input:resize(sample.gt_input:size()):copy(sample.gt_input)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   if (self.opt.dataset == 'FlyingChairs') and (epoch >= 60) then
      if (epoch%30 == 0) then
      	return self.optimState.learningRate/2
      else
        return self.optimState.learningRate
      end
   else
	    return self.optimState.learningRate
   end 
end

return M.Trainer
