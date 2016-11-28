--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--  Architecture borrowed from FlowNet:Simple
--
--  Fischer, Philipp, et al. "Flownet: Learning optical flow with convolutional networks."
--  arXiv preprint arXiv:1504.06852 (2015).
--

require 'nn'
require 'cunn'
require 'cudnn'
require 'math'
local nninit = require 'nninit'

local numResNetBlocks   = 15
local Conv              = cudnn.SpatialConvolution
local deConv            = cudnn.SpatialFullConvolution
local ReLU              = cudnn.ReLU
local BatchNorm         = cudnn.SpatialBatchNormalization

local function residual_block()
        local block         = nn.Sequential()
        local block_concat  = nn.ConcatTable()
        local block_seq     = nn.Sequential()

        block_seq:add(BatchNorm(64))
        block_seq:add(ReLU(true))
        block_seq:add(Conv(64, 64, 3, 3, 1, 1, 1, 1))
        block_seq:add(BatchNorm(64))
        block_seq:add(ReLU(true))
        block_seq:add(Conv(64, 64, 3, 3, 1, 1, 1, 1))

        block_concat:add(block_seq)
        block_concat:add(nn.Identity())

        block:add(block_concat)
        block:add(nn.CAddTable())
 
        return block
end

local function createModel(opt)
    
    local model         = nn.Sequential()

    model:add(Conv(3, 64, 9, 9, 1, 1, 4, 4))
    -- model:add(ReLU(true))

    for ii = 1, numResNetBlocks do
        model:add(residual_block())
    end

    model:add(deConv(64, 64, 4, 4, 2, 2, 1, 1, 0, 0))
    model:add(ReLU(true))
    model:add(deConv(64, 64, 4, 4, 2, 2, 1, 1, 0, 0))
    model:add(ReLU(true))
    model:add(Conv(64, 3, 9, 9, 1, 1, 4, 4))
    
    local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
    end

    local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
    end

    ConvInit('cudnn.SpatialConvolution')
    ConvInit('nn.SpatialConvolution')
    BNInit('cudnn.SpatialBatchNormalization')
    BNInit('nn.SpatialBatchNormalization')

    model:get(1).gradInput = nil

    return model
end

return createModel