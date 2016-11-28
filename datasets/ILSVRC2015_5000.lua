--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--  Downloads data from web and save them as .t7 extensions
--
require 'utils'

local image = require 'image'
local paths = require 'paths'
local ffi = require 'ffi'
local t = require 'datasets/transforms'

local M = {}
local ILSVRC = torch.class('SRResNet.ILSVRC', M)

function ILSVRC:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
--    self.dir = paths.concat(opt.data, split)
--    assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function ILSVRC:get(i)
   local path_img = ffi.string(self.imageInfo.imagePath[i]:data())
   local path_gt  = ffi.string(self.imageInfo.gtPath[i]:data())

   local image_img = self:_loadImage(path_img)
   local image_gt = self:_loadImage(path_gt)

   return {
    input     = image_img,
    gt_input  = image_gt,
   }
end

function ILSVRC:_loadImage(path)
   local ok, input = pcall(function()
      return image.load(path, 3, 'float')
   end)

   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()

      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))

      input = image.decompress(b, 3, 'float')
   end

   return input
end

function ILSVRC:size()
   return self.imageInfo.imagePath:size(1)
end

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

function ILSVRC:preprocess(tr, gt)
   if self.split == 'train' then
         return t.SelectTransform{
			t.randomCrop(tr, gt),
        }

   elseif self.split == 'val' then
          return t.SelectTransform{
            --   t.Identity(),
        }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.ILSVRC
