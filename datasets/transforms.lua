--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Image transforms for data augmentation and input normalization
--

require 'image'
require 'nn'

local M = {}

function M.SelectTransform(transforms)
   return function(tr, gt)
      if #transforms == 1 then
        tr, gt  = transforms[1](tr, gt)
      end
      return tr, gt
   end
end

function M.Compose(transforms)
   return function(input)
      for idx, transform in ipairs(transforms) do
        --  print(tostring(idx) .. '-->' .. tostring(transform))
         input = transform(input)
        --  print(tostring(idx) .. '-input -->' .. tostring(input:size()))
      end
    --   print('outa for loop-input -->' .. tostring(input:size()))
      return input
   end
end

function M.ColorNormalize(meanstd)
   return function(img)
      img = img:clone()
      for i=1,3 do
         img[i]:add(-meanstd.mean[i])
         img[i]:div(meanstd.std[i])
      end
      return img
   end
end


-- Lighting noise (AlexNet-style PCA-based noise)
function M.Lighting(alphastd, eigval, eigvec)
   return function(input)
      if alphastd == 0 then
         return input
      end

      local alpha = torch.Tensor(3):normal(0, alphastd)
      local rgb = eigvec:clone()
         :cmul(alpha:view(1, 3):expand(3, 3))
         :cmul(eigval:view(1, 3):expand(3, 3))
         :sum(2)
         :squeeze()

      input = input:clone()
      for i=1,3 do
         input[i]:add(rgb[i])
      end
      return input
   end
end

local function blend(img1, img2, alpha)
   return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
   dst:resizeAs(img)
   dst[1]:zero()
   dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
   dst[2]:copy(dst[1])
   dst[3]:copy(dst[1])
   return dst
end

function M.Saturation(var)
   local gs

   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function M.Brightness(var)
   local gs

   return function(input)
      gs = gs or input.new()
      gs:resizeAs(input):zero()

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end


function M.AdditiveGausNoise(var_1, var_2)

    return function(input)
        local gs = input.new()
        gs:resizeAs(input):zero()
        -- print(gs:size())

        local sigma = torch.uniform(var_1, var_2)
        torch.normal(gs:select(1,1), 0, sigma)
            gs:select(1,2):copy(gs:select(1,1))
            gs:select(1,3):copy(gs:select(1,1))
            gs:select(1,4):copy(gs:select(1,1))
            gs:select(1,5):copy(gs:select(1,1))
            gs:select(1,6):copy(gs:select(1,1))
       
        return input:add(gs)
    end
end

function M.Contrast(var_1, var_2)

   return function(input)
      local gs = input.new()
      gs:resizeAs(input):zero()
    --   local ref_gray = rgb2gray(input[{ {1,3},{},{} }])
    --   local tar_gray = rgb2gray(input[{ {4,6},{},{} }])
      grayscale(gs[{ {1,3},{},{} }], input[{ {1,3},{},{} }])
      grayscale(gs[{ {4,6},{},{} }], input[{ {4,6},{},{} }])
      gs[{ {1,3},{},{} }]:fill(gs[{ {1,3},{},{} }][1]:mean())
      gs[{ {4,6},{},{} }]:fill(gs[{ {4,6},{},{} }][1]:mean())

      local alpha = 1.0 + torch.uniform(var_1, var_2)
      blend(input, gs, alpha)
      return input
   end
end

function M.MultiplicativeColorChange(var_1, var_2)

    return function(input)

      local mult_R = torch.uniform(var_1, var_2)
      local mult_G = torch.uniform(var_1, var_2)
      local mult_B = torch.uniform(var_1, var_2)

      input:select(1,1):mul(mult_R)
      input:select(1,2):mul(mult_G)
      input:select(1,3):mul(mult_B)
      input:select(1,4):mul(mult_R)
      input:select(1,5):mul(mult_G)
      input:select(1,6):mul(mult_B)


      return input
    end
end

function M.AdditiveBrightness(var)

    return function(input) 

      local ref_hsl = image.rgb2hsl(input[{ {1,3},{},{} }])
      local tar_hsl = image.rgb2hsl(input[{ {4,6},{},{} }])
      local changes = torch.normal(0, 0.2)
      ref_hsl:select(1,3):add(changes)
      tar_hsl:select(1,3):add(changes)
      input[{ {1,3},{},{} }]:copy(image.hsl2rgb(ref_hsl))
      input[{ {4,6},{},{} }]:copy(image.hsl2rgb(tar_hsl))

      return input
    end
end

function M.GammaChanges(var_1, var_2)

    return function(input) 
      local ref_hsl = image.rgb2hsl(input[{ {1,3},{},{} }])
      local tar_hsl = image.rgb2hsl(input[{ {4,6},{},{} }])
      local gamma   = torch.uniform(var_1, var_2)
      ref_hsl:select(1,3):pow(gamma)
      tar_hsl:select(1,3):pow(gamma)
      input[{ {1,3},{},{} }]:copy(image.hsl2rgb(ref_hsl))
      input[{ {4,6},{},{} }]:copy(image.hsl2rgb(tar_hsl))

      return input
    end
end

function M.HorizontalFlip(prob)
   return function(input)
      if torch.uniform() < prob then
       input = image.hflip(input)
      end
      return input
   end
end

function M.randomCrop(tr, gt)
    return function(tr, gt)
       
       local inputSz     = tr:size()
     	 local x_from      = torch.random(1,inputSz[3]-24)
       local y_from      = torch.random(1,inputSz[2]-24)
      --  print('tr' .. tostring(tr:size()))
      --  print(x_from,y_from,x_from + 24, y_from + 24)
      --  print('gt' .. tostring(gt:size()))
      --  print(x_from*4,y_from*4,x_from*4 + 96, y_from*4 + 96)
       

       return image.crop(tr, x_from, y_from, x_from + 24, y_from + 24), image.crop(gt, x_from*4, y_from*4, x_from*4 + 96, y_from*4 + 96)
    end
end

function M.Identity()
   return function(input)
      return input
   end
end

return M
