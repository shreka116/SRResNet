require 'torch'
require 'math'
require 'paths'

TAG_FLOAT = 202021.25


local function byte2float(src)
   local conversion = false
   local dest = src
   if src:type() == "torch.ByteTensor" then
      conversion = true
      dest = src:float():div(255.0)
   end
   return dest, conversion
end

local function float2byte(src)
   local conversion = false
   local dest = src
   if src:type() == "torch.FloatTensor" then
      conversion = true
      dest = (src + clip_eps8):mul(255.0)
      dest[torch.lt(dest, 0.0)] = 0
      dest[torch.gt(dest, 255.0)] = 255.0
      dest = dest:byte()
   end
   return dest, conversion
end

local function rgb2y_matlab(x)
   local y = torch.Tensor(1, x:size(2), x:size(3)):zero():cuda()
   x = byte2float(x)
--    print(torch.max(x))
   y:add(x[1] * 65.481)
   y:add(x[2] * 128.553)
   y:add(x[3] * 24.966)
   y:add(16.0)
   
   -- 235
   return y:byte():float()
end

local function YMSE(x1, x2)
      local x1_2 = rgb2y_matlab(x1)
      local x2_2 = rgb2y_matlab(x2)
      return (x1_2 - x2_2):pow(2):mean()
end


local function MSE(x1, x2)
      return YMSE(x1, x2)
end

local function PSNR(x1, x2)
   local mse = math.max(MSE(x1, x2), 1)
   return 10 * math.log10((255.0 * 255.0) / mse)
end

function evaluatePSNR(est_img, gt_img)
    -- how many boundary pixels to ignore
    local bndPixels = 4

    local inputSize = est_img:size()
    local avgPSNR = 0.0
    local inputSize = est_img:size()
    for i = 1, inputSize[1] do 
        avgPSNR = avgPSNR + PSNR(est_img[{ {i},{},{1+bndPixels,inputSize[3]-bndPixels},{1+bndPixels,inputSize[4]-bndPixels} }]:reshape(inputSize[2],inputSize[3]-2*bndPixels,inputSize[4]-2*bndPixels),gt_img[{ {i},{},{1+bndPixels, inputSize[3]-bndPixels},{1+bndPixels,inputSize[4]-bndPixels} }]:reshape(inputSize[2],inputSize[3]-2*bndPixels,inputSize[4]-2*bndPixels))
    end

    return avgPSNR/inputSize[1]
end