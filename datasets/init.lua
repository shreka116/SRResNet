--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--

local M = {}

local function isvalid(opt, cachePath)
   local imageInfo = torch.load(cachePath)
   if imageInfo.basedir and imageInfo.basedir ~= opt.data then
      return false
   end
   return true
end

function M.create(opt, split)
   local cachePath
   if opt.dataset == 'ILSVRC2015_5000' then
        cachePath = (opt.genData .. '/SR_ILSVRC2015_val_4_rgb/' .. opt.dataset ..'_' .. opt.testData .. '.t7')
   else
        cachePath = (opt.genData .. '/' .. opt.dataset .. '/' .. opt.dataset .. '.t7')
   end
   
   if not paths.filep(cachePath) or not isvalid(opt, cachePath) then
      paths.mkdir('../dataset')

      local script = paths.dofile(opt.dataset .. '-genData.lua')
      script.exec(opt, cachePath)
   end
   print(cachePath)
   local imageInfo = torch.load(cachePath)
   
   local Dataset = require('datasets/' .. opt.dataset)
   
   return Dataset(imageInfo, opt, split)
end

return M
