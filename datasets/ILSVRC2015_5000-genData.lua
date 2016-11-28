--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--  Downloads data from web and save them as .t7 extensions
--

local image = require 'image'
local paths = require 'paths'
local ffi = require 'ffi'

local M = {}

local data_URL  = ' https://www.dropbox.com/s/3li9izne2t5kn0r/SR_ILSVRC2015_val_4_rgb.zip?dl=0'

local function findImages(dir)
        local imagePath = torch.CharTensor()
        ----------------------------------------------------------------------
        -- Options for the GNU and BSD find command
        local extensionList = {'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
        local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
        for i=2,#extensionList do
            findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
        end

        -- Find all the images using the find command
        local f = io.popen('find -L ' .. dir .. findOptions)

        local maxLength = -1
        local imagePaths = {}

        -- Generate a list of all the images and their class
        while true do
            local line = f:read('*line')
            if not line then break end

            local className = paths.basename(paths.dirname(line))
            local filename = paths.basename(line)
            local path = paths.concat(dir, filename)

            table.insert(imagePaths, path)

            maxLength = math.max(maxLength, #path + 1)
        end

        f:close()

        -- Convert the generated list to a tensor for faster loading
        table.sort(imagePaths)
        local nImages = #imagePaths
        local imagePath = torch.CharTensor(nImages, maxLength):zero()
        for i, path in ipairs(imagePaths) do
            ffi.copy(imagePath[i]:data(), path)
        end

        return imagePath
end

function M.exec(opt, cacheFile)
    if not paths.dirp(opt.genData .. '/SR_ILSVRC2015_val_4_rgb') then
        print("=> Downloading SR_ILSVRC2015_val_4_rgb dataset from " .. data_URL)
        local down_ok   = os.execute('wget -P ' .. opt.genData .. '/ ' .. data_URL)
        assert(down_ok == true or down_ok == 0, 'error downloading SR_ILSVRC2015_val_4_rgb')
        local unzip_ok  = os.execute('unzip ' .. opt,genData .. '/SR_ILSVRC2015_val_4_rgb.zip')
        assert(unzip_ok == true or unzip_ok == 0, 'error extracting SR_ILSVRC2015_val_4_rgb.zip')
    end

    local imagePath = torch.CharTensor()  -- path to each image in dataset
    local gtPath    = torch.CharTensor()  -- path to each image in dataset

    local trainDir  = paths.concat(opt.genData, 'SR_ILSVRC2015_val_4_rgb/SR_ILSVRC2015_val_4_LR')
    local gtDir  = paths.concat(opt.genData, 'SR_ILSVRC2015_val_4_rgb/SR_ILSVRC2015_val_4_GT')
    assert(paths.dirp(trainDir), 'train directory not found: ' .. trainDir)
    assert(paths.dirp(gtDir), 'gt directory not found: ' .. gtDir)


    print(" | finding all SR_ILSVRC2015 training images")
    local trainImagePath    = findImages(trainDir, opt)
    local trainGtPath       = findImages(gtDir, opt)

    local valDir
    local valGtDir
    if opt.testData == "Set5" then
        -- Set5 dataset
        valDir  = paths.concat(opt.genData, 'Set5/train')
        valGtDir  = paths.concat(opt.genData, 'Set5/gt')
        assert(paths.dirp(valDir), 'train directory not found: ' .. valDir)
        assert(paths.dirp(valGtDir), 'gt directory not found: ' .. valGtDir)

        print(" | finding all Set5 testing images")
    elseif opt.testData == "Set14" then
        -- Set14 dataset
         valDir  = paths.concat(opt.genData, 'Set14/train')
         valGtDir  = paths.concat(opt.genData, 'Set14/gt')
        assert(paths.dirp(valDir), 'train directory not found: ' .. valDir)
        assert(paths.dirp(valGtDir), 'gt directory not found: ' .. valGtDir)

        print(" | finding all Set14 testing images")
    elseif opt.testData == "BSD100" then
        -- BSD100 dataset
         valDir  = paths.concat(opt.genData, 'BSD100/train')
         valGtDir  = paths.concat(opt.genData, 'BSD100/gt')
        assert(paths.dirp(valDir), 'train directory not found: ' .. valDir)
        assert(paths.dirp(valGtDir), 'gt directory not found: ' .. valGtDir)

        print(" | finding all BSD100 testing images")        
    
    end

    local valImagePath    = findImages(valDir)
    local valGtPath       = findImages(valGtDir)


    local datasetInfo = {
        train   =   {
            imagePath   =   trainImagePath,
            gtPath      =   trainGtPath,
        },
        val     =   {
            imagePath   =   valImagePath,
            gtPath      =   valGtPath,
        },
    }

    print(" | saving list of images to " .. cacheFile)
    torch.save(cacheFile, datasetInfo)
    return datasetInfo    
end

return M