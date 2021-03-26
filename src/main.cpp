#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <assert.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan.h>
#include <cmath>
#include <ctime>
#include <stdexcept>
#include <vector>
#include "Bitmap.h"
#include "bilateral.hpp"

const int WORKGROUP_SIZE = 16;

enum mode { cpu, cpuMultiThread, gpu };

#define GPU

#ifdef CPU
constexpr mode mode = CPU;
#elif defined CPU_MULTI_THREAD
constexpr mode = CPU_MULTI_THREAD;
#else
constexpr mode mode = gpu;
#endif

enum storageMode { img, buf };

#define NLM

#ifdef BILATERAL_IMAGE
constexpr char shader[30] = "shaders/bilateral_image.spv\0";
constexpr storageMode storageMode = img;

#elif defined BILATERAL
constexpr char shader[30] = "shaders/bilateral.spv\0";
constexpr storageMode storageMode = buf;
#elif defined NLM
constexpr char shader[30] = "shaders/nlm.spv\0";
constexpr storageMode storageMode = buf;
#else
constexpr char shader[30] = "shaders/nlm_image.spv\0";
constexpr storageMode storageMode = img;
#endif

const char F_IMAGE[100] = "Bathroom_LDR_0001.png\0";
const char FINAL_IMAGE[100] = "images/filtered.jpg\0";

unsigned int WIDTH;
unsigned int HEIGHT;

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

#include "vk_utils.h"

class ComputeApplication {
private:
    struct Pixel {
        float r, g, b, a;
    };

    VkInstance instance;

    VkDebugReportCallbackEXT debugReportCallback;

    VkPhysicalDevice physicalDevice;

    VkDevice device;

    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkShaderModule computeShaderModule;

    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;

    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;

    VkImage image, imageDst;
    VkImageView imageView;
    VkSampler sampler;
    VkDeviceMemory imageMemory;
    VkBuffer bufferGPU, bufferStaging, bufferDynamic;
    VkDeviceMemory bufferMemoryGPU, bufferMemoryStaging, bufferMemoryDynamic;

    std::vector<const char *> enabledLayers;

    VkQueue queue;

    float *pixels;

public:
    void run()
    {
        const int deviceId = 0;
        std::cout << "init vulkan for device " << deviceId << " ... "
                  << std::endl;

        instance =
            vk_utils::CreateInstance(enableValidationLayers, enabledLayers);

        if (enableValidationLayers) {
            vk_utils::InitDebugReportCallback(instance, &debugReportCallbackFn,
                                              &debugReportCallback);
        }

        physicalDevice = vk_utils::FindPhysicalDevice(instance, true, deviceId);

        uint32_t queueFamilyIndex =
            vk_utils::GetComputeQueueFamilyIndex(physicalDevice);

        device = vk_utils::CreateLogicalDevice(queueFamilyIndex, physicalDevice,
                                               enabledLayers);

        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

        if (storageMode == buf) {
            readFile();
            size_t bufferSize = sizeof(Pixel) * WIDTH * HEIGHT;
            std::cout << "creating resources ... " << std::endl;

            createBuffer(device, physicalDevice, bufferSize, &bufferStaging,
                         &bufferMemoryStaging,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

            createBuffer(device, physicalDevice, bufferSize, &bufferGPU,
                         &bufferMemoryGPU, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

            readFileToMemory(device, bufferMemoryStaging);

            createDescriptorSetLayout(
                device, &descriptorSetLayout, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);  // here we will create a
                                                     // binding of buffer to
                                                     // shader via
                                                     // descriptorSet
            createDescriptorSetForOurBuffer(
                device, bufferStaging, bufferGPU, bufferSize,
                &descriptorSetLayout,  // (device, buffer, bufferSize,
                                       // descriptorSetLayout) ==>
                &descriptorPool,
                &descriptorSet);  // (descriptorPool, descriptorSet)
            std::cout << "compiling shaders  ... " << std::endl;
            createComputePipeline(device, descriptorSetLayout,
                                  &computeShaderModule, &pipeline,
                                  &pipelineLayout);

            createCommandBuffer(device, queueFamilyIndex, &commandPool,
                                &commandBuffer);
            recordCommandsTo(commandBuffer, pipeline, pipelineLayout,
                             descriptorSet, device);
            std::time_t t1 = time(nullptr);

            std::cout << "doing computations ... " << std::endl;
            runCommandBuffer(commandBuffer, queue, device);
            std::time_t t2 = time(nullptr);
            std::cout << "saving image       ... " << std::endl;
            saveRenderedImageFromDeviceMemory(device, bufferMemoryGPU, 0, WIDTH,
                                              HEIGHT);
            std::time_t t3 = time(nullptr);
            std::cout << "destroying all     ... " << std::endl;
            std::cout << "Time without copying: " << t2 - t1 << std::endl;
            std::cout << "Time with copying: " << t3 - t1 << std::endl;
            std::cout << "Copying time: " << t3 - t2 << std::endl;
            cleanup();
        }
        else {
            readFile();
            size_t bufferSize = sizeof(Pixel) * WIDTH * HEIGHT;

            std::cout << "creating resources ... " << std::endl;

            createImage(
                WIDTH, HEIGHT, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                image, imageMemory);

            createBuffer(device, physicalDevice, bufferSize, &bufferStaging,
                         &bufferMemoryStaging,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            createBuffer(device, physicalDevice, bufferSize, &bufferGPU,
                         &bufferMemoryGPU,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            createBuffer(device, physicalDevice, bufferSize, &bufferDynamic,
                         &bufferMemoryDynamic,
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            readFileToMemory(device, bufferMemoryDynamic);

            createImageView(image, imageView);
            createTextureSampler(sampler);

            createDescriptorSetLayout(
                device, &descriptorSetLayout, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);  // here we will
                                                             // create a binding
                                                             // of buffer to
                                                             // shader via
                                                             // descriptorSet
            createDescriptorSetForImages(
                device, image, bufferGPU, imageView, sampler, bufferSize,
                &descriptorSetLayout,  // (device, buffer, bufferSize,
                                       // descriptorSetLayout) ==>
                &descriptorPool,
                &descriptorSet);  // (descriptorPool, descriptorSet)

            std::cout << "compiling shaders  ... " << std::endl;

            createComputePipeline(device, descriptorSetLayout,
                                  &computeShaderModule, &pipeline,
                                  &pipelineLayout);

            createCommandBuffer(device, queueFamilyIndex, &commandPool,
                                &commandBuffer);
            vkResetCommandBuffer(commandBuffer, 0);
            RecordCommandsOfCopyImageDataToTexture(
                commandBuffer, pipeline, WIDTH, HEIGHT,
                bufferDynamic,  // bufferDynamic ==> imageGPU
                &image, bufferStaging);

            runCommandBuffer(commandBuffer, queue, device);

            std::cout << "doing computations ... " << std::endl;
            RecordCommandsOfExecuteAndTransfer(
                commandBuffer, pipeline, pipelineLayout, descriptorSet, image,
                bufferSize, bufferGPU, bufferStaging);
            std::time_t t1 = time(nullptr);
            runCommandBuffer(commandBuffer, queue, device);
            std::cout << "saving image       ... " << std::endl;
            std::time_t t2 = time(nullptr);
            saveRenderedImageFromDeviceMemory(device, bufferMemoryStaging, 0,
                                              WIDTH, HEIGHT);
            std::time_t t3 = time(nullptr);
            std::cout << "destroying all     ... " << std::endl;
            std::cout << "Time without copying: " << t2 - t1 << std::endl;
            std::cout << "Time with copying: " << t3 - t1 << std::endl;
            std::cout << "Copying time: " << t3 - t2 << std::endl;
            cleanupImage();
        }
    }

    static void saveRenderedImageFromDeviceMemory(VkDevice a_device,
                                                  VkDeviceMemory a_bufferMemory,
                                                  size_t a_offset, int a_width,
                                                  int a_height)
    {
        const int a_bufferSize = a_width * sizeof(Pixel);
        void *mappedMemory = nullptr;
        std::vector<unsigned char> image;
        image.reserve(a_width * a_height * 4);

        for (int i = 0; i < a_height; ++i) {
            size_t offset = a_offset + i * a_width * sizeof(Pixel);

            mappedMemory = nullptr;

            vkMapMemory(a_device, a_bufferMemory, offset, a_bufferSize, 0,
                        &mappedMemory);
            Pixel *pmappedMemory = (Pixel *)mappedMemory;
            for (int j = 0; j < a_width; j += 1) {
                image.push_back(
                    ((unsigned char)(255.0f * (pmappedMemory[j].r))));
                image.push_back(
                    ((unsigned char)(255.0f * (pmappedMemory[j].g))));
                image.push_back(
                    ((unsigned char)(255.0f * (pmappedMemory[j].b))));
                image.push_back(
                    ((unsigned char)(255.0f * (pmappedMemory[j].a))));
            }
            vkUnmapMemory(a_device, a_bufferMemory);
        }
        stbi_write_jpg(FINAL_IMAGE, a_width, a_height, 4, &image[0], 100);
    }
    static void saveRenderedImageFromDeviceMemoryImage(
        VkDevice a_device, VkDeviceMemory a_bufferMemory, size_t a_offset,
        int a_width, int a_height)
    {
        const int a_bufferSize = a_width * a_height * sizeof(Pixel);
        std::cout << a_bufferSize << "AAA" << std::endl;
        void *mappedMemory = nullptr;
        std::vector<unsigned char> image;
        image.reserve(a_width * a_height * 4);
        vkMapMemory(a_device, a_bufferMemory, 0, a_bufferSize, 0,
                    &mappedMemory);
        Pixel *pmappedMemory = (Pixel *)mappedMemory;

        for (int i = 0; i < a_height; ++i) {
            for (int j = 0; j < a_width * a_height; j += a_height) {
                image.push_back(
                    ((unsigned char)255.0f * (pmappedMemory[j + i].r)));
                image.push_back(
                    ((unsigned char)255.0f * (pmappedMemory[j + i].g)));
                image.push_back(
                    ((unsigned char)255.0f * (pmappedMemory[j + i].b)));
                image.push_back(
                    ((unsigned char)255.0f * (pmappedMemory[j + i].a)));
            }
        }

        vkUnmapMemory(a_device, a_bufferMemory);
        stbi_write_jpg(FINAL_IMAGE, a_width, a_height, 4, &image[0], 100);
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
        VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType,
        uint64_t object, size_t location, int32_t messageCode,
        const char *pLayerPrefix, const char *pMessage, void *pUserData)
    {
        printf("Debug Report: %s: %s\n", pLayerPrefix, pMessage);
        return VK_FALSE;
    }

    static void createBuffer(VkDevice a_device, VkPhysicalDevice a_physDevice,
                             const size_t a_bufferSize, VkBuffer *a_pBuffer,
                             VkDeviceMemory *a_pBufferMemory,
                             VkBufferUsageFlags usage)
    {
        VkBufferCreateInfo bufferCreateInfo = {};
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = a_bufferSize;  // buffer size in bytes.
        bufferCreateInfo.usage = usage;        // buffer is used as a storage
                                               // buffer.
        bufferCreateInfo.sharingMode =
            VK_SHARING_MODE_EXCLUSIVE;  // buffer is exclusive to a single queue
                                        // family at a time.

        VK_CHECK_RESULT(vkCreateBuffer(a_device, &bufferCreateInfo, NULL,
                                       a_pBuffer));  // create buffer.

        VkMemoryRequirements memoryRequirements;
        vkGetBufferMemoryRequirements(a_device, (*a_pBuffer),
                                      &memoryRequirements);

        VkMemoryAllocateInfo allocateInfo = {};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize =
            memoryRequirements.size;  // specify required memory.
        allocateInfo.memoryTypeIndex =
            vk_utils::FindMemoryType(memoryRequirements.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                     a_physDevice);
        std::cout << memoryRequirements.size << std::endl;
        VK_CHECK_RESULT(
            vkAllocateMemory(a_device, &allocateInfo, NULL,
                             a_pBufferMemory));  // allocate memory on device.

        VK_CHECK_RESULT(
            vkBindBufferMemory(a_device, (*a_pBuffer), (*a_pBufferMemory), 0));
    }

    void createImage(uint32_t width, uint32_t height, VkImageTiling tiling,
                     VkImageLayout layout, VkImageUsageFlags usage,
                     VkImage &image, VkDeviceMemory &imageMemory)
    {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = layout;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VK_CHECK_RESULT(vkCreateImage(device, &imageInfo, nullptr, &image));

        VkMemoryRequirements memoryRequirements;
        vkGetImageMemoryRequirements(device, image, &memoryRequirements);
        std::cout << memoryRequirements.size << std::endl;
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memoryRequirements.size;
        allocInfo.memoryTypeIndex =
            vk_utils::FindMemoryType(memoryRequirements.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                     physicalDevice);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(device, image, imageMemory, 0);
    }

    void readFileToMemory(VkDevice &device, VkDeviceMemory &bufMemory)
    {
        VkDeviceSize bufSize = WIDTH * HEIGHT * 4 * sizeof(float);
        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        void *data;
        vkMapMemory(device, bufMemory, 0, bufSize, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(bufSize));
        vkUnmapMemory(device, bufMemory);
        stbi_image_free(pixels);
    }

    void readFile()
    {
        int texChannels;
        pixels = stbi_loadf(F_IMAGE, (int *)&WIDTH, (int *)&HEIGHT,
                            &texChannels, STBI_rgb_alpha);
    }

    static void copyImageToImage(VkCommandBuffer &commandBuffer,
                                 VkImage &srcImage, VkImage &dstImage)
    {
        std::cout << time(NULL) << std::endl;

        VkImageSubresourceLayers imgSub;
        imgSub.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imgSub.baseArrayLayer = 0;
        imgSub.layerCount = 1;
        imgSub.mipLevel = 0;
        VkImageSubresourceLayers imgSub2;
        imgSub2.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imgSub2.baseArrayLayer = 0;
        imgSub2.layerCount = 1;
        imgSub2.mipLevel = 0;

        VkOffset3D offset;
        offset.x = 0;
        offset.y = 0;
        offset.z = 0;

        VkExtent3D extent;
        extent.depth = 1;
        extent.width = WIDTH;
        extent.height = HEIGHT;

        VkImageCopy imgCopy;
        imgCopy.srcSubresource = imgSub;
        imgCopy.srcOffset = offset;
        imgCopy.dstSubresource = imgSub2;
        imgCopy.dstOffset = offset;
        imgCopy.extent = extent;
        vkCmdCopyImage(commandBuffer, srcImage,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imgCopy);
    }
    static void copyImageToBuffer(VkCommandBuffer &commandBuffer,
                                  VkImage &srcImage, VkBuffer &dstBuffer)
    {
        std::cout << time(NULL) << std::endl;

        VkImageSubresourceLayers imgSub;
        imgSub.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imgSub.baseArrayLayer = 0;
        imgSub.layerCount = 1;
        imgSub.mipLevel = 0;

        VkOffset3D offset;
        offset.x = 0;
        offset.y = 0;
        offset.z = 0;

        VkExtent3D extent;
        extent.depth = 1;
        extent.width = WIDTH;
        extent.height = HEIGHT;

        VkBufferImageCopy imgBufInfo;
        imgBufInfo.imageOffset = offset;
        imgBufInfo.bufferOffset = 0;
        imgBufInfo.imageExtent = extent;
        imgBufInfo.bufferRowLength = WIDTH;
        imgBufInfo.bufferImageHeight = HEIGHT;
        imgBufInfo.imageSubresource = imgSub;

        vkCmdCopyImageToBuffer(commandBuffer, srcImage,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstBuffer,
                               1, &imgBufInfo);
    }

    void createImageView(VkImage &image, VkImageView &imView)
    {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = image;
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;
        VK_CHECK_RESULT(
            vkCreateImageView(device, &createInfo, nullptr, &imView));
    }

    void createTextureSampler(VkSampler &sampler)
    {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.pNext = nullptr;
        samplerInfo.flags = 0;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.compareOp = VK_COMPARE_OP_NEVER;
        samplerInfo.minLod = 0;
        samplerInfo.maxLod = 0;
        samplerInfo.maxAnisotropy = 1.0;
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        samplerInfo.unnormalizedCoordinates = VK_TRUE;

        if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }
    }

    static void createDescriptorSetLayout(VkDevice a_device,
                                          VkDescriptorSetLayout *a_pDSLayout,
                                          VkDescriptorType descriptorType1,
                                          VkDescriptorType descriptorType2)
    {
        VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {};
        descriptorSetLayoutBinding.binding = 0;  // binding = 0
        descriptorSetLayoutBinding.descriptorType = descriptorType1;
        descriptorSetLayoutBinding.descriptorCount = 1;
        descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding descriptorSetLayoutBinding2 = {};
        descriptorSetLayoutBinding2.binding = 1;  // binding = 0
        descriptorSetLayoutBinding2.descriptorType = descriptorType2;
        descriptorSetLayoutBinding2.descriptorCount = 1;
        descriptorSetLayoutBinding2.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding pointer[2] = {descriptorSetLayoutBinding,
                                                   descriptorSetLayoutBinding2};

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
        descriptorSetLayoutCreateInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.bindingCount =
            2;  // only a single binding in this descriptor set layout.
        descriptorSetLayoutCreateInfo.pBindings = pointer;

        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
            a_device, &descriptorSetLayoutCreateInfo, NULL, a_pDSLayout));
    }

    void createDescriptorSetForOurBuffer(
        VkDevice a_device, VkBuffer a_buffer, VkBuffer a_buffer2,
        size_t a_bufferSize, const VkDescriptorSetLayout *a_pDSLayout,
        VkDescriptorPool *a_pDSPool, VkDescriptorSet *a_pDS)
    {
        VkDescriptorPoolSize descriptorPoolSize = {};
        descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorPoolSize.descriptorCount = 2;

        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
        descriptorPoolCreateInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.maxSets =
            1;  // we only need to allocate one descriptor set from the pool.
        descriptorPoolCreateInfo.poolSizeCount = 1;
        descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;

        VK_CHECK_RESULT(vkCreateDescriptorPool(
            a_device, &descriptorPoolCreateInfo, NULL, a_pDSPool));

        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
        descriptorSetAllocateInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocateInfo.descriptorPool =
            (*a_pDSPool);  // pool to allocate from.
        descriptorSetAllocateInfo.descriptorSetCount =
            1;  // allocate a single descriptor set.
        descriptorSetAllocateInfo.pSetLayouts = a_pDSLayout;

        VK_CHECK_RESULT(vkAllocateDescriptorSets(
            a_device, &descriptorSetAllocateInfo, a_pDS));

        VkDescriptorBufferInfo descriptorBufferInfo;
        descriptorBufferInfo.buffer = a_buffer;
        descriptorBufferInfo.range = VK_WHOLE_SIZE;
        descriptorBufferInfo.offset = 0;

        VkDescriptorBufferInfo descriptorBufferInfo2;
        descriptorBufferInfo2.buffer = a_buffer2;
        descriptorBufferInfo2.range = VK_WHOLE_SIZE;
        descriptorBufferInfo2.offset = 0;

        VkWriteDescriptorSet writeDescriptorSet = {};
        writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.dstSet = (*a_pDS);  // write to this descriptor set.
        writeDescriptorSet.dstBinding =
            0;  // write to the first, and only binding.
        writeDescriptorSet.descriptorCount = 1;  // update a single descriptor.
        writeDescriptorSet.descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;  // storage buffer.
        writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;

        VkWriteDescriptorSet writeDescriptorSet2 = {};
        writeDescriptorSet2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet2.dstSet = (*a_pDS);  // write to this descriptor set.
        writeDescriptorSet2.dstBinding =
            1;  // write to the first, and only binding.
        writeDescriptorSet2.descriptorCount = 1;  // update a single descriptor.
        writeDescriptorSet2.descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;  // storage buffer.

        writeDescriptorSet2.pBufferInfo = &descriptorBufferInfo2;

        VkWriteDescriptorSet p_Write[2] = {writeDescriptorSet,
                                           writeDescriptorSet2};

        vkUpdateDescriptorSets(a_device, 2, p_Write, 0, NULL);
    }

    void createDescriptorSetForImages(VkDevice a_device, VkImage &imageSrc,
                                      VkBuffer &buffer, VkImageView &iViewSrc,
                                      VkSampler &sampler, size_t a_imageSize,
                                      const VkDescriptorSetLayout *a_pDSLayout,
                                      VkDescriptorPool *a_pDSPool,
                                      VkDescriptorSet *a_pDS)
    {
        VkDescriptorPoolSize descriptorPoolSize = {};
        descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorPoolSize.descriptorCount = 1;

        VkDescriptorPoolSize descriptorPoolSize2 = {};
        descriptorPoolSize2.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorPoolSize2.descriptorCount = 1;

        VkDescriptorPoolSize pool[2] = {descriptorPoolSize,
                                        descriptorPoolSize2};

        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
        descriptorPoolCreateInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.maxSets =
            2;  // we only need to allocate one descriptor set from the pool.
        descriptorPoolCreateInfo.poolSizeCount = 2;
        descriptorPoolCreateInfo.pPoolSizes = pool;

        VK_CHECK_RESULT(vkCreateDescriptorPool(
            a_device, &descriptorPoolCreateInfo, NULL, a_pDSPool));

        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
        descriptorSetAllocateInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocateInfo.descriptorPool =
            (*a_pDSPool);  // pool to allocate from.
        descriptorSetAllocateInfo.descriptorSetCount =
            1;  // allocate a single descriptor set.
        descriptorSetAllocateInfo.pSetLayouts = a_pDSLayout;

        VK_CHECK_RESULT(vkAllocateDescriptorSets(
            a_device, &descriptorSetAllocateInfo, a_pDS));

        VkDescriptorImageInfo descriptorImageInfo;
        descriptorImageInfo.imageLayout =
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        descriptorImageInfo.imageView = iViewSrc;
        descriptorImageInfo.sampler = sampler;

        VkDescriptorBufferInfo descriptorBufferInfo;
        descriptorBufferInfo.buffer = buffer;
        descriptorBufferInfo.range = VK_WHOLE_SIZE;
        descriptorBufferInfo.offset = 0;

        VkWriteDescriptorSet writeDescriptorSet = {};
        writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.dstSet = (*a_pDS);  // write to this descriptor set.
        writeDescriptorSet.dstBinding =
            1;  // write to the first, and only binding.
        writeDescriptorSet.descriptorCount = 1;  // update a single descriptor.
        writeDescriptorSet.descriptorType =
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;  // storage buffer.
        writeDescriptorSet.pImageInfo = &descriptorImageInfo;

        VkWriteDescriptorSet writeDescriptorSet2 = {};
        writeDescriptorSet2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet2.dstSet = (*a_pDS);  // write to this descriptor set.
        writeDescriptorSet2.dstBinding =
            0;  // write to the first, and only binding.
        writeDescriptorSet2.descriptorCount = 1;  // update a single descriptor.
        writeDescriptorSet2.descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;  // storage buffer.

        writeDescriptorSet2.pBufferInfo = &descriptorBufferInfo;

        VkWriteDescriptorSet p_Write[2] = {writeDescriptorSet,
                                           writeDescriptorSet2};
        vkUpdateDescriptorSets(a_device, 2, p_Write, 0, NULL);
    }
    static void createComputePipeline(VkDevice a_device,
                                      const VkDescriptorSetLayout &a_dsLayout,
                                      VkShaderModule *a_pShaderModule,
                                      VkPipeline *a_pPipeline,
                                      VkPipelineLayout *a_pPipelineLayout)
    {
        std::vector<uint32_t> code = vk_utils::ReadFile(shader);
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.pCode = code.data();
        createInfo.codeSize = code.size() * sizeof(uint32_t);

        VK_CHECK_RESULT(
            vkCreateShaderModule(a_device, &createInfo, NULL, a_pShaderModule));

        VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
        shaderStageCreateInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStageCreateInfo.module = (*a_pShaderModule);
        shaderStageCreateInfo.pName = "main";

        VkPushConstantRange pcRange =
            {};  // #NOTE: we updated this to pass W/H inside shader
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = 2 * sizeof(int);

        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
        pipelineLayoutCreateInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCreateInfo.setLayoutCount = 1;
        pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
        pipelineLayoutCreateInfo.pPushConstantRanges = &pcRange;
        pipelineLayoutCreateInfo.pSetLayouts = &a_dsLayout;
        VK_CHECK_RESULT(vkCreatePipelineLayout(
            a_device, &pipelineLayoutCreateInfo, NULL, a_pPipelineLayout));

        VkComputePipelineCreateInfo pipelineCreateInfo = {};
        pipelineCreateInfo.sType =
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.stage = shaderStageCreateInfo;
        pipelineCreateInfo.layout = (*a_pPipelineLayout);

        VK_CHECK_RESULT(vkCreateComputePipelines(a_device, VK_NULL_HANDLE, 1,
                                                 &pipelineCreateInfo, NULL,
                                                 a_pPipeline));
    }

    static void createCommandBuffer(VkDevice a_device,
                                    uint32_t queueFamilyIndex,
                                    VkCommandPool *a_pool,
                                    VkCommandBuffer *a_pCmdBuff)
    {
        VkCommandPoolCreateInfo commandPoolCreateInfo = {};
        commandPoolCreateInfo.sType =
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.flags =
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
        VK_CHECK_RESULT(vkCreateCommandPool(a_device, &commandPoolCreateInfo,
                                            NULL, a_pool));

        VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
        commandBufferAllocateInfo.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.commandPool =
            (*a_pool);  // specify the command pool to allocate from.

        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount =
            1;  // allocate a single command buffer.
        VK_CHECK_RESULT(
            vkAllocateCommandBuffers(a_device, &commandBufferAllocateInfo,
                                     a_pCmdBuff)); 
    }

    static void recordCommandsTo(VkCommandBuffer a_cmdBuff,
                                 VkPipeline a_pipeline,
                                 VkPipelineLayout a_layout,
                                 const VkDescriptorSet &a_ds, VkDevice device)
    {
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags =
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;  
        VK_CHECK_RESULT(vkBeginCommandBuffer(
            a_cmdBuff, &beginInfo));  

        vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE,
                          a_pipeline);
        vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE,
                                a_layout, 0, 1, &a_ds, 0, NULL);
        int wh[2] = {(int)WIDTH, (int)HEIGHT};
        vkCmdPushConstants(a_cmdBuff, a_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(int) * 2, wh);
        vkCmdDispatch(a_cmdBuff, (uint32_t)ceil(WIDTH / float(WORKGROUP_SIZE)),
                      (uint32_t)ceil(HEIGHT / float(WORKGROUP_SIZE)), 1);

        VK_CHECK_RESULT(
            vkEndCommandBuffer(a_cmdBuff)); 
    }

    static VkImageMemoryBarrier imBarTransfer(
        VkImage a_image, const VkImageSubresourceRange &a_range,
        VkImageLayout before, VkImageLayout after) 
    {
        VkImageMemoryBarrier moveToGeneralBar = {};
        moveToGeneralBar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        moveToGeneralBar.pNext = nullptr;
        moveToGeneralBar.srcAccessMask = 0;
        moveToGeneralBar.dstAccessMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
        moveToGeneralBar.oldLayout = before;
        moveToGeneralBar.newLayout = after;
        moveToGeneralBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        moveToGeneralBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        moveToGeneralBar.image = a_image;
        moveToGeneralBar.subresourceRange = a_range;
        return moveToGeneralBar;
    }

    static VkImageSubresourceRange WholeImageRange()
    {
        VkImageSubresourceRange rangeWholeImage = {};
        rangeWholeImage.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        rangeWholeImage.baseMipLevel = 0;
        rangeWholeImage.levelCount = 1;
        rangeWholeImage.baseArrayLayer = 0;
        rangeWholeImage.layerCount = 1;
        return rangeWholeImage;
    }

    static void RecordCommandsOfExecuteAndTransfer(
        VkCommandBuffer a_cmdBuff, VkPipeline a_pipeline,
        VkPipelineLayout a_layout, const VkDescriptorSet &a_ds, VkImage a_image,
        size_t a_bufferSize, VkBuffer a_bufferGPU, VkBuffer a_bufferStaging)
    {
      
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags =
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; 
        VK_CHECK_RESULT(vkBeginCommandBuffer(
            a_cmdBuff, &beginInfo)); 

        vkCmdFillBuffer(
            a_cmdBuff, a_bufferStaging, 0, a_bufferSize,
            0); 

     
        vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE,
                          a_pipeline);
        vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE,
                                a_layout, 0, 1, &a_ds, 0, NULL);

        int wh[2] = {(int)WIDTH, (int)HEIGHT};
        vkCmdPushConstants(a_cmdBuff, a_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(int) * 2, wh);
    
        vkCmdDispatch(a_cmdBuff, (uint32_t)ceil(WIDTH / float(WORKGROUP_SIZE)),
                      (uint32_t)ceil(HEIGHT / float(WORKGROUP_SIZE)), 1);

        VkBufferMemoryBarrier bufBarr = {};
        bufBarr.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        bufBarr.pNext = nullptr;
        bufBarr.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bufBarr.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bufBarr.size = VK_WHOLE_SIZE;
        bufBarr.offset = 0;
        bufBarr.buffer = a_bufferGPU;
        bufBarr.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bufBarr.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(a_cmdBuff, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                             &bufBarr, 0, nullptr);

        VkBufferCopy copyInfo = {};
        copyInfo.dstOffset = 0;
        copyInfo.srcOffset = 0;
        copyInfo.size = a_bufferSize;

        vkCmdCopyBuffer(a_cmdBuff, a_bufferGPU, a_bufferStaging, 1, &copyInfo);

        VK_CHECK_RESULT(
            vkEndCommandBuffer(a_cmdBuff));  // end recording commands.
    }

    static void RecordCommandsOfCopyImageDataToTexture(
        VkCommandBuffer a_cmdBuff, VkPipeline a_pipeline, int a_width,
        int a_height, VkBuffer a_bufferDynamic, VkImage *a_images,
        VkBuffer a_bufferStaging)
    {
        //// Now we shall start recording commands into the newly allocated
        /// command bufferStaging.
        //
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags =
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;  // the bufferStaging
                                                          // is only submitted
                                                          // and used once in
                                                          // this application.
        VK_CHECK_RESULT(vkBeginCommandBuffer(
            a_cmdBuff, &beginInfo));  // start recording commands.

        vkCmdFillBuffer(
            a_cmdBuff, a_bufferStaging, 0,
            a_width * a_height * sizeof(float) * 4,
            0);  // clear this buffer just for an example and test cases. if we
                 // comment 'vkCmdCopyBuffer', we'll get black image

        // we want to work with the whole image
        //
        VkImageSubresourceRange rangeWholeImage = WholeImageRange();

        VkImageSubresourceLayers shittylayers = {};
        shittylayers.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        shittylayers.mipLevel = 0;
        shittylayers.baseArrayLayer = 0;
        shittylayers.layerCount = 1;

        VkBufferImageCopy wholeRegion = {};
        wholeRegion.bufferOffset = 0;
        wholeRegion.bufferRowLength = uint32_t(a_width);
        wholeRegion.bufferImageHeight = uint32_t(a_height);
        wholeRegion.imageExtent =
            VkExtent3D{uint32_t(a_width), uint32_t(a_height), 1};
        wholeRegion.imageOffset = VkOffset3D{0, 0, 0};
        wholeRegion.imageSubresource = shittylayers;

        // at first we must move our images to 'VK_IMAGE_LAYOUT_GENERAL' layout
        // to further clear them
        //
        VkImageMemoryBarrier moveToGeneralBar = imBarTransfer(
            a_images[0], rangeWholeImage, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        vkCmdPipelineBarrier(a_cmdBuff, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                             nullptr,                // general memory barriers
                             0, nullptr,             // buffer barriers
                             1, &moveToGeneralBar);  // image  barriers

        // now we can clear images
        //
        VkClearColorValue clearVal = {};
        clearVal.float32[0] = 1.0f;
        clearVal.float32[1] = 1.0f;
        clearVal.float32[2] = 1.0f;
        clearVal.float32[3] = 1.0f;

        vkCmdClearColorImage(a_cmdBuff, a_images[0],
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearVal, 1,
                             &rangeWholeImage);  // clear image with (1,1,1,1)

        vkCmdCopyBufferToImage(a_cmdBuff, a_bufferDynamic, a_images[0],
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &wholeRegion);

        // transfer our texture from VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL to
        // VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        VkImageMemoryBarrier imgBar =
            {};  // imBarTransfer(a_images[0], rangeWholeImage,
                 // VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 // VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        {
            imgBar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imgBar.pNext = nullptr;
            imgBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imgBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

            imgBar.srcAccessMask = 0;
            imgBar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            imgBar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            imgBar.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imgBar.image = a_images[0];

            imgBar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            imgBar.subresourceRange.baseMipLevel = 0;
            imgBar.subresourceRange.levelCount = 1;
            imgBar.subresourceRange.baseArrayLayer = 0;
            imgBar.subresourceRange.layerCount = 1;
        };

        vkCmdPipelineBarrier(a_cmdBuff, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &imgBar);

        VK_CHECK_RESULT(
            vkEndCommandBuffer(a_cmdBuff));  // end recording commands.
    }

    static void runCommandBuffer(VkCommandBuffer a_cmdBuff, VkQueue a_queue,
                                 VkDevice a_device)
    {
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;  // submit a single command buffer
        submitInfo.pCommandBuffers =
            &a_cmdBuff;  // the command buffer to submit.

        VkFence fence;
        VkFenceCreateInfo fenceCreateInfo = {};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.flags = 0;
        VK_CHECK_RESULT(
            vkCreateFence(a_device, &fenceCreateInfo, NULL, &fence));
        VK_CHECK_RESULT(vkQueueSubmit(a_queue, 1, &submitInfo, fence));
        VK_CHECK_RESULT(
            vkWaitForFences(a_device, 1, &fence, VK_TRUE, 100000000000));

        vkDestroyFence(a_device, fence, NULL);
    }

    void cleanup()
    {
        if (enableValidationLayers) {
            auto func =
                (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(
                    instance, "vkDestroyDebugReportCallbackEXT");
            if (func == nullptr) {
                throw std::runtime_error(
                    "Could not load vkDestroyDebugReportCallbackEXT");
            }
            func(instance, debugReportCallback, NULL);
        }

        vkFreeMemory(device, bufferMemoryGPU, NULL);
        vkFreeMemory(device, bufferMemoryStaging, NULL);
        vkDestroyBuffer(device, bufferGPU, NULL);
        vkDestroyBuffer(device, bufferStaging, NULL);
        vkDestroyShaderModule(device, computeShaderModule, NULL);
        vkDestroyDescriptorPool(device, descriptorPool, NULL);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
        vkDestroyPipelineLayout(device, pipelineLayout, NULL);
        vkDestroyPipeline(device, pipeline, NULL);
        vkDestroyCommandPool(device, commandPool, NULL);
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
    }

    void cleanupImage()
    {
        if (enableValidationLayers) {
            auto func =
                (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(
                    instance, "vkDestroyDebugReportCallbackEXT");
            if (func == nullptr) {
                throw std::runtime_error(
                    "Could not load vkDestroyDebugReportCallbackEXT");
            }
            func(instance, debugReportCallback, NULL);
        }
        vkFreeMemory(device, imageMemory, NULL);
        vkFreeMemory(device, bufferMemoryDynamic, NULL);
        vkFreeMemory(device, bufferMemoryStaging, NULL);
        vkFreeMemory(device, bufferMemoryGPU, NULL);
        vkDestroySampler(device, sampler, NULL);
        vkDestroyImageView(device, imageView, NULL);
        vkDestroyBuffer(device, bufferStaging, NULL);
        vkDestroyBuffer(device, bufferDynamic, NULL);
        vkDestroyBuffer(device, bufferGPU, NULL);
        vkDestroyImage(device, image, NULL);
        vkDestroyShaderModule(device, computeShaderModule, NULL);
        vkDestroyDescriptorPool(device, descriptorPool, NULL);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
        vkDestroyPipelineLayout(device, pipelineLayout, NULL);
        vkDestroyPipeline(device, pipeline, NULL);
        vkDestroyCommandPool(device, commandPool, NULL);
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
    }
};

class CPUApp {
public:
    static void run()
    {
        struct Pixel {
            float r, g, b, a;
        };

        int texChannels;
        float *oldData = stbi_loadf(F_IMAGE, (int *)&WIDTH, (int *)&HEIGHT,
                                    &texChannels, STBI_rgb_alpha);
        float *newData = (float *)malloc(WIDTH * HEIGHT * 4 * sizeof(*newData));
        BilateralFilter b(oldData, newData, WIDTH, HEIGHT);
        b.run();

        std::vector<unsigned char> image;
        image.reserve(WIDTH * HEIGHT * 4);

        for (int i = 0; i < HEIGHT; ++i) {
            for (int j = 0; j < WIDTH; j += 1) {
                image.push_back(
                    (unsigned char)(255.0f *
                                    newData[4 * WIDTH * i + 4 * j + 0]));
                image.push_back(
                    (unsigned char)(255.0f *
                                    newData[4 * WIDTH * i + 4 * j + 1]));
                image.push_back(
                    (unsigned char)(255.0f *
                                    newData[4 * WIDTH * i + 4 * j + 2]));
                image.push_back(
                    (unsigned char)(255.0f *
                                    newData[4 * WIDTH * i + 4 * j + 3]));
            }
        }
        stbi_write_png(FINAL_IMAGE, WIDTH, HEIGHT, 4, &image[0], WIDTH * 4);
        std::cout << newData[0] << std::endl;
        free(newData);
        free(oldData);
    }
};

int main()
{
    if (mode == gpu) {
        ComputeApplication app;

        try {
            app.run();
        }
        catch (const std::runtime_error &e) {
            printf("%s\n", e.what());
            return EXIT_FAILURE;
        }
    }
    else {
        CPUApp app;

        try {
            app.run();
        }
        catch (const std::runtime_error &e) {
            printf("%s\n", e.what());
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
