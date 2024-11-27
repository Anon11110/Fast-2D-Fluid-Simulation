#pragma once
#ifndef RESOURCE_MANAGER_HPP
#define RESOURCE_MANAGER_HPP

#include <liblava/lava.hpp>
#include <memory>
#include <string>
#include <unordered_map>

namespace FluidSimulation
{

struct TextureCreateInfo
{
    glm::uvec2 size;
    VkFormat format;
    VkImageUsageFlags usage;
    VkSamplerAddressMode address_mode;
    VkFilter filter;
    VkSamplerMipmapMode mipmap_mode;
};

struct BufferCreateInfo
{
    VkDeviceSize size;
    VkBufferUsageFlags usage;
    VmaMemoryUsage memory_usage;
    bool mapped = false;
};

class ResourceManager
{
  public:
    static ResourceManager &GetInstance(lava::engine *app = nullptr)
    {
        static ResourceManager instance;
        if (app && !instance.app_)
        {
            instance.app_ = app;
        }
        return instance;
    }

    ResourceManager(const ResourceManager &) = delete;
    ResourceManager &operator=(const ResourceManager &) = delete;
    ResourceManager(ResourceManager &&) = delete;
    ResourceManager &operator=(ResourceManager &&) = delete;

    void CreateTexture(const std::string &name, const TextureCreateInfo &create_info);
    lava::texture::s_ptr GetTexture(const std::string &name);
    void DestroyTexture(const std::string &name);
    bool HasTexture(const std::string &name) const;

    void CreateBuffer(const std::string &name, VkDeviceSize size, VkBufferUsageFlags usage,
                      VmaMemoryUsage memory_usage);
    void CreateMappedBuffer(const std::string &name, void *data, VkDeviceSize size, VkBufferUsageFlags usage,
                            VmaMemoryUsage memory_usage);
    lava::buffer::s_ptr GetBuffer(const std::string &name);
    void DestroyBuffer(const std::string &name);
    bool HasBuffer(const std::string &name) const;
    void InsertBuffer(const std::string &name, lava::buffer::s_ptr buffer)
    {
        if (!buffer)
            throw std::invalid_argument("Cannot insert a null buffer.");
        buffers_[name] = buffer;
    }

    void DestroyAllTextures();
    void DestroyAllBuffers();
    void DestroyAllResources();

    void Initialize(lava::engine *app)
    {
        if (!app_)
        {
            app_ = app;
        }
    }

  private:
    ResourceManager() : app_(nullptr)
    {
    }
    ~ResourceManager()
    {
        DestroyAllResources();
    }

    lava::engine *app_;
    std::unordered_map<std::string, lava::texture::s_ptr> textures_;
    std::unordered_map<std::string, lava::buffer::s_ptr> buffers_;
};

} // namespace FluidSimulation

#endif // RESOURCE_MANAGER_HPP