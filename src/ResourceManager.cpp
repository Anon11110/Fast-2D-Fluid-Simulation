#include "ResourceManager.hpp"

namespace FluidSimulation
{

void ResourceManager::CreateTexture(const std::string &name, const TextureCreateInfo &create_info)
{
    if (!app_)
    {
        throw std::runtime_error("ResourceManager not initialized with engine");
    }

    if (HasTexture(name))
    {
        throw std::runtime_error("Texture already exists: " + name);
    }

    auto texture = lava::texture::make();
    if (!texture->create(app_->device, create_info.size, create_info.format, {}, lava::texture_type::tex_2d,
                         create_info.address_mode, create_info.usage, create_info.filter, create_info.mipmap_mode))
    {
        throw std::runtime_error("Failed to create texture: " + name);
    }
    textures_[name] = texture;
}

lava::texture::s_ptr ResourceManager::GetTexture(const std::string &name)
{
    if (!HasTexture(name))
    {
        throw std::runtime_error("Texture not found: " + name);
    }
    return textures_[name];
}

void ResourceManager::DestroyTexture(const std::string &name)
{
    auto it = textures_.find(name);
    if (it != textures_.end())
    {
        if (it->second)
        {
            it->second->destroy();
        }
        textures_.erase(it);
    }
}

void ResourceManager::DestroyAllTextures()
{
    for (auto &[name, texture] : textures_)
    {
        if (texture)
        {
            texture->destroy();
        }
    }
    textures_.clear();
}

bool ResourceManager::HasTexture(const std::string &name) const
{
    return textures_.find(name) != textures_.end();
}

void ResourceManager::CreateBuffer(const std::string &name, VkDeviceSize size, VkBufferUsageFlags usage,
                                   VmaMemoryUsage memory_usage)
{
    if (!app_)
        throw std::runtime_error("ResourceManager not initialized with engine");

    if (HasBuffer(name))
        throw std::runtime_error("Buffer already exists: " + name);

    auto buffer = lava::buffer::make();
    if (!buffer->create(app_->device, nullptr, size, usage, memory_usage))
    {
        throw std::runtime_error("Failed to create buffer: " + name);
    }
    buffers_[name] = buffer;
}

void ResourceManager::CreateMappedBuffer(const std::string &name, void *data, VkDeviceSize size,
                                         VkBufferUsageFlags usage, VmaMemoryUsage memory_usage)
{
    if (!app_)
        throw std::runtime_error("ResourceManager not initialized with engine");

    if (HasBuffer(name))
        throw std::runtime_error("Buffer already exists: " + name);

    auto buffer = lava::buffer::make();
    if (!buffer->create_mapped(app_->device, data, size, usage, memory_usage))
    {
        throw std::runtime_error("Failed to create mapped buffer: " + name);
    }
    buffers_[name] = buffer;
}
lava::buffer::s_ptr ResourceManager::GetBuffer(const std::string &name)
{
    if (!HasBuffer(name))
    {
        throw std::runtime_error("Buffer not found: " + name);
    }
    return buffers_[name];
}

void ResourceManager::DestroyBuffer(const std::string &name)
{
    auto it = buffers_.find(name);
    if (it != buffers_.end())
    {
        if (it->second)
        {
            it->second->destroy();
        }
        buffers_.erase(it);
    }
}

bool ResourceManager::HasBuffer(const std::string &name) const
{
    return buffers_.find(name) != buffers_.end();
}

void ResourceManager::DestroyAllBuffers()
{
    for (auto &[name, buffer] : buffers_)
    {
        if (buffer)
        {
            buffer->destroy();
        }
    }
    buffers_.clear();
}

void ResourceManager::DestroyAllResources()
{
    DestroyAllTextures();
    DestroyAllBuffers();
}

} // namespace FluidSimulation