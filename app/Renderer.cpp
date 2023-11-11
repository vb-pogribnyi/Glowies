/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#include <sstream>
#include <exception>


#define STB_IMAGE_IMPLEMENTATION
#include "obj_loader.h"
#include "stb_image.h"

#include "Renderer.h"
#include "nvh/alignment.hpp"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvk/buffers_vk.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


extern std::vector<std::string> defaultSearchPaths;

void Renderer::prepareFrame()
{
  if (is_rebuild_tlas) m_rtBuilder.buildTlas(m_tlas, m_rtFlags, true);
  is_rebuild_tlas = false;
  nvvkhl::AppBaseVk::prepareFrame();
}

//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void Renderer::setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, 
    uint32_t queueFamily)
{
  AppBaseVk::setup(instance, device, physicalDevice, queueFamily);
  m_alloc.init(instance, device, physicalDevice);
  m_debug.setup(m_device);
  m_offscreenDepthFormat = nvvk::findDepthFormat(physicalDevice);
  is_rebuild_tlas = false;
}

void Renderer::loadModels(uint32_t nParticles) {
  indices.cube_pos_idx = loadModel(nvh::findFile("media/scenes/cube.obj", defaultSearchPaths, true),
                    nvmath::translation_mat4(nvmath::vec3f(1, 0.5, 0)) * 
                    nvmath::scale_mat4(nvmath::vec3f(0.0f, 0.0f, 0.0f)),
                    MODEL_POSITIVE);
  indices.cube_neg_idx = loadModel(nvh::findFile("media/scenes/cube.obj", defaultSearchPaths, true),
                    nvmath::translation_mat4(nvmath::vec3f(1, 0.5, 0)) * 
                    nvmath::scale_mat4(nvmath::vec3f(0.0f, 0.0f, 0.0f)),
                    MODEL_NEGATIVE);
  indices.cube_pos_prt_idx = loadModel(nvh::findFile("media/scenes/cube.obj", defaultSearchPaths, true),
                    nvmath::translation_mat4(nvmath::vec3f(1, 0.5, 0)) * 
                    nvmath::scale_mat4(nvmath::vec3f(0.0f, 0.0f, 0.0f)),
                    MODEL_POSITIVE | MODEL_PARTIAL);
  indices.cube_neg_prt_idx = loadModel(nvh::findFile("media/scenes/cube.obj", defaultSearchPaths, true),
                    nvmath::translation_mat4(nvmath::vec3f(1, 0.5, 0)) * 
                    nvmath::scale_mat4(nvmath::vec3f(0.0f, 0.0f, 0.0f)),
                    MODEL_NEGATIVE | MODEL_PARTIAL);
  indices.filler_idx = loadModel(nvh::findFile("media/scenes/cube.obj", defaultSearchPaths, true),
                    nvmath::scale_mat4(nvmath::vec3f(0.0f, 0.0f, 0.0f)),
                    MODEL_FILLER);
  indices.glass_idx = loadModel(nvh::findFile("media/scenes/cube.obj", defaultSearchPaths, true),
                    nvmath::translation_mat4(nvmath::vec3f(1, 0.4, 0)) * 
                    nvmath::scale_mat4(nvmath::vec3f(0.0f, 0.0f, 0.0f)),
                    MODEL_GLASS);
  indices.particle_pos_idx = loadModel(nvh::findFile("media/scenes/particle.obj", defaultSearchPaths, true),
                    nvmath::translation_mat4(nvmath::vec3f(1, 0.5, 0)) * 
                    nvmath::scale_mat4(nvmath::vec3f(.0f, .0f, .0f)),
                    MODEL_POSITIVE | MODEL_GLOWING);
  indices.particle_pos_shell_idx = loadModel(nvh::findFile("media/scenes/particle.obj", defaultSearchPaths, true),
                    nvmath::translation_mat4(nvmath::vec3f(1, 0.5, 0)) * 
                    nvmath::scale_mat4(nvmath::vec3f(.0f, .0f, .0f)),
                    MODEL_POSITIVE | MODEL_GLOWING | MODEL_SHELL);
  indices.particle_neg_idx = loadModel(nvh::findFile("media/scenes/particle.obj", defaultSearchPaths, true),
                    nvmath::translation_mat4(nvmath::vec3f(1, 0.5, 0)) * 
                    nvmath::scale_mat4(nvmath::vec3f(.0f, .0f, .0f)),
                    MODEL_NEGATIVE | MODEL_GLOWING);
  indices.particle_neg_shell_idx = loadModel(nvh::findFile("media/scenes/particle.obj", defaultSearchPaths, true),
                    nvmath::translation_mat4(nvmath::vec3f(1, 0.5, 0)) * 
                    nvmath::scale_mat4(nvmath::vec3f(.0f, .0f, .0f)),
                    MODEL_NEGATIVE | MODEL_GLOWING | MODEL_SHELL);
  indices.particle_neutral_idx = loadModel(nvh::findFile("media/scenes/particle.obj", defaultSearchPaths, true),
                    nvmath::translation_mat4(nvmath::vec3f(1, 0.5, 0)) * 
                    nvmath::scale_mat4(nvmath::vec3f(.0f, .0f, .0f)),
                    MODEL_NEUTRAL);
          
  for (int i = 0; i < nParticles; i++) {
    ParticleIdxs idxs;
    // Positive particle
    m_instances.push_back({nvmath::translation_mat4(nvmath::vec3f(0.0f)) * 
                    nvmath::scale_mat4(nvmath::vec3f(0.0f)), indices.particle_pos_idx, 0});
    idxs.particle_signed = m_instances.size() - 1;
    m_instances.push_back({nvmath::translation_mat4(nvmath::vec3f(0.0f)) * 
                    nvmath::scale_mat4(nvmath::vec3f(0.0f)), indices.particle_pos_shell_idx, 0});
    idxs.shell = m_instances.size() - 1;
    m_instances.push_back({nvmath::translation_mat4(nvmath::vec3f(0.0f)) * 
                    nvmath::scale_mat4(vec3(0.0f)), indices.filler_idx, 0});
    idxs.filler = m_instances.size() - 1;
    m_instances.push_back({nvmath::translation_mat4(nvmath::vec3f(0.0f)) * 
                    nvmath::scale_mat4(vec3(0.0f)), indices.particle_neutral_idx, 0});
    idxs.particle_neutral = m_instances.size() - 1;

    particles_pos_free.push_back(idxs);

    // Negative particle
    m_instances.push_back({nvmath::translation_mat4(nvmath::vec3f(0.0f)) * 
                    nvmath::scale_mat4(nvmath::vec3f(0.0f)), indices.particle_neg_idx, 0});
    idxs.particle_signed = m_instances.size() - 1;
    m_instances.push_back({nvmath::translation_mat4(nvmath::vec3f(0.0f)) * 
                    nvmath::scale_mat4(nvmath::vec3f(0.0f)), indices.particle_neg_shell_idx, 0});
    idxs.shell = m_instances.size() - 1;
    m_instances.push_back({nvmath::translation_mat4(nvmath::vec3f(0.0f)) * 
                    nvmath::scale_mat4(vec3(0.0f)), indices.filler_idx, 0});
    idxs.filler = m_instances.size() - 1;
    m_instances.push_back({nvmath::translation_mat4(nvmath::vec3f(0.0f)) * 
                    nvmath::scale_mat4(vec3(0.0f)), indices.particle_neutral_idx, 0});
    idxs.particle_neutral = m_instances.size() - 1;

    particles_neg_free.push_back(idxs);
  }
}

ParticleIdxs Renderer::getParticle(bool is_positive) {
  ParticleIdxs result;
  // std::cout << "Retrieving particle " << particles_pos_free.size() << ' ' << particles_neg_free.size() << std::endl;
  if (is_positive) {
    if (particles_pos_free.size() == 0) throw std::runtime_error("Positive particles over-allocation");
    result = particles_pos_free.front();
    particles_pos_free.pop_front();
  } else {
    if (particles_neg_free.size() == 0) throw std::runtime_error("Negative particles over-allocation");
    result = particles_neg_free.front();
    particles_neg_free.pop_front();
  }
  // std::cout << "Positive particles left: " << particles_pos_free.size() << ' ' << particles_neg_free.size() << std::endl;
  // std::cout << result.filler << ' ' << result.particle_neutral << ' ' << result.particle_signed << ' '  << result.shell << ' '  << std::endl;

  return result;
}

void Renderer::releaseParticle(bool is_positive, ParticleIdxs idxs) {
  if (is_positive) particles_pos_free.push_back(idxs);
  else particles_neg_free.push_back(idxs);
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void Renderer::updateUniformBuffer(const VkCommandBuffer& cmdBuf)
{
  // Prepare new UBO contents on host.
  const float    aspectRatio = m_size.width / static_cast<float>(m_size.height);
  GlobalUniforms hostUBO     = {};
  const auto&    view        = CameraManip.getMatrix();
  const auto&    proj        = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);
  // proj[1][1] *= -1;  // Inverting Y for Vulkan (not needed with perspectiveVK).

  hostUBO.viewProj    = proj * view;
  hostUBO.viewInverse = nvmath::invert(view);
  hostUBO.projInverse = nvmath::invert(proj);
  hostUBO.focalDist = 10;
  hostUBO.aperture = 0;

  // UBO on the device, and what stages access it.
  VkBuffer deviceUBO      = m_bGlobals.buffer;
  auto     uboUsageStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

  // Ensure that the modified UBO is not visible to previous frames.
  VkBufferMemoryBarrier beforeBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  beforeBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  beforeBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  beforeBarrier.buffer        = deviceUBO;
  beforeBarrier.offset        = 0;
  beforeBarrier.size          = sizeof(hostUBO);
  vkCmdPipelineBarrier(cmdBuf, uboUsageStages, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &beforeBarrier, 0, nullptr);


  // Schedule the host-to-device upload. (hostUBO is copied into the cmd
  // buffer so it is okay to deallocate when the function returns).
  vkCmdUpdateBuffer(cmdBuf, m_bGlobals.buffer, 0, sizeof(GlobalUniforms), &hostUBO);

  // Making sure the updated UBO will be visible.
  VkBufferMemoryBarrier afterBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  afterBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  afterBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  afterBarrier.buffer        = deviceUBO;
  afterBarrier.offset        = 0;
  afterBarrier.size          = sizeof(hostUBO);
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, uboUsageStages, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &afterBarrier, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Describing the layout pushed when rendering
//
void Renderer::createDescriptorSetLayout()
{
  // auto nbTxt = static_cast<uint32_t>(m_textures.size());

  // Camera matrices
  m_descSetLayoutBind.addBinding(SceneBindings::eGlobals, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                 VK_SHADER_STAGE_COMPUTE_BIT);
  // Obj descriptions
  m_descSetLayoutBind.addBinding(SceneBindings::eObjDescs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                  VK_SHADER_STAGE_COMPUTE_BIT);


  m_descSetLayout = m_descSetLayoutBind.createLayout(m_device);
  m_descPool      = m_descSetLayoutBind.createPool(m_device, 1);
  m_descSet       = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Setting up the buffers in the descriptor set
//
void Renderer::updateDescriptorSet()
{
  std::vector<VkWriteDescriptorSet> writes;

  // Camera matrices and scene description
  VkDescriptorBufferInfo dbiUnif{m_bGlobals.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eGlobals, &dbiUnif));

  VkDescriptorBufferInfo dbiSceneDesc{m_bObjDesc.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eObjDescs, &dbiSceneDesc));

  // Writing the information
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Loading the OBJ file and setting up all buffers
//
uint32_t Renderer::loadModel(const std::string& filename, nvmath::mat4f transform, uint64_t flags)
{
  LOGI("Loading File:  %s \n", filename.c_str());
  ObjLoader loader;
  loader.loadModel(filename);

  if (flags & MODEL_POSITIVE) loader.m_materials[0].diffuse = vec3(0.5, 0.01, 0.03);
  if (flags & MODEL_NEGATIVE) loader.m_materials[0].diffuse = vec3(0.03, 0.01, 0.5);
  if (flags & MODEL_GLOWING) loader.m_materials[0].emission = loader.m_materials[0].diffuse;
  if (flags & MODEL_GLOWING) loader.m_materials[0].emission *= 10;
  if (flags & MODEL_NEUTRAL) loader.m_materials[0].emission = vec3(10, 10, 10); // TODO: Switch to warm light
  if (flags & MODEL_GLASS) loader.m_materials[0].transmittance = vec3(1.0, 0, 0);
  if (flags & MODEL_GLASS) loader.m_materials[0].diffuse = vec3(0.9);
  if (flags & MODEL_PARTIAL) loader.m_materials[0].illum = 8;
  if (flags & MODEL_SHELL) loader.m_materials[0].illum = 5;
  if (flags & MODEL_FILLER) loader.m_materials[0].illum = 4;
  if (flags & MODEL_FILLER) loader.m_materials[0].diffuse = vec3(0.9);
  if (flags & MODEL_FILLER) loader.m_materials[0].transmittance = vec3(0.0, 0.2, 1.0);
  

  // Converting from Srgb to linear
  for(auto& m : loader.m_materials)
  {
    m.ambient  = nvmath::pow(m.ambient, 2.2f);
    m.diffuse  = nvmath::pow(m.diffuse, 2.2f);
    m.specular = nvmath::pow(m.specular, 2.2f);
  }

  ObjModel model;
  model.nbIndices  = static_cast<uint32_t>(loader.m_indices.size());
  model.nbVertices = static_cast<uint32_t>(loader.m_vertices.size());

  // Create the buffers on Device and copy vertices, indices and materials
  nvvk::CommandPool  cmdBufGet(m_device, m_graphicsQueueIndex);
  VkCommandBuffer    cmdBuf = cmdBufGet.createCommandBuffer();
  VkBufferUsageFlags flag   = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  VkBufferUsageFlags rayTracingFlags =  // used also for building acceleration structures
      flag | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  model.vertexBuffer        = m_alloc.createBuffer(cmdBuf, loader.m_vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | rayTracingFlags);
  model.indexBuffer         = m_alloc.createBuffer(cmdBuf, loader.m_indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | rayTracingFlags);
  model.matColorBuffer = m_alloc.createBuffer(cmdBuf, loader.m_materials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | flag);
  model.matIndexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_matIndx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | flag);

  cmdBufGet.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();

  std::string objNb = std::to_string(m_objModel.size());
  m_debug.setObjectName(model.vertexBuffer.buffer, (std::string("vertex_" + objNb)));
  m_debug.setObjectName(model.indexBuffer.buffer, (std::string("index_" + objNb)));
  m_debug.setObjectName(model.matColorBuffer.buffer, (std::string("mat_" + objNb)));
  m_debug.setObjectName(model.matIndexBuffer.buffer, (std::string("matIdx_" + objNb)));

  // Keeping transformation matrix of the instance
  ObjInstance instance;
  instance.transform = transform;
  instance.objIndex  = static_cast<uint32_t>(m_objModel.size());
  m_instances.push_back(instance);

  // Creating information for device access
  ObjDesc desc;
  desc.vertexAddress        = nvvk::getBufferDeviceAddress(m_device, model.vertexBuffer.buffer);
  desc.indexAddress         = nvvk::getBufferDeviceAddress(m_device, model.indexBuffer.buffer);
  desc.materialAddress      = nvvk::getBufferDeviceAddress(m_device, model.matColorBuffer.buffer);
  desc.materialIndexAddress = nvvk::getBufferDeviceAddress(m_device, model.matIndexBuffer.buffer);

  // Keeping the obj host model and device description
  m_objModel.emplace_back(model);
  m_objDesc.emplace_back(desc);

  // Return the instance id
  return m_instances.size() - 1;
}


//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the camera matrices
// - Buffer is host visible
//
void Renderer::createUniformBuffer()
{
  m_bGlobals = m_alloc.createBuffer(sizeof(GlobalUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_bGlobals.buffer, "Globals");
}

//--------------------------------------------------------------------------------------------------
// Create a storage buffer containing the description of the scene elements
// - Which geometry is used by which instance
// - Transformation
// - Offset for texture
//
void Renderer::createObjDescriptionBuffer()
{
  nvvk::CommandPool cmdGen(m_device, m_graphicsQueueIndex);

  auto cmdBuf = cmdGen.createCommandBuffer();
  m_bObjDesc  = m_alloc.createBuffer(cmdBuf, m_objDesc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  cmdGen.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();
  m_debug.setObjectName(m_bObjDesc.buffer, "ObjDescs");
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void Renderer::destroyResources()
{
  m_alloc.destroy(m_bGlobals);
  m_alloc.destroy(m_bObjDesc);

  for(auto& m : m_objModel)
  {
    m_alloc.destroy(m.vertexBuffer);
    m_alloc.destroy(m.indexBuffer);
    m_alloc.destroy(m.matColorBuffer);
    m_alloc.destroy(m.matIndexBuffer);
  }

  //#Post
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);
  vkDestroyPipeline(m_device, m_postPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_postPipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_postDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_postDescSetLayout, nullptr);
  vkDestroyRenderPass(m_device, m_offscreenRenderPass, nullptr);
  vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);


  // #VKRay
  m_rtBuilder.destroy();
  vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
  vkDestroyPipeline(m_device, m_rtPipeline_simpli, nullptr);
  vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_rtDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_rtDescSetLayout, nullptr);

  // TODO: No more needed with SBT wrapper?
  m_alloc.destroy(m_rtSBTBuffer);

  m_alloc.deinit();
}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void Renderer::onResize(int /*w*/, int /*h*/)
{
  createOffscreenRender();
  updatePostDescriptorSet();
  updateRtDescriptorSet();
}


//////////////////////////////////////////////////////////////////////////
// Post-processing
//////////////////////////////////////////////////////////////////////////


//--------------------------------------------------------------------------------------------------
// Creating an offscreen frame buffer and the associated render pass
//
void Renderer::createOffscreenRender()
{
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);

  // Creating the color image
  {
    auto colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenColorFormat,
                                                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                                           | VK_IMAGE_USAGE_STORAGE_BIT);

    nvvk::Image           image  = m_alloc.createImage(colorCreateInfo);
    VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    VkSamplerCreateInfo   sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    m_offscreenColor                        = m_alloc.createTexture(image, ivInfo, sampler);
    m_offscreenColor.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  // Creating the depth buffer
  auto depthCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  {
    nvvk::Image image = m_alloc.createImage(depthCreateInfo);


    VkImageViewCreateInfo depthStencilView{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    depthStencilView.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    depthStencilView.format           = m_offscreenDepthFormat;
    depthStencilView.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    depthStencilView.image            = image.image;

    m_offscreenDepth = m_alloc.createTexture(image, depthStencilView);
  }

  // Setting the image layout for both color and depth
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenColor.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenDepth.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

    genCmdBuf.submitAndWait(cmdBuf);
  }

  // Creating a renderpass for the offscreen
  if(!m_offscreenRenderPass)
  {
    m_offscreenRenderPass = nvvk::createRenderPass(m_device, {m_offscreenColorFormat}, m_offscreenDepthFormat, 1, true,
                                                   true, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
  }


  // Creating the frame buffer for offscreen
  std::vector<VkImageView> attachments = {m_offscreenColor.descriptor.imageView, m_offscreenDepth.descriptor.imageView};

  vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);
  VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  info.renderPass      = m_offscreenRenderPass;
  info.attachmentCount = 2;
  info.pAttachments    = attachments.data();
  info.width           = m_size.width;
  info.height          = m_size.height;
  info.layers          = 1;
  vkCreateFramebuffer(m_device, &info, nullptr, &m_offscreenFramebuffer);
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void Renderer::createPostPipeline()
{
  // Push constants in the fragment shader
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float)};

  // Creating the pipeline layout
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = 1;
  createInfo.pSetLayouts            = &m_postDescSetLayout;
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_postPipelineLayout);


  // Pipeline: completely generic, no vertices
  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_postPipelineLayout, m_renderPass);
  pipelineGenerator.addShader(nvh::loadFile("spv/passthrough.vert.spv", true, defaultSearchPaths, true), VK_SHADER_STAGE_VERTEX_BIT);
  pipelineGenerator.addShader(nvh::loadFile("spv/post.frag.spv", true, defaultSearchPaths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
  pipelineGenerator.rasterizationState.cullMode = VK_CULL_MODE_NONE;
  m_postPipeline                                = pipelineGenerator.createPipeline();
  m_debug.setObjectName(m_postPipeline, "post");
}

//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void Renderer::createPostDescriptor()
{
  m_postDescSetLayoutBind.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  m_postDescSetLayout = m_postDescSetLayoutBind.createLayout(m_device);
  m_postDescPool      = m_postDescSetLayoutBind.createPool(m_device);
  m_postDescSet       = nvvk::allocateDescriptorSet(m_device, m_postDescPool, m_postDescSetLayout);
}


//--------------------------------------------------------------------------------------------------
// Update the output
//
void Renderer::updatePostDescriptorSet()
{
  VkWriteDescriptorSet writeDescriptorSets = m_postDescSetLayoutBind.makeWrite(m_postDescSet, 0, &m_offscreenColor.descriptor);
  vkUpdateDescriptorSets(m_device, 1, &writeDescriptorSets, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Draw a full screen quad with the attached image
//
void Renderer::drawPost(VkCommandBuffer cmdBuf)
{
  m_debug.beginLabel(cmdBuf, "Post");

  setViewport(cmdBuf);

  auto aspectRatio = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);
  vkCmdPushConstants(cmdBuf, m_postPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &aspectRatio);
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipelineLayout, 0, 1, &m_postDescSet, 0, nullptr);
  vkCmdDraw(cmdBuf, 3, 1, 0, 0);


  m_debug.endLabel(cmdBuf);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Initialize Vulkan ray tracing
// #VKRay
void Renderer::initRayTracing()
{
  // Requesting ray tracing properties
  VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  prop2.pNext = &m_rtProperties;
  vkGetPhysicalDeviceProperties2(m_physicalDevice, &prop2);

  m_rtBuilder.setup(m_device, &m_alloc, m_graphicsQueueIndex);
}

//--------------------------------------------------------------------------------------------------
// Convert an OBJ model into the ray tracing geometry used to build the BLAS
//
auto Renderer::objectToVkGeometryKHR(const ObjModel& model)
{
  // BLAS builder requires raw device addresses.
  VkDeviceAddress vertexAddress = nvvk::getBufferDeviceAddress(m_device, model.vertexBuffer.buffer);
  VkDeviceAddress indexAddress  = nvvk::getBufferDeviceAddress(m_device, model.indexBuffer.buffer);

  uint32_t maxPrimitiveCount = model.nbIndices / 3;

  // Describe buffer as array of VertexObj.
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
  triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 vertex position data.
  triangles.vertexData.deviceAddress = vertexAddress;
  triangles.vertexStride             = sizeof(VertexObj);
  // Describe index data (32-bit unsigned int)
  triangles.indexType               = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress = indexAddress;
  // Indicate identity transform by setting transformData to null device pointer.
  //triangles.transformData = {};
  triangles.maxVertex = model.nbVertices - 1;

  // Identify the above data as containing opaque triangles.
  VkAccelerationStructureGeometryKHR asGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  asGeom.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  asGeom.flags              = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
  asGeom.geometry.triangles = triangles;

  // The entire array will be used to build the BLAS.
  VkAccelerationStructureBuildRangeInfoKHR offset;
  offset.firstVertex     = 0;
  offset.primitiveCount  = maxPrimitiveCount;
  offset.primitiveOffset = 0;
  offset.transformOffset = 0;

  // Our blas is made from only one geometry, but could be made of many geometries
  nvvk::RaytracingBuilderKHR::BlasInput input;
  input.asGeometry.emplace_back(asGeom);
  input.asBuildOffsetInfo.emplace_back(offset);

  return input;
}

//--------------------------------------------------------------------------------------------------
//
//
void Renderer::createBottomLevelAS()
{
  // BLAS - Storing each primitive in a geometry
  std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;
  allBlas.reserve(m_objModel.size());
  for(const auto& obj : m_objModel)
  {
    auto blas = objectToVkGeometryKHR(obj);

    // We could add more geometry in each BLAS, but we add only one for now
    allBlas.emplace_back(blas);
  }
  m_rtBuilder.buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

//--------------------------------------------------------------------------------------------------
//
//
void Renderer::createTopLevelAS()
{
  m_tlas.reserve(m_instances.size());
  for(const Renderer::ObjInstance& inst : m_instances)
  {
    VkAccelerationStructureInstanceKHR rayInst{};
    rayInst.transform                      = nvvk::toTransformMatrixKHR(inst.transform);  // Position of the instance
    rayInst.instanceCustomIndex            = inst.objIndex;                               // gl_InstanceCustomIndexEXT
    rayInst.accelerationStructureReference = m_rtBuilder.getBlasDeviceAddress(inst.objIndex);
    rayInst.flags                          = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    rayInst.mask                           = 0xFF;       //  Only be hit if rayMask & instance.mask != 0
    rayInst.instanceShaderBindingTableRecordOffset = inst.hitgroup;  // We will use the same hit group for all objects
    m_tlas.emplace_back(rayInst);
  }
  m_rtFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  m_rtBuilder.buildTlas(m_tlas, m_rtFlags);
}

//--------------------------------------------------------------------------------------------------
// This descriptor set holds the Acceleration structure and the output image
//
void Renderer::createRtDescriptorSet()
{
  // Top-level acceleration structure, usable by both the ray generation and the closest hit (to shoot shadow rays)
  m_rtDescSetLayoutBind.addBinding(RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
                                   VK_SHADER_STAGE_COMPUTE_BIT);  // TLAS
  m_rtDescSetLayoutBind.addBinding(RtxBindings::eOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                   VK_SHADER_STAGE_COMPUTE_BIT);  // Output image

  m_rtDescPool      = m_rtDescSetLayoutBind.createPool(m_device);
  m_rtDescSetLayout = m_rtDescSetLayoutBind.createLayout(m_device);

  VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocateInfo.descriptorPool     = m_rtDescPool;
  allocateInfo.descriptorSetCount = 1;
  allocateInfo.pSetLayouts        = &m_rtDescSetLayout;
  vkAllocateDescriptorSets(m_device, &allocateInfo, &m_rtDescSet);


  VkAccelerationStructureKHR                   tlas = m_rtBuilder.getAccelerationStructure();
  VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  descASInfo.accelerationStructureCount = 1;
  descASInfo.pAccelerationStructures    = &tlas;
  VkDescriptorImageInfo imageInfo{{}, m_offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eTlas, &descASInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eOutImage, &imageInfo));
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Writes the output image to the descriptor set
// - Required when changing resolution
//
void Renderer::updateRtDescriptorSet()
{
  // (1) Output buffer
  VkDescriptorImageInfo imageInfo{{}, m_offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet  wds = m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eOutImage, &imageInfo);
  vkUpdateDescriptorSets(m_device, 1, &wds, 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void Renderer::createRtPipelineLayout() {
  // Push constant: we want to be able to update constants used by the shaders
  VkPushConstantRange pushConstant{VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(PushConstantRay)};


  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstant;

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<VkDescriptorSetLayout> rtDescSetLayouts = {m_rtDescSetLayout, m_descSetLayout};
  pipelineLayoutCreateInfo.setLayoutCount             = static_cast<uint32_t>(rtDescSetLayouts.size());
  pipelineLayoutCreateInfo.pSetLayouts                = rtDescSetLayouts.data();

  vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_rtPipelineLayout);
}

void Renderer::createRtPipeline(std::string shader, VkPipeline& pipeline)
{

  VkComputePipelineCreateInfo computePipelineCreateInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  computePipelineCreateInfo.layout       = m_rtPipelineLayout;
  computePipelineCreateInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  computePipelineCreateInfo.stage.module = nvvk::createShaderModule(m_device, nvh::loadFile(shader, true, defaultSearchPaths, true));
  computePipelineCreateInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
  computePipelineCreateInfo.stage.pName  = "main";

  vkCreateComputePipelines(m_device, {}, 1, &computePipelineCreateInfo, nullptr, &pipeline);
}


//--------------------------------------------------------------------------------------------------
// Ray Tracing the scene
//
void Renderer::raytrace(const VkCommandBuffer& cmdBuf, const nvmath::vec4f& clearColor, VkPipeline& pipeline)
{
  m_debug.beginLabel(cmdBuf, "Ray trace");

  m_pcRay.maxDepth = 64;               // How deep the path is
  m_pcRay.maxSamples = 80;             // How many samples to do per render
  m_pcRay.fireflyClampThreshold = 1.0;  // to cut fireflies
  m_pcRay.hdrMultiplier = 1;          // To brightening the scene
  m_pcRay.debugging_mode = 0;         // See DebugMode
  m_pcRay.size = {m_size.width, m_size.height};                   // rendering size
  m_pcRay.minHeatmap = 0;             // Debug mode - heat map
  m_pcRay.maxHeatmap = 3000000;

  m_pcRay.debugging_mode = eRayDir;

  std::vector<VkDescriptorSet> descSets{m_rtDescSet, m_descSet};
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipelineLayout, 0,
                          (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
  vkCmdPushConstants(cmdBuf, m_rtPipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT,
                     0, sizeof(PushConstantRay), &m_pcRay);

  vkCmdDispatch(cmdBuf, (m_size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (m_size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);

  m_debug.endLabel(cmdBuf);

  updateFrame();
}

//--------------------------------------------------------------------------------------------------
// If the camera matrix or the the fov has changed, resets the frame.
// otherwise, increments frame.
//
void Renderer::updateFrame()
{
  static nvmath::mat4f refCamMatrix;
  static float         refFov{CameraManip.getFov()};

  const auto& m   = CameraManip.getMatrix();
  const auto  fov = CameraManip.getFov();

  if(memcmp(&refCamMatrix.a00, &m.a00, sizeof(nvmath::mat4f)) != 0 || refFov != fov)
  {
    resetFrame();
    refCamMatrix = m;
    refFov       = fov;
  }
  m_pcRay.frame++;
}

void Renderer::resetFrame()
{
  m_pcRay.frame = -1;
}

void Renderer::imageToBuffer(const nvvk::Texture& imgIn, const VkBuffer& pixelBufferOut)
{
  nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
  VkCommandBuffer cmdBuff = genCmdBuf.createCommandBuffer();

  VkImageSubresourceRange subresourceRange;
  subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  subresourceRange.levelCount     = 1;
  subresourceRange.layerCount     = 1;
  subresourceRange.baseMipLevel   = 0;
  subresourceRange.baseArrayLayer = 0;
  nvvk::cmdBarrierImageLayout(cmdBuff, imgIn.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, subresourceRange);

  VkBufferImageCopy copyRegion;
  copyRegion.bufferOffset = 0;
  copyRegion.bufferRowLength = 0;
  copyRegion.bufferImageHeight = 0;
  copyRegion.imageOffset = {0, 0, 0};
  copyRegion.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copyRegion.imageExtent = {m_size.width, m_size.height, 1};
  
  vkCmdCopyImageToBuffer( cmdBuff,
                            imgIn.image,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            pixelBufferOut,
                            1,
                            &copyRegion );

  nvvk::cmdBarrierImageLayout(cmdBuff, imgIn.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, subresourceRange);
  genCmdBuf.submitAndWait(cmdBuff);
}

//--------------------------------------------------------------------------------------------------
// Save the image to disk
//
void Renderer::saveImage(const std::string& outFilename)
{
  // Create a temporary buffer to hold the pixels of the image
  VkBufferUsageFlags usage{VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
  VkDeviceSize       bufferSize = 4 * sizeof(float) * m_size.width * m_size.height;
  nvvk::Buffer pixelBuffer        = m_alloc.createBuffer(bufferSize, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

  imageToBuffer(m_offscreenColor, pixelBuffer.buffer);

  // Write the buffer to disk
  const void* data_float = m_alloc.map(pixelBuffer);
  std::vector<uint8_t> data(m_size.width * m_size.height * 4);
  for (int i = 0; i < data.size(); i++) {
    float val = ((float*)data_float)[i];
    // Emulate post shader
    val = pow(val, 1 / 2.2);

    val *= 255;
    if (val > 255) val = 255;
    if (val < 0) val = 0;
    data[i] = (uint8_t)val;
  }
  stbi_write_png(outFilename.c_str(), m_size.width, m_size.height, 4, data.data(), 0);
  m_alloc.unmap(pixelBuffer);

  // Destroy temporary buffer
  m_alloc.destroy(pixelBuffer);
}
