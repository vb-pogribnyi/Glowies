#pragma once

#define MODEL_POSITIVE 1
#define MODEL_NEGATIVE 2
#define MODEL_PARTIAL 4
#define MODEL_FILLER 8
#define MODEL_GLOWING 16
#define MODEL_GLASS 32
#define MODEL_SHELL 64
#define MODEL_NEUTRAL 128

#include <list>

#include "nvvkhl/appbase_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/memallocator_dma_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "shaders/host_device.h"

// #VKRay
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"


struct ModelIndices {
  uint32_t cube_pos_idx;
  uint32_t cube_neg_idx;
  uint32_t cube_pos_prt_idx;
  uint32_t cube_neg_prt_idx;
  uint32_t filler_idx;
  uint32_t glass_idx;
  uint32_t particle_pos_idx;
  uint32_t particle_pos_shell_idx;
  uint32_t particle_neg_idx;
  uint32_t particle_neg_shell_idx;
  uint32_t particle_neutral_idx;
};

struct ParticleIdxs {
  uint32_t particle_signed;
  uint32_t shell;
  uint32_t particle_neutral;
  uint32_t filler;
};


//--------------------------------------------------------------------------------------------------
// Simple rasterizer of OBJ objects
// - Each OBJ loaded are stored in an `ObjModel` and referenced by a `ObjInstance`
// - It is possible to have many `ObjInstance` referencing the same `ObjModel`
// - Rendering is done in an offscreen framebuffer
// - The image of the framebuffer is displayed in post-process in a full-screen quad
//
class Renderer : public nvvkhl::AppBaseVk
{
public:
  ModelIndices indices;
  bool is_rebuild_tlas;
  std::list<ParticleIdxs> particles_pos_free;
  std::list<ParticleIdxs> particles_neg_free;
  ParticleIdxs getParticle(bool is_positive);
  void releaseParticle(bool is_positive, ParticleIdxs idxs);

  void setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, 
      uint32_t queueFamily) override;
  void createDescriptorSetLayout();
  void createGraphicsPipeline();
  uint32_t loadModel(const std::string& filename, nvmath::mat4f transform = nvmath::mat4f(1), uint64_t flags = 0);
  void loadModels(uint32_t nParticles);
  void updateDescriptorSet();
  void createUniformBuffer();
  void createObjDescriptionBuffer();
  void updateUniformBuffer(const VkCommandBuffer& cmdBuf);
  void onResize(int /*w*/, int /*h*/) override;
  void destroyResources();
  void rasterize(const VkCommandBuffer& cmdBuff);
  void prepareFrame();
  void saveImage(const std::string& outFilename);
  void imageToBuffer(const nvvk::Texture& imgIn, const VkBuffer& pixelBufferOut);

  // The OBJ model
  struct ObjModel
  {
    uint32_t     nbIndices{0};
    uint32_t     nbVertices{0};
    nvvk::Buffer vertexBuffer;    // Device buffer of all 'Vertex'
    nvvk::Buffer indexBuffer;     // Device buffer of the indices forming triangles
    nvvk::Buffer matColorBuffer;  // Device buffer of array of 'Wavefront material'
    nvvk::Buffer matIndexBuffer;  // Device buffer of array of 'Wavefront material'
  };

  struct ObjInstance
  {
    nvmath::mat4f transform;    // Matrix of the instance
    uint32_t      objIndex{0};  // Model index reference
    int           hitgroup{0};
  };


  // Information pushed at each draw call
  PushConstantRaster m_pcRaster{
      {1},                // Identity matrix
      {10.f, 15.f, 8.f},  // light position
      0,                  // instance Id
      100.f,              // light intensity
      0                   // light type
  };

  // Array of objects and instances in the scene
  std::vector<ObjModel>    m_objModel;   // Model on host
  std::vector<ObjDesc>     m_objDesc;    // Model description for device access
  std::vector<ObjInstance> m_instances;  // Scene model instances


  // Graphic pipeline
  VkPipelineLayout            m_pipelineLayout;
  VkPipeline                  m_graphicsPipeline;
  nvvk::DescriptorSetBindings m_descSetLayoutBind;
  VkDescriptorPool            m_descPool;
  VkDescriptorSetLayout       m_descSetLayout;
  VkDescriptorSet             m_descSet;

  nvvk::Buffer m_bGlobals;  // Device-Host of the camera matrices
  nvvk::Buffer m_bObjDesc;  // Device buffer of the OBJ descriptions

  nvvk::ResourceAllocatorDma m_alloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil            m_debug;  // Utility to name objects


  // #Post - Draw the rendered image on a quad using a tonemapper
  void createOffscreenRender();
  void createPostPipeline();
  void createPostDescriptor();
  void updatePostDescriptorSet();
  void drawPost(VkCommandBuffer cmdBuf);

  std::vector<VkAccelerationStructureInstanceKHR> m_tlas;
  nvvk::DescriptorSetBindings m_postDescSetLayoutBind;
  VkDescriptorPool            m_postDescPool{VK_NULL_HANDLE};
  VkDescriptorSetLayout       m_postDescSetLayout{VK_NULL_HANDLE};
  VkDescriptorSet             m_postDescSet{VK_NULL_HANDLE};
  VkPipeline                  m_postPipeline{VK_NULL_HANDLE};
  VkPipelineLayout            m_postPipelineLayout{VK_NULL_HANDLE};
  VkRenderPass                m_offscreenRenderPass{VK_NULL_HANDLE};
  VkFramebuffer               m_offscreenFramebuffer{VK_NULL_HANDLE};
  nvvk::Texture               m_offscreenColor;
  nvvk::Texture               m_offscreenDepth;
  VkFormat                    m_offscreenColorFormat{VK_FORMAT_R32G32B32A32_SFLOAT};
  VkFormat                    m_offscreenDepthFormat{VK_FORMAT_X8_D24_UNORM_PACK32};

  // #VKRay
  void initRayTracing();
  auto objectToVkGeometryKHR(const ObjModel& model);
  void createBottomLevelAS();
  void createTopLevelAS();
  void createRtDescriptorSet();
  void updateRtDescriptorSet();
  void createRtPipeline();
  void raytrace(const VkCommandBuffer& cmdBuf, const nvmath::vec4f& clearColor);

  void resetFrame();
  void updateFrame();

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::RaytracingBuilderKHR                      m_rtBuilder;
  nvvk::DescriptorSetBindings                     m_rtDescSetLayoutBind;
  VkDescriptorPool                                m_rtDescPool;
  VkDescriptorSetLayout                           m_rtDescSetLayout;
  VkDescriptorSet                                 m_rtDescSet;
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups;
  VkPipelineLayout                                  m_rtPipelineLayout;
  VkPipeline                                        m_rtPipeline;
  nvvk::SBTWrapper                                  m_sbtWrapper;
  VkBuildAccelerationStructureFlagsKHR m_rtFlags;

  nvvk::Buffer                    m_rtSBTBuffer;
  VkStridedDeviceAddressRegionKHR m_rgenRegion{};
  VkStridedDeviceAddressRegionKHR m_missRegion{};
  VkStridedDeviceAddressRegionKHR m_hitRegion{};
  VkStridedDeviceAddressRegionKHR m_callRegion{};

  // Push constant for ray tracer
  PushConstantRay m_pcRay{};
};
