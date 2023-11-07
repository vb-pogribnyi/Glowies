#include <array>
#include <random>
#include <iostream>

#define IMGUI_DEFINE_MATH_OPERATORS
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include "imgui.h"
#include "imgui_helper.h"

#include "Renderer.h"
#include "DataItem.h"
#include "imgui/imgui_camera_widget.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvpsystem.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/context_vk.hpp"


#include "nvh/inputparser.h"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/vulkanhppsupport.hpp"

#include "stb_image_write.h"



//////////////////////////////////////////////////////////////////////////
#define UNUSED(x) (void)(x)
//////////////////////////////////////////////////////////////////////////

bool is_recording = false;

// Default search path for shaders
std::vector<std::string> defaultSearchPaths;


// GLFW Callback functions
static void onErrorCallback(int error, const char* description)
{
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Extra UI
void renderUI(Renderer& renderer)
{
  ImGuiH::CameraWidget();
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
static int const SAMPLE_WIDTH  = 1280;
static int const SAMPLE_HEIGHT = 720;

//--------------------------------------------------------------------------------------------------
// Application Entry
//
int main(int argc, char** argv)
{
  UNUSED(argc);


  // Setup GLFW window
  glfwSetErrorCallback(onErrorCallback);
  if(!glfwInit())
  {
    return 1;
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window = glfwCreateWindow(SAMPLE_WIDTH, SAMPLE_HEIGHT, PROJECT_NAME, nullptr, nullptr);


  // Setup camera
  CameraManip.setWindowSize(SAMPLE_WIDTH, SAMPLE_HEIGHT);
  CameraManip.setLookat(nvmath::vec3f(2.0f, 2.0f, 2.0f), nvmath::vec3f(0, 0, 0), nvmath::vec3f(0, 1, 0));

  // Setup Vulkan
  if(!glfwVulkanSupported())
  {
    printf("GLFW: Vulkan Not Supported\n");
    return 1;
  }

  // setup some basic things for the sample, logging file for example
  NVPSystem system(PROJECT_NAME);

  // Search path for shaders and other media
  defaultSearchPaths = {
      NVPSystem::exePath() + PROJECT_RELDIRECTORY,
      NVPSystem::exePath() + PROJECT_RELDIRECTORY "..",
      std::string(PROJECT_NAME),
  };

  // Vulkan required extensions
  assert(glfwVulkanSupported() == 1);
  uint32_t count{0};
  auto     reqExtensions = glfwGetRequiredInstanceExtensions(&count);

  // Requesting Vulkan extensions and layers
  nvvk::ContextCreateInfo contextInfo;
  contextInfo.setVersion(1, 2);                       // Using Vulkan 1.2
  for(uint32_t ext_id = 0; ext_id < count; ext_id++)  // Adding required extensions (surface, win32, linux, ..)
    contextInfo.addInstanceExtension(reqExtensions[ext_id]);
  contextInfo.addInstanceLayer("VK_LAYER_LUNARG_monitor", true);              // FPS in titlebar
  contextInfo.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, true);  // Allow debug names
  contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);            // Enabling ability to present rendering
  
  // #VKRay: Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  contextInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accelFeature);  // To build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  contextInfo.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rtPipelineFeature);  // To use vkCmdTraceRaysKHR
  contextInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline

  // Creating Vulkan base application
  nvvk::Context vkctx{};
  vkctx.initInstance(contextInfo);
  // Find all compatible devices
  auto compatibleDevices = vkctx.getCompatibleDevices(contextInfo);
  assert(!compatibleDevices.empty());
  // Use a compatible device
  vkctx.initDevice(compatibleDevices[0], contextInfo);

  // Create example
  Renderer renderer;

  // Window need to be opened to get the surface on which to draw
  const VkSurfaceKHR surface = renderer.getVkSurface(vkctx.m_instance, window);
  vkctx.setGCTQueueWithPresent(surface);

  renderer.setup(vkctx.m_instance, vkctx.m_device, vkctx.m_physicalDevice, vkctx.m_queueGCT.familyIndex);
  renderer.loadModels(RESERVE_PARTICLES);
  renderer.createSwapchain(surface, SAMPLE_WIDTH, SAMPLE_HEIGHT);
  renderer.createDepthBuffer();
  renderer.createRenderPass();
  renderer.createFrameBuffers();

  // Setup Imgui
  renderer.initGUI(0);  // Using sub-pass 0

  renderer.loadModel(nvh::findFile("media/scenes/plane.obj", defaultSearchPaths), 
                    nvmath::translation_mat4(nvmath::vec3f(0.0f, -0.02f, 0.0f)) * 
                    nvmath::scale_mat4(nvmath::vec3f(2.f, 1.f, 2.f)));
  renderer.loadModel(nvh::findFile("media/scenes/plane_light.obj", defaultSearchPaths),
                    nvmath::translation_mat4(nvmath::vec3f(0, 10.0, 0)) * 
                    nvmath::scale_mat4(nvmath::vec3f(0.18f, 0.02f, 0.02f)));

  Data data(renderer, "data.npy");
  Filter f(renderer, "weights.npy");

  auto start = std::chrono::system_clock::now();

  renderer.createOffscreenRender();
  renderer.createDescriptorSetLayout();
  // renderer.createGraphicsPipeline();
  renderer.createUniformBuffer();
  renderer.createObjDescriptionBuffer();
  renderer.updateDescriptorSet();
  renderer.initRayTracing();
  renderer.createBottomLevelAS();
  renderer.createTopLevelAS();
  renderer.createRtDescriptorSet();
  renderer.createRtPipelineLayout();
  renderer.createRtPipeline("spv/pathtrace.comp.spv", renderer.m_rtPipeline);
  renderer.createRtPipeline("spv/raytrace.comp.spv", renderer.m_rtPipeline_simpli);

  renderer.createPostDescriptor();
  renderer.createPostPipeline();
  renderer.updatePostDescriptorSet();
  nvmath::vec4f clearColor = nvmath::vec4f(1, 1, 1, 1.00f);
  bool          useRaytracer = false;


  renderer.setupGlfwCallbacks(window);
  ImGui_ImplGlfw_InitForVulkan(window, true);
  const float max_time = 4;
  const float min_time = 0;
  float time = min_time;
  int filter_x = 0, filter_y = 0;

  FilterProps filterProps = {
    .prts_per_size = 100,
    .src = data.getRange(filter_x, filter_x + f.width - 1, filter_y, filter_y + f.height - 1)
  };
  f.init(filterProps, TIME_OFFSET); // This function needs TLAS to be built
  // Main loop
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();
    if(renderer.isMinimized())
      continue;

    // Start the Dear ImGui frame
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Show UI window.
    if(renderer.showGui())
    {
      ImGuiH::Panel::Begin();
      ImGui::ColorEdit3("Clear color", reinterpret_cast<float*>(&clearColor));
      if (ImGui::Checkbox("Ray Tracer mode", &useRaytracer)) renderer.resetFrame();
      if (ImGui::SliderFloat("Time", &time, min_time, max_time)) {
        f.setStage(time);
        renderer.resetFrame();
      }
      if (ImGui::SliderInt("Filter X", &filter_x, 0, data.width - f.width) ||
            ImGui::SliderInt("Filter Y", &filter_y, 0, data.height - f.height)) {
        filterProps.src = data.getRange(filter_x, filter_x + f.width - 1, filter_y, filter_y + f.height - 1);
        f.init(filterProps, TIME_OFFSET);
        time = min_time;
        f.setStage(time);
        renderer.resetFrame();
      }
      if (ImGui::Button("Save image")) {
        renderer.saveImage("result.png");
      }
      if (!is_recording && ImGui::Button("Start recording")) is_recording = true;

      renderUI(renderer);
      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
      ImGuiH::Panel::End();
    }

    if (is_recording) {
      if (time < max_time) {
        if (renderer.m_pcRay.frame > 100) {
          time += 0.01;
          int img_id = time * 1000;
          std::string img_name = std::to_string(img_id);
          img_name.insert(img_name.begin(), 5 - img_name.size(), '0');
          renderer.saveImage("images/" + img_name + ".png");
          std::cout << img_name << std::endl;
          f.setStage(time);
          renderer.resetFrame();
        }
      } else {
        is_recording = false;
      } 
    }

    // Start rendering the scene
    // std::chrono::duration<float> diff = std::chrono::system_clock::now() - start;
    renderer.prepareFrame();

    // Start command buffer of this frame
    auto                   curFrame = renderer.getCurFrame();
    const VkCommandBuffer& cmdBuf   = renderer.getCommandBuffers()[curFrame];

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    // Updating camera buffer
    renderer.updateUniformBuffer(cmdBuf);

    // Clearing screen
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color        = {{clearColor[0], clearColor[1], clearColor[2], clearColor[3]}};
    clearValues[1].depthStencil = {1.0f, 0};

    // Offscreen render pass
    {
      // Rendering Scene
      if (useRaytracer) {
        renderer.raytrace(cmdBuf, clearColor, renderer.m_rtPipeline);
      } else {
        renderer.raytrace(cmdBuf, clearColor, renderer.m_rtPipeline_simpli);
      }
    }


    // 2nd rendering pass: tone mapper, UI
    {
      VkRenderPassBeginInfo postRenderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      postRenderPassBeginInfo.clearValueCount = 2;
      postRenderPassBeginInfo.pClearValues    = clearValues.data();
      postRenderPassBeginInfo.renderPass      = renderer.getRenderPass();
      postRenderPassBeginInfo.framebuffer     = renderer.getFramebuffers()[curFrame];
      postRenderPassBeginInfo.renderArea      = {{0, 0}, renderer.getSize()};

      // Rendering tonemapper
      vkCmdBeginRenderPass(cmdBuf, &postRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
      renderer.drawPost(cmdBuf);
      // Rendering UI
      ImGui::Render();
      ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuf);
      vkCmdEndRenderPass(cmdBuf);
    }

    // Submit for display
    vkEndCommandBuffer(cmdBuf);
    renderer.submitFrame();
  }

  // Cleanup
  vkDeviceWaitIdle(renderer.getDevice());

  renderer.destroyResources();
  renderer.destroy();
  vkctx.deinit();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
