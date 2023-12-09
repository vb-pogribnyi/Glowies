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
#include "Layers.h"
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
#include "vraf.h"



//////////////////////////////////////////////////////////////////////////
#define UNUSED(x) (void)(x)
//////////////////////////////////////////////////////////////////////////

bool is_recording = false;

// Default search path for shaders
std::vector<std::string> defaultSearchPaths;
std::vector<Layer*> layers;
std::vector<Data> datas;


// GLFW Callback functions
static void onErrorCallback(int error, const char* description)
{
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Extra UI
void renderUI(Renderer& renderer)
{
  // ImGuiH::CameraWidget();
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
  CameraManip.setMode(nvh::CameraManipulator::Modes::Walk);

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

  // Data data(renderer, "data.npy", vec3(0, 0, 0));
  // Filter* f = new Filter(renderer, "filter", 0);
  // Data data_out(renderer, "output.npy", vec3(SPACING * (float)(f->width - 1) / 2, LAYER_HEIGHT, SPACING * (float)(f->height - 1) / 2), 0);

  // std::cout << "Data input" << std::endl;
  int stride_xy = 8;
  int stride_z = 4;
  datas.push_back(Data(renderer, "data/input.npy", vec3(0, 0, 0), -1, stride_xy, stride_xy, stride_z));
  datas.push_back(Data(renderer, "data/out_conv1.npy", vec3(0, 2, 0), -1, stride_xy, stride_xy, stride_z));
  datas.push_back(Data(renderer, "data/out_pool1.npy", vec3(0, 35, 0), -1, stride_xy * 3, stride_xy * 3, stride_z));
  datas.push_back(Data(renderer, "data/out_acti1.npy", vec3(0, 35, 0), -1, stride_xy * 3, stride_xy * 3, stride_z));
  datas.push_back(Data(renderer, "data/out_conv2.npy", vec3(0, 70, 0), -1, stride_xy, stride_xy, stride_z));
  datas.push_back(Data(renderer, "data/out_pool2.npy", vec3(0, 80, 0), -1, stride_xy, stride_xy, stride_z));
  datas.push_back(Data(renderer, "data/out_acti2.npy", vec3(0, 80, 0), -1, stride_xy, stride_xy, stride_z));
  datas.push_back(Data(renderer, "data/out_dense1.npy", vec3(0, 93, 0), -1, stride_xy, stride_xy, 1));
  datas.push_back(Data(renderer, "data/out_dense1a.npy", vec3(0, 93, 0), -1, stride_xy, stride_xy, 1));
  datas.push_back(Data(renderer, "data/out_dense2.npy", vec3(0, 96, 0), -1, stride_xy, stride_xy, 1));
  datas.push_back(Data(renderer, "data/out_dense2a.npy", vec3(0, 96, 0), -1, stride_xy, stride_xy, 1));
  
  layers.push_back(new Conv("Conv 1", renderer, datas[0], datas[1], "data/filter_1"));
  layers.push_back(new AvgPool("Pool 1", renderer, datas[1], datas[2], 3));
  layers.push_back(new Transition("Activation 1", renderer, datas[2], datas[3]));
  layers.push_back(new Conv("Conv 2", renderer, datas[3], datas[4], "data/filter_2"));
  layers.push_back(new AvgPool("Pool 2", renderer, datas[4], datas[5], 2));
  layers.push_back(new Transition("Activation 2", renderer, datas[5], datas[6]));
  layers.push_back(new Conv("Dense 1", renderer, datas[6], datas[7], "data/dense_1"));
  layers.push_back(new Transition("Activation d1", renderer, datas[7], datas[8]));
  layers.push_back(new Conv("Dense 2", renderer, datas[8], datas[9], "data/dense_2"));
  layers.push_back(new Transition("Activation d2", renderer, datas[9], datas[10]));
  

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
  // const float max_time = 5;
  // const float min_time = 0;
  // float time = min_time;
  // int filter_x = 0, filter_y = 0;
  // float filter_x_f = filter_x, filter_y_f = filter_y;

  VRaF::Sequencer sequencer;

  // FilterProps filterProps = {
  //   .prts_per_size = PRTS_PER_SIZE,
  //   .src = data.getRange(filter_x, filter_x + f->width - 1, filter_y, filter_y + f->height - 1),
  //   .dst = data_out.getRange(filter_x, filter_x, filter_y, filter_y)[0]
  // };
  // data_out.hide();

  bool is_hide_output = false;
  sequencer.onFrameUpdated([&](int frame) {if (frame == 1) {is_hide_output = true;}});
  // data.show();
  for (Data &d : datas) d.show();
  for (Layer *l : layers) l->init();
  // f->init(filterProps, TIME_OFFSET); // This function needs TLAS to be built
  // f->hide_layer(4);
  // f->hide_layer(3);
  // // f->hide_layer(2);
  // // f->hide_layer(1);
  // // f->hide_layer(0);
  // data.hide_layer(4);
  // data.hide_layer(3);

  // auto updateConvLocation = [&]() {
  //   if ((int)pending_update->filter_x_f != pending_update->filter_x || (int)pending_update->filter_y_f != pending_update->filter_y) {
  //     pending_update->filter_x = (int)pending_update->filter_x_f;
  //     pending_update->filter_y = (int)pending_update->filter_y_f;

  //     // std::cout << filter_x << ' ' << filter_y << std::endl;

  //     pending_update->filterProps.src = pending_update->input.getRange(pending_update->filter_x, pending_update->filter_x + 
  //         pending_update->active_filter->width - 1, pending_update->filter_y, pending_update->filter_y + pending_update->active_filter->height - 1);

  //     pending_update->filterProps.dst = pending_update->output.getRange(pending_update->filter_x, pending_update->filter_x, pending_update->filter_y, pending_update->filter_y)[0];
  //     pending_update->active_filter->init(pending_update->filterProps, TIME_OFFSET);
  //     renderer.resetFrame();
  //   }
  // };
  auto updateCameraPos = [&]() {
    renderer.resetFrame();
  };
  for (Layer* l : layers) l->setupSequencer(sequencer);
  sequencer.track("Camera pos", &renderer.camera.pos, updateCameraPos);
  sequencer.track("Camera tgt", &renderer.camera.tgt, updateCameraPos);
  sequencer.loadFile("data/sequences.json");
  // Main loop
  float moveSpeed = 15.8;
  float lastTime = (float)glfwGetTime();

  std::function<void(bool, bool, int)> showFrame = [&](bool showGUI, bool is_raytrace, int img_id) {
      // if(pending_update) {
      //   updateConvLocation();
      //   pending_update = 0;
      // }
      for (Layer* layer : layers) layer->update();

      // Start the Dear ImGui frame
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();
      // Show UI window.
      if(showGUI)
      {
        ImGuiH::Panel::Begin();
        ImGui::ColorEdit3("Clear color", reinterpret_cast<float*>(&clearColor));
        if (ImGui::Checkbox("Ray Tracer mode", &useRaytracer)) renderer.resetFrame();
        // if (ImGui::SliderFloat("Time", &time, min_time, max_time)) {
        //   // f->setStage(time);
        //   renderer.resetFrame();
        // }


        if(ImGui::BeginTabBar("Content"))
        {
          if(ImGui::BeginTabItem("Data"))
          {
            int idx = 0;
            for (Data &data : datas) {
              if (ImGui::CollapsingHeader(std::to_string(idx++).c_str())) {
                data.drawGui(idx);
              }
            }
            ImGui::EndTabItem();
          }

          if(ImGui::BeginTabItem("Layers"))
          {
            for (Layer* layer : layers) {
              if (ImGui::CollapsingHeader(layer->name.c_str())) {
                layer->drawGui();
              }
            }
            ImGui::EndTabItem();
          }

          ImGui::EndTabBar();
        }
        // if (ImGui::SliderInt("Filter X", &filter_x, 0, data.width - f->width) ||
        //       ImGui::SliderInt("Filter Y", &filter_y, 0, data.height - f->height)) {
        //   filter_x_f = filter_x;
        //   filter_y_f = filter_y;
        //   filterProps.src = data.getRange(filter_x, filter_x + f->width - 1, filter_y, filter_y + f->height - 1);

        //   filterProps.dst = data_out.getRange(filter_x, filter_x, filter_y, filter_y)[0];
        //   f->init(filterProps, TIME_OFFSET);
        //   time = min_time;
        //   f->setStage(time);
        //   renderer.resetFrame();
        // }
        if (ImGui::Button("Save image")) {
          renderer.saveImage("result.png");
        }
        if (!is_recording && ImGui::Button("Start recording")) is_recording = true;
        if (ImGui::Button("Save sequence")) sequencer.saveFile("sequences.json");
        if (ImGui::Button("Generate sequence")) {
          // sequencer.clear();
          // int step_global = 0;
          // int nx = data.width - f->width + 1, ny = data.height - f->height + 1;
          // int nsteps = nx * ny * FRAMES_PER_CONV_STEP + 1;
          // for (int conv_x = 0; conv_x < nx; conv_x++) {
          //   for (int conv_y = 0; conv_y < ny; conv_y++) {
          //     for (int step = 0; step <= FRAMES_PER_CONV_STEP; step++) {
          //       time = (float)step / FRAMES_PER_CONV_STEP * (max_time - min_time) + min_time;
          //       // std::cout << conv_x << ' ' << conv_y << ' ' << time << std::endl;
          //       sequencer.addKeyframe("X", (float)step_global / nsteps, nsteps, conv_x);
          //       sequencer.addKeyframe("Y", (float)step_global / nsteps, nsteps, conv_y);
          //       sequencer.addKeyframe("Animation time", (float)step_global / nsteps, nsteps, time);
          //       step_global++;
          //     }
          //   }
          // }
        }

        renderUI(renderer);
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGuiH::Panel::End();
      }
      ImGui::Begin("Dock_down", 0, 16);
      sequencer.draw();
      ImGui::End();

      sequencer.update((float)glfwGetTime());

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
        if (is_raytrace) {
          renderer.raytrace(cmdBuf, clearColor, renderer.m_rtPipeline);
        } else {
          renderer.raytrace(cmdBuf, clearColor, renderer.m_rtPipeline_simpli);
        }
      }

      if (img_id >=  0) {
        std::string img_name = std::to_string(img_id);
        img_name.insert(img_name.begin(), 5 - img_name.size(), '0');
        renderer.saveImage("images/" + img_name + ".png");
        std::cout << img_name << std::endl;
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
      if (is_recording) {
        is_recording = false;
        for (int frame : sequencer) {
          for (int i = 0; i < FRAMES_TO_RENDER; i++) {
            int image_id = i == FRAMES_TO_RENDER - 1 ? frame : -1;
            showFrame(false, true, image_id);
          }
        }
      }
  };

  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();
    if(renderer.isMinimized())
      continue;

    float dtime = (float)glfwGetTime() - lastTime;
    lastTime = (float)glfwGetTime();

    float l = moveSpeed * dtime;
    renderer.camera.move(l * renderer.camera.move_fw, l * renderer.camera.move_rt, l * renderer.camera.move_up);
    if (is_hide_output) {
      // data_out.hide();
      is_hide_output = false;
    }

    showFrame(renderer.showGui(), useRaytracer, -1);
  }

  // Cleanup
  vkDeviceWaitIdle(renderer.getDevice());

  // delete f;

  renderer.destroyResources();
  renderer.destroy();
  vkctx.deinit();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
