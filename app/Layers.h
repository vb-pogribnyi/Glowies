#ifndef LAYERS_H
#define LAYERS_H

#include "imgui.h"
#include "vraf.h"
#include "Renderer.h"
#include "DataItem.h"

class Layer
{
public:
    std::string name;
    Renderer &renderer;
    const float max_time = 1;
    const float min_time = 0;
    float time;

    Layer(std::string name, Renderer &renderer);
    virtual void drawGui();
    virtual void init();
    virtual void setupSequencer(VRaF::Sequencer &sequencer);
    virtual void update();
};

class Conv : public Layer
{
public:
    std::vector<Filter*> filters;
    Filter *active_filter;
    int filter_x = 0, filter_y = 0;
    int filter_idx = 0;
    int stride;
    float filter_x_f, filter_y_f;
    Data& input;
    Data& output;
    FilterProps filterProps;
    const float max_time = 5;
    bool is_pos_updated = false;

    Conv(std::string name, Renderer &renderer, Data &input, Data &output, std::string weights_path, int stride=1);
    Conv(std::string name, Renderer &renderer, Data &input, Data &output, int stride=1);
    void setWeights(std::vector<unsigned long> weights_shape, std::vector<double> weights_data, std::vector<float> bias);
    virtual void drawGui() override;
    virtual void init() override;
    virtual void setupSequencer(VRaF::Sequencer &sequencer) override;
    virtual void update() override;
};

class Transition : public Layer
{
    //
};

class AvgPool : public Conv
{
public:
    AvgPool(std::string name, Renderer &renderer, Data &input, Data &output, int stride);
    virtual void init() override;
};

class Linear : public Conv
{
    //
};

#endif
