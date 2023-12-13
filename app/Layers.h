#ifndef LAYERS_H
#define LAYERS_H

#include "imgui.h"
#include "vraf.h"
#include "Renderer.h"
#include "DataItem.h"

class Layer;
class Conv;
class Transition;
class AvgPool;

struct LayerState
{
    int x = 0, y = 0, z = 0;
    int step = 0;
    int nsteps;
    bool operator==(const LayerState other) const {return x == other.x && y == other.y && z == other.z;}
    bool operator!=(const LayerState other) const {return x != other.x || y != other.y || z != other.z;}
};

class LayerIterator
{
public:
    LayerIterator(Layer* target, int width, int height, int depth);
    LayerIterator(Layer* target, int x, int y, int z, int width, int height, int depth);
    bool operator==(const LayerIterator& other) const {return state == other.state && target == other.target;}
    bool operator!=(const LayerIterator& other) const {return state != other.state || target != other.target;}
    LayerIterator operator++();
    LayerIterator operator++(int);
    LayerState operator*() {return state;}
private:
    LayerState state;
    int x, y, z, width, height, depth;
    Layer* target;
};

class Layer
{
public:
    friend class LayerIterator;
    virtual LayerIterator begin();
    virtual LayerIterator end();

    std::string name;
    Renderer &renderer;
    const float max_time = 1;
    const float min_time = 0;
    int filter_x = 0, filter_y = 0;
    float time;
    Data& input;
    Data& output;

    Layer(std::string name, Renderer &renderer, Data &input, Data &output);
    virtual void drawGui();
    virtual void init();
    virtual void setupSequencer(VRaF::Sequencer &sequencer);
    virtual void update();
    virtual void toMax();
    virtual void toMin();
    virtual float getMaxTime();
    virtual float getMinTime();
};

class Conv : public Layer
{
public:
    int in_lrs_visible = 0, out_lrs_visible = 0;
    bool is_visible = true, should_be_visible = false;
    std::vector<Filter*> filters;
    Filter *active_filter;
    int filter_idx = 0;
    int stride;
    float filter_x_f, filter_y_f;
    FilterProps filterProps;
    const float max_time = 5;
    bool is_pos_updated = false;

    Conv(std::string name, Renderer &renderer, Data &input, Data &output, std::string weights_path, int stride=1);
    Conv(std::string name, Renderer &renderer, Data &input, Data &output, int stride=1);
    void _Conv();
    void setWeights(std::vector<unsigned long> weights_shape, std::vector<double> weights_data, std::vector<float> bias);
    virtual void drawGui() override;
    virtual void init() override;
    virtual void setupSequencer(VRaF::Sequencer &sequencer) override;
    virtual void update() override;
    virtual float getMaxTime();
    virtual float getMinTime();
};
                                        
class Transition : public Layer
{
public:
    virtual LayerIterator begin();
    virtual LayerIterator end();

    std::vector<float> in_scales;
    std::vector<float> out_scales;

    Transition(std::string name, Renderer &renderer, Data &input, Data &output);
    virtual void drawGui() override;                                          
    virtual void setupSequencer(VRaF::Sequencer &sequencer) override;  
    virtual void init();                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
};

class AvgPool : public Conv
{
public:
    AvgPool(std::string name, Renderer &renderer, Data &input, Data &output, int stride);
    virtual void init() override;
};

// class Linear : public Conv
// {
//     //
// };                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

#endif
