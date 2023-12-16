#include "Layers.h"

Layer::Layer(std::string name, Renderer &renderer, Data &input, Data &output) 
        : name(name), renderer(renderer), input(input), output(output) {
    // NewState and State values differ, so that an update is triggered immideately
    state.time = 1;
    newState.time = 0;
    state.pos = {1, 1, 1};
    newState.pos = {0, 0, 0};
    state.is_visible = true;
    newState.is_visible = false;
    for (int i = 0; i < input.depth; i++) {
        state.inputsVisible.push_back(false);
        newState.inputsVisible.push_back(true);
    }
}

void Layer::drawGui() {
    ImGui::Text("Base layer. You shouldn't see this.");
}

void Layer::init() {
    // throw std::runtime_error("Calling plain Layer init");
    std::cout << "Calling plain Layer init" << std::endl;
}

void Layer::setupSequencer(VRaF::Sequencer &sequencer) {
    // throw std::runtime_error("Calling plain Layer setupSequencer");
    std::cout << "Calling plain Layer setupSequencer" << std::endl;
}

bool Layer::update() {
    // throw std::runtime_error("Calling plain Layer update");
    // std::cout << "Calling plain Layer update" << std::endl;
    return false;
}

void Layer::toMax() {
    // std::cout << "Calling " << name << " toMax" << std::endl;
    newState.time = getMaxTime();
    update();
}

void Layer::toMin() {
    // std::cout << "Calling " << name << " toMin" << std::endl;
    newState.time = getMinTime();
    update();
}

float Layer::getMaxTime() {
    return max_time;
}

float Layer::getMinTime() {
    return min_time;
}

int Layer::getWidth() {
    return output.width;
}

int Layer::getHeight() {
    return output.height;
}

int Layer::getDepth() {
    return output.depth;
}

void Conv::_Conv() {
    newState.time = min_time;
    newState.pos.x = filter_x;
    newState.pos.y = filter_y;
    // for (int i = 0; i < input.depth; i++) {
    //     in_lrs_visible |= 1 << i;
    // }
    // for (int i = 0; i < output.depth; i++) {
    //     out_lrs_visible |= 1 << i;
    // }
}

Conv::Conv(std::string name, Renderer &renderer, Data &input, Data &output, int stride)
        : Layer(name, renderer, input, output), stride(stride) {
    _Conv();
}

Conv::Conv(std::string name, Renderer &renderer, Data &input, Data &output, std::string weights_path, int stride) 
        : Layer(name, renderer, input, output), stride(stride) {
    for (int layer = 0; layer < output.depth; layer++) {
        filters.push_back(new Filter(renderer, weights_path, layer));
    }
    if (filters.size() == 0) throw std::runtime_error("Filters are empty");
    active_filter = filters[filter_idx];

    _Conv();
}

void Conv::drawGui() {
    auto filter_pos = [&]() {
        newState.pos.x = filter_x;
        newState.pos.y = filter_y;
        filterProps.src = input.getRange(filter_x * stride, filter_x * stride + active_filter->width - 1, 
                filter_y * stride, filter_y * stride + active_filter->height - 1);
        filterProps.dst = output.getRange(filter_x, filter_x, filter_y, filter_y)[filter_idx];
        active_filter->init(filterProps, TIME_OFFSET);
        newState.time = min_time;
    };
    if (ImGui::SliderInt((std::string("Filter index##") + name).c_str(), &filter_idx, 0, output.depth - 1)) {
        active_filter->hide_layer(-1);
        active_filter = filters[filter_idx];
        if (state.is_visible) active_filter->show_layer(-1);
        filter_pos();
    }
    if (ImGui::SliderInt((std::string("Filter X##") + name).c_str(), &filter_x, 0, (input.width - active_filter->width) / stride)) {
        filter_pos();
    }
    if (ImGui::SliderInt((std::string("Filter Y##") + name).c_str(), &filter_y, 0, (input.height - active_filter->height) / stride)) {
        filter_pos();
    }
    if (ImGui::SliderFloat((std::string("Time##") + name).c_str(), &newState.time, min_time, max_time)) {
        //
    }
    ImGui::Checkbox((std::string("Visible##") + name).c_str(), &newState.is_visible);
    ImGui::Text("Input layers");
    for (int i = 0; i < input.depth; i++) {
        bool temp = newState.inputsVisible[i];
        if (ImGui::Checkbox((std::string("Layer ") + std::to_string(i) + "##" + name + "_" + std::to_string(i)).c_str(), &temp)) {
            if (temp) newState.inputsVisible[i] = true;
            else newState.inputsVisible[i] = false;

            std::cout << "Inputs visible: ";
            for (bool v : newState.inputsVisible) std::cout << v;
            std::cout << std::endl;
        }
    }
    ImGui::Text("Output layers");
    // for (int i = 0; i < output.depth; i++) {
    //     out_lrs_visible |= 1 << i;
    // }
}

void Conv::init() {
    int i = 0;
    for (Filter *filter : filters) {
        filterProps = {
            .prts_per_size = PRTS_PER_SIZE,
            .src = input.getRange(filter_x, filter_x + active_filter->width - 1, filter_y, filter_y + active_filter->height - 1),
            .dst = output.getRange(filter_x, filter_x, filter_y, filter_y)[i]
        };
        filter->init(filterProps, TIME_OFFSET);
        filter->hide_layer(-1);
        i++;
    }
    if (state.is_visible) filters[filter_idx]->show_layer(-1);
}

void Conv::setupSequencer(VRaF::Sequencer &sequencer) {
    sequencer.track(name + ": Time", &newState.time);

    sequencer.track(name + ": X", &newState.pos.x);
    sequencer.track(name + ": Y", &newState.pos.y);
}

bool Conv::update() {
    bool result = false;
    if(newState.pos != state.pos) {
        if ((int)newState.pos.x != filter_x || (int)newState.pos.y != filter_y) {
            filter_x = (int)newState.pos.x;
            filter_y = (int)newState.pos.y;

            std::cout << "Conv update: " << filter_x << ' ' << filter_y << std::endl;

            filterProps.src = input.getRange(filter_x, filter_x + 
                active_filter->width - 1, filter_y, filter_y + active_filter->height - 1);

            filterProps.dst = output.getRange(filter_x, filter_x, filter_y, filter_y)[0];
            active_filter->init(filterProps, TIME_OFFSET);
            renderer.resetFrame();
        }
        state.pos = newState.pos;
        result = true;
    }
    if (newState.is_visible != state.is_visible) {
        // is_visible = should_be_visible;
        if (newState.is_visible) {
            std::cout << "Showing layer " << name << std::endl;
            active_filter->show_layer(-1);
        }
        else {
            std::cout << "Hiding layer " << name << std::endl;
            active_filter->hide_layer(-1);
        }

        state.is_visible = newState.is_visible;
        result = true;
    }
    if (newState.inputsVisible != state.inputsVisible) {
        for (int i = 0; i < newState.inputsVisible.size(); i++) {
            if (newState.inputsVisible[i]) {
                input.show_layer(i);
                if (state.is_visible) active_filter->show_layer(i);
            } else {
                input.hide_layer(i);
                if (state.is_visible) active_filter->hide_layer(i);
            }
        }
        state.inputsVisible = newState.inputsVisible;
        result = true;
    }
    if (newState.time != state.time) {
        state.time = newState.time;
        active_filter->setStage(state.time);
        renderer.resetFrame();
        result = true;
    }

    return result;
}

void Conv::setWeights(std::vector<unsigned long> weights_shape, std::vector<double> weights_data, std::vector<float> bias) {
    for (int layer = 0; layer < output.depth; layer++) {
        filters.push_back(new Filter(renderer, weights_shape, weights_data, bias[layer], layer));
    }
    if (filters.size() == 0) throw std::runtime_error("Filters are empty");
    active_filter = filters[filter_idx];
}

float Conv::getMaxTime() {
    return max_time;
}

float Conv::getMinTime() {
    return min_time;
}

AvgPool::AvgPool(std::string name, Renderer &renderer, Data &input, Data &output, int stride) : Conv(name, renderer, input, output, stride) {
    if (input.depth != output.depth) {
        throw std::runtime_error("For pooling layer, input and output depths must match.");
    }
    std::vector<unsigned long> weights_shape = {(unsigned long)output.depth, (unsigned long)input.depth, (unsigned long)stride, (unsigned long)stride};
    std::vector<double> weights_data(output.depth * input.depth * stride * stride, 0);
    for (int i = 0; i < weights_data.size(); i++) {
        int layer_out = i / (input.depth * stride * stride);
        int i_in = i - layer_out * input.depth * stride * stride;
        int layer_in  = i_in / (stride * stride);
        if (layer_in == layer_out) weights_data[i] = 1.0 / stride / stride;
    }
    setWeights(weights_shape, weights_data, std::vector<float>(output.depth, 0));
}

void AvgPool::init() {
    Conv::init();
    for (int f = 0; f < filters.size(); f++) {
        for (int l = 0; l < output.depth; l++) {
            if (f != l) filters[f]->hide_layer(l);
        }
    }
}

Transition::Transition(std::string name, Renderer &renderer, Data &input, Data &output) : Layer(name, renderer, input, output) {
    if (input.items.size() != output.items.size())  throw std::runtime_error("For Tranlation, input and output size must be equal");

    out_scales.reserve(output.items.size());
    in_scales.reserve(input.items.size());
    for (int i = 0; i < input.items.size(); i++) {
        in_scales.push_back(input.items[i].props.scale);
        out_scales.push_back(output.items[i].props.scale);
    }
    newState.time = 0;
}

void Transition::drawGui() {
    if (ImGui::SliderFloat((std::string("Time##") + name).c_str(), &newState.time, min_time, max_time)) {
        //
    }
}

void Transition::init() {
    output.hide();
}

void Transition::setupSequencer(VRaF::Sequencer &sequencer) {
    sequencer.track(name + ": Time", &newState.time);
}

bool Transition::update() {
    bool result = false;
    if (newState.time != state.time) {
        state.time = newState.time;
        if (newState.time == max_time) {
            output.show();
            input.hide();
        } else {
            output.hide();
            input.show();
            float alpha = (newState.time - min_time) / max_time;
            for (int i = 0; i < input.items.size(); i++) {
                input.items[i].setScale(alpha * out_scales[i] + (1 - alpha) * in_scales[i]);
            }
        }
        renderer.resetFrame();
        result = true;
    }
    return result;
}

int Transition::getWidth() {
    return 1;
}

int Transition::getHeight() {
    return 1;
}
