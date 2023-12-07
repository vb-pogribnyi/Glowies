#include "Layers.h"

Layer::Layer(std::string name, Renderer &renderer, Data &input, Data &output) 
        : name(name), renderer(renderer), input(input), output(output) {
    //
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

void Layer::update() {
    // throw std::runtime_error("Calling plain Layer update");
    std::cout << "Calling plain Layer update" << std::endl;
}

Conv::Conv(std::string name, Renderer &renderer, Data &input, Data &output, int stride)
        : Layer(name, renderer, input, output), stride(stride) {
    time = min_time;
    filter_x_f = filter_x;
    filter_y_f = filter_y;
}

Conv::Conv(std::string name, Renderer &renderer, Data &input, Data &output, std::string weights_path, int stride) 
        : Layer(name, renderer, input, output), stride(stride) {
    for (int layer = 0; layer < output.depth; layer++) {
        filters.push_back(new Filter(renderer, weights_path, layer));
    }
    if (filters.size() == 0) throw std::runtime_error("Filters are empty");
    active_filter = filters[filter_idx];

    time = min_time;
    filter_x_f = filter_x;
    filter_y_f = filter_y;
}

void Conv::drawGui() {
    auto filter_pos = [&]() {
        filter_x_f = filter_x;
        filter_y_f = filter_y;
        filterProps.src = input.getRange(filter_x * stride, filter_x * stride + active_filter->width - 1, 
                filter_y * stride, filter_y * stride + active_filter->height - 1);
        filterProps.dst = output.getRange(filter_x, filter_x, filter_y, filter_y)[filter_idx];
        active_filter->init(filterProps, TIME_OFFSET);
        time = min_time;
        active_filter->setStage(time);
        renderer.resetFrame();
    };
    if (ImGui::SliderInt((std::string("Filter index##") + name).c_str(), &filter_idx, 0, output.depth - 1)) {
        active_filter->hide_layer(-1);
        active_filter = filters[filter_idx];
        active_filter->show_layer(-1);
        filter_pos();
    }
    if (ImGui::SliderInt((std::string("Filter X##") + name).c_str(), &filter_x, 0, (input.width - active_filter->width) / stride)) {
        filter_pos();
    }
    if (ImGui::SliderInt((std::string("Filter Y##") + name).c_str(), &filter_y, 0, (input.height - active_filter->height) / stride)) {
        filter_pos();
    }
    if (ImGui::SliderFloat((std::string("Time##") + name).c_str(), &time, min_time, max_time)) {
        active_filter->setStage(time);
        renderer.resetFrame();
    }
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
    filters[filter_idx]->show_layer(-1);
}

void Conv::setupSequencer(VRaF::Sequencer &sequencer) {
    sequencer.track(name + ": Time", &time, [&]() {
        active_filter->setStage(time);
        renderer.resetFrame();
    });

    sequencer.track(name + ": X", &filter_x_f, [&]() {is_pos_updated = true;});
    sequencer.track(name + ": Y", &filter_y_f, [&]() {is_pos_updated = true;});
}

void Conv::update() {
    if(is_pos_updated) {
        if ((int)filter_x_f != filter_x || (int)filter_y_f != filter_y) {
            filter_x = (int)filter_x_f;
            filter_y = (int)filter_y_f;

            // std::cout << filter_x << ' ' << filter_y << std::endl;

            filterProps.src = input.getRange(filter_x, filter_x + 
                active_filter->width - 1, filter_y, filter_y + active_filter->height - 1);

            filterProps.dst = output.getRange(filter_x, filter_x, filter_y, filter_y)[0];
            active_filter->init(filterProps, TIME_OFFSET);
            renderer.resetFrame();
        }
        is_pos_updated = false;
    }
}

void Conv::setWeights(std::vector<unsigned long> weights_shape, std::vector<double> weights_data, std::vector<float> bias) {
    for (int layer = 0; layer < output.depth; layer++) {
        filters.push_back(new Filter(renderer, weights_shape, weights_data, bias[layer], layer));
    }
    if (filters.size() == 0) throw std::runtime_error("Filters are empty");
    active_filter = filters[filter_idx];
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
    time = 0;
}

void Transition::drawGui() {
    if (ImGui::SliderFloat((std::string("Time##") + name).c_str(), &time, min_time, max_time)) {
        if (time == max_time) {
            output.show();
            input.hide();
        } else {
            output.hide();
            input.show();
            float alpha = (time - min_time) / max_time;
            for (int i = 0; i < input.items.size(); i++) {
                input.items[i].setScale(alpha * out_scales[i] + (1 - alpha) * in_scales[i]);
            }
        }
        renderer.resetFrame();
    }
}

void Transition::init() {
    output.hide();
}

void Transition::setupSequencer(VRaF::Sequencer &sequencer) {
    sequencer.track(name + ": Time", &time, [&]() {
        if (time == max_time) {
            output.show();
            input.hide();
        } else {
            output.hide();
            input.show();
            float alpha = (time - min_time) / max_time;
            for (int i = 0; i < input.items.size(); i++) {
                input.items[i].setScale(alpha * out_scales[i] + (1 - alpha) * in_scales[i]);
            }
        }
        renderer.resetFrame();
    });
}
