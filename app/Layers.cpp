#include "Layers.h"

Layer::Layer(std::string name, Renderer &renderer) : name(name), renderer(renderer) {
    //
}

void Layer::drawGui() {
    ImGui::Text("Base layer. You shouldn't see this.");
}

void Layer::init() {
    throw std::runtime_error("Calling plain Layer init");
}

void Layer::setupSequencer(VRaF::Sequencer &sequencer) {
    throw std::runtime_error("Calling plain Layer setupSequencer");
}

void Layer::update() {
    throw std::runtime_error("Calling plain Layer update");
}

Conv::Conv(std::string name, Renderer &renderer, Data &input, Data &output, std::string weights_path) 
        : Layer(name, renderer), input(input), output(output) {
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
        filterProps.src = input.getRange(filter_x, filter_x + active_filter->width - 1, filter_y, filter_y + active_filter->height - 1);
        filterProps.dst = output.getRange(filter_x, filter_x, filter_y, filter_y)[filter_idx];
        active_filter->init(filterProps, TIME_OFFSET);
        time = min_time;
        active_filter->setStage(time);
        renderer.resetFrame();
    };
    if (ImGui::SliderInt("Filter index", &filter_idx, 0, output.depth - 1)) {
        active_filter->hide_layer(-1);
        active_filter = filters[filter_idx];
        active_filter->show_layer(-1);
        filter_pos();
    }
    if (ImGui::SliderInt("Filter X", &filter_x, 0, input.width - active_filter->width)) {
        filter_pos();
    }
    if (ImGui::SliderInt("Filter Y", &filter_y, 0, input.height - active_filter->height)) {
        filter_pos();
    }
    if (ImGui::SliderFloat("Time", &time, min_time, max_time)) {
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
