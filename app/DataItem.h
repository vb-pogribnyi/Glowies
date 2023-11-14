#ifndef DATA_ITEM_H
#define DATA_ITEM_H

#include <vector>
#include "Renderer.h"
#include "npy.hpp"

// Distance constraints
#define SPACING 1.2
#define MERGE_HEIGHT 3.0
#define LAYER_HEIGHT 4.0
#define MAX_POSITION 100

// Time constraints
#define CONSTRUCTION_DELAY 1.5
#define TIME_OFFSET 1.0
#define TIME_OFFSET_DI_MOVEMENT 0.1
#define ANIMATION_DURATION 1.0
#define TRANSFORM_DURATION 0.1


// Quantity constraints
#define RESERVE_PARTICLES 1024 * 1


struct DIProperties {
    bool is_has_reference;
    bool is_construction;
    vec3 position;
    float scale;
    float scale_ref;
};

struct PRTProperties {
    bool is_positive;
    bool is_splashing;
    vec3 position;
    vec3 filler_scale;
};
class DataItem {
public:
    Renderer& renderer;
    vec3 position;
    mat4 transform;

    // Indices of the models: positive, negative, (construction) positive&negative, reference (glass)
    int idx_pos;
    int idx_neg;
    int idx_pos_constr;
    int idx_neg_constr;
    int idx_ref;

    bool is_static;
    DIProperties props;

    DataItem(Renderer &renderer, DIProperties props, const ModelIndices &indices);
    void moveTo(vec3 position, bool is_hidden=false);
    std::vector<vec3> split(float n, float& w, float& h);
    void setScale(float scale, float scale_ref = 0.0f);
    void hide();
    void show();
    void showStatic();
    void hideStatic();
};

class Particle {
public:
    static float scale;
    static float shell_scale;
    vec3 position;
    ParticleIdxs idxs;
    PRTProperties props;
    Renderer &renderer;

    Particle(Renderer &renderer, PRTProperties props, const ModelIndices &indices);
    ~Particle();
    void hide();
    void moveTo(vec3 position, float filler_transition, vec3 filler_scale, float show_transition = 1.0);
};


struct FilterProps {
    float prts_per_size;        // Number of particles per size unit
    DataItem *result;
    std::vector<DataItem*> src;
    DataItem* dst;              // Static DataItem, part of Data
};

struct BCurve {
    float time_offset;
    vec3 p1;
    vec3 p2;
    vec3 p3;
    vec3 p4;
    vec3 eval(float t) const;
};

class Filter {
public:
    FilterProps props;
    Renderer& renderer;
    std::vector<Particle*> particles;
    std::vector<BCurve> curves;
    std::vector<BCurve> di_curves_start;
    std::vector<BCurve> di_curves_mid;
    std::vector<BCurve> di_curves_end;
    std::vector<float> movement_offsets;
    DataItem* dst;              // Construction DataItem, part of Filter
    float result_value;
    int width, height;
    std::vector<double> weights;
    double bias = 0.0;
    std::vector<std::pair<float, float>> weights_scales;
    std::vector<vec3> weights_positions;
    std::vector<std::pair<float, float>> weights_scales_old;
    std::vector<vec3> weights_positions_old;
    std::vector<DataItem> weights_di;
    float prt_w, prt_h;
    float time_offset = 0.0;

    Filter(Renderer& renderer, std::string weightsPath);
    ~Filter();
    void init(FilterProps props, float time_offset);
    void init_di_curves();
    void init_prt_curves();
    vec3 get_di_movement_pos(const BCurve &start, const BCurve &mid, const BCurve &end, float value);

    // The transition stage would vary between 0 and 5
    void setStage(float value);
};

class Data {
public:
    int width, height;
    std::vector<DataItem> items;
    Data(Renderer& renderer, const std::string path, vec3 offset);
    std::vector<DataItem*> getRange(int x1, int x2, int y1, int y2);
    void hide();
    void show();
};

#endif
