#ifndef DATA_ITEM_H
#define DATA_ITEM_H

#include <vector>
#include "Renderer.h"

struct DIProperties {
    bool is_has_reference;
    bool is_construction;
    vec3 position;
    float scale;
};

struct PRTProperties {
    bool is_positive;
    bool is_splashing;
    vec3 position;
    vec3 filler_scale;
};
class DataItem {
public:
    vec3 position;
    mat4 transform;
    int idx_main;
    int idx_ref;
    DIProperties props;

    DataItem(Renderer &renderer, DIProperties props, const ModelIndices &indices);
    void moveTo(vec3 position, Renderer &renderer);
    std::vector<vec3> split(float n, float& w, float& h);
};

class Particle {
public:
    static float scale;
    static float shell_scale;
    vec3 position;
    ParticleIdxs idxs;
    // int idx_main;
    // int idx_shell;
    // int idx_di_shell;
    // int idx_neutral;
    PRTProperties props;
    Renderer &renderer;

    Particle(Renderer &renderer, PRTProperties props, const ModelIndices &indices);
    ~Particle();
    void hide();
    void moveTo(vec3 position, Renderer &renderer, float filler_transition, vec3 filler_scale);
};


struct FilterProps {
    float prts_per_size;        // Number of particles per size unit
    DataItem *result;
    std::vector<DataItem*> src;
};

struct BCurve {
    float time_offset;
    vec3 p1;
    vec3 p2;
    vec3 p3;
    vec3 p4;
    vec3 eval(float t);
};

class Filter {
public:
    FilterProps props;
    Renderer& renderer;
    std::vector<Particle*> particles;
    std::vector<BCurve> curves;
    float prt_w, prt_h;

    Filter(Renderer &renderer, FilterProps props, const ModelIndices &indices, float time_offset);
    ~Filter();

    // The transition stage would vary between 0 and 1
    void setStage(float value);
};

#endif
