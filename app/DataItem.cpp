#include "DataItem.h"

float Particle::scale = 0.01;
float Particle::shell_scale = 0.03;

DataItem::DataItem(Renderer &renderer, DIProperties props, const ModelIndices &indices) : props(props), renderer(renderer) {
    uint32_t instance_id = 0;
    is_static = false;
    transform = nvmath::translation_mat4(nvmath::vec3f(props.position.x, props.position.y + 0.5, props.position.z)) * 
                    nvmath::scale_mat4(nvmath::vec3f(1, 1, 1));
    renderer.m_instances.push_back({transform, indices.cube_pos_idx, 0});
    idx_pos = renderer.m_instances.size() - 1;
    renderer.m_instances.push_back({transform, indices.cube_neg_idx, 0});
    idx_neg = renderer.m_instances.size() - 1;
    idx_ref = -1;
    idx_pos_constr = -1;
    idx_neg_constr = -1;
    if (props.is_construction) {
        renderer.m_instances.push_back({transform, indices.cube_pos_prt_idx, 0});
        idx_pos_constr = renderer.m_instances.size() - 1;
        renderer.m_instances.push_back({transform, indices.cube_neg_prt_idx, 0});
        idx_neg_constr = renderer.m_instances.size() - 1;
    }
    if (props.is_has_reference) {
        renderer.m_instances.push_back({nvmath::translation_mat4(nvmath::vec3f(props.position.x, props.position.y + 0.4, props.position.z)) * 
                        nvmath::scale_mat4(nvmath::vec3f(1.0f, 0.8f, 1.0f)), indices.glass_idx, 0});

        idx_ref = renderer.m_instances.size() - 1;
    }
}

void DataItem::moveTo(vec3 position, bool is_hidden) {
    if (nvmath::length(position) > MAX_POSITION) {
        throw std::runtime_error("Position too large");
    }
    float scale_pos = 0;
    float scale_neg = 0;
    // Effective scale. To avoid too large data items
    float eff_scale = std::min(MAX_SIZE, std::abs(props.scale)) * (props.scale / std::abs(props.scale));
    if (eff_scale > 0) scale_pos = eff_scale;
    else scale_neg = -eff_scale;
    mat4 transform_pos;
    mat4 transform_neg;
    float height = getHeight();

    if (renderer.m_tlas.size() == 0) std::runtime_error("TLAS haven't been built yet");
    props.position = position;
    transform = nvmath::translation_mat4(nvmath::vec3f(position.x, position.y + 0.5, position.z)) * 
         nvmath::scale_mat4(is_hidden ? vec3(0.0f) : nvmath::vec3f(std::abs(eff_scale), height, std::abs(eff_scale)));
    transform_pos = nvmath::translation_mat4(nvmath::vec3f(position.x, position.y + 0.5, position.z)) * 
         nvmath::rotation_mat4_x(props.rotation.x) * 
         nvmath::rotation_mat4_z(props.rotation.y) * 
         nvmath::scale_mat4(is_hidden ? vec3(0.0f) : nvmath::vec3f(scale_pos, height, scale_pos));
    transform_neg = nvmath::translation_mat4(nvmath::vec3f(position.x, position.y + 0.5, position.z)) * 
         nvmath::rotation_mat4_x(props.rotation.x) * 
         nvmath::rotation_mat4_z(props.rotation.y) * 
         nvmath::scale_mat4(is_hidden ? vec3(0.0f) : nvmath::vec3f(scale_neg, height, scale_neg));
    if (props.is_construction) {
        renderer.m_instances[idx_pos_constr].transform = transform_pos;
        renderer.m_tlas[idx_pos_constr].transform = nvvk::toTransformMatrixKHR(transform_pos);
        renderer.m_instances[idx_neg_constr].transform = transform_neg;
        renderer.m_tlas[idx_neg_constr].transform = nvvk::toTransformMatrixKHR(transform_neg);


        if (!is_static) is_hidden = true;
        transform_pos = nvmath::translation_mat4(nvmath::vec3f(position.x, position.y + 0.5, position.z)) * 
            nvmath::rotation_mat4_x(props.rotation.x) * 
            nvmath::rotation_mat4_z(props.rotation.y) * 
            nvmath::scale_mat4(is_hidden ? vec3(0.0f) : nvmath::vec3f(scale_pos, height, scale_pos));
        transform_neg = nvmath::translation_mat4(nvmath::vec3f(position.x, position.y + 0.5, position.z)) * 
            nvmath::rotation_mat4_x(props.rotation.x) * 
            nvmath::rotation_mat4_z(props.rotation.y) * 
            nvmath::scale_mat4(is_hidden ? vec3(0.0f) : nvmath::vec3f(scale_neg, height, scale_neg));
    }
    renderer.m_instances[idx_pos].transform = transform_pos;
    renderer.m_tlas[idx_pos].transform = nvvk::toTransformMatrixKHR(transform_pos);
    renderer.m_instances[idx_neg].transform = transform_neg;
    renderer.m_tlas[idx_neg].transform = nvvk::toTransformMatrixKHR(transform_neg);
    if (props.is_has_reference) {
        renderer.m_instances[idx_ref].transform = nvmath::translation_mat4(nvmath::vec3f(position.x, position.y + 0.5, position.z)) * 
            nvmath::scale_mat4(is_hidden ? vec3(0.0f) : nvmath::vec3f(props.scale_ref));
        renderer.m_tlas[idx_ref].transform = nvvk::toTransformMatrixKHR(
            renderer.m_instances[idx_ref].transform);
    }

    renderer.is_rebuild_tlas = true;
}

float DataItem::getHeight() {
    float height = props.height > 0 ? props.height : std::abs(props.scale);
    if (props.scale_ref > 0) height *= props.scale_ref;

    return height;
}

std::vector<vec3> DataItem::split(float n, float& w, float& h) {
    n = std::max(n, 1.f);
    int nrows, ncols, nlrs;                         // The last one is number of layers
    float height = getHeight();

    float di_volume = props.scale * props.scale * height;    // Height is 1. scale x scale x 1
    float point_volume = di_volume / n;
    float point_height = cbrt(point_volume);
    nlrs = floor(height / point_height);
    nlrs = std::min((float)nlrs, n);
    if (nlrs == 0) nlrs = 1;
    h = 1.0 / nlrs;
    float pts_per_layer = n / nlrs;
    nrows = sqrt(pts_per_layer);
    if (nrows == 0) nrows = 1;
    ncols = nrows;
    w = 1.0 / nrows;


    std::vector<vec3> result;
    result.reserve(n);
    for (int layer = 0; layer < nlrs; layer++) {
        for (int row = 0; row < nrows; row++) {
            for (int col = 0; col < ncols; col++) {
                vec3 prt_pos = vec3(row * w, layer * h, col * w);
                prt_pos += vec3(w / 2, h / 2, w / 2);
                prt_pos -= vec3(0.5);
                result.push_back(prt_pos);
            }
        }
    }

    for (int residual = 0; residual < (n - nrows * ncols * nlrs); residual++) {
        float x = (float)(rand() % 100) / 100 - 0.5;
        float y = (float)(rand() % 100) / 100 - 0.5;
        float z = (float)(rand() % 100) / 100 - 0.5;
        result.push_back(vec3(x, y, z));
    }

    return result;
}

void DataItem::setScale(float scale, float scale_ref) {
    props.scale = scale;
    props.scale_ref = scale_ref;
    moveTo(props.position); // This updates transform matrix and triggers TLAS update
}

void DataItem::hide() {
    moveTo(props.position, true);
}

void DataItem::show() {
    setScale(props.scale, props.scale_ref);
}

void DataItem::showStatic() {
    if (!props.is_construction) return;
    is_static = true;
    moveTo(props.position);
}

void DataItem::hideStatic() {
    if (!props.is_construction) return;
    is_static = false;
    moveTo(props.position);
}

Particle::Particle(Renderer &renderer, PRTProperties props, const ModelIndices &indices) : props(props), renderer(renderer) {
    idxs = renderer.getParticle(props.is_positive);
    moveTo(props.position, 0, vec3(0.0f), 0.0);
}

Particle::~Particle()
{
    hide();
    renderer.releaseParticle(props.is_positive, idxs);
}

void Particle::hide()
{
    if (renderer.m_tlas.size() == 0) std::runtime_error("TLAS haven't been built yet");
    renderer.m_instances[idxs.particle_signed].transform = nvmath::translation_mat4(nvmath::vec3f(0.0f)) * 
         nvmath::scale_mat4(nvmath::vec3f(0.0f));
    renderer.m_tlas[idxs.particle_signed].transform = nvvk::toTransformMatrixKHR(
        renderer.m_instances[idxs.particle_signed].transform);

    renderer.m_instances[idxs.shell].transform = nvmath::translation_mat4(nvmath::vec3f(0.0f)) * 
         nvmath::scale_mat4(nvmath::vec3f(0.0f));
    renderer.m_tlas[idxs.shell].transform = nvvk::toTransformMatrixKHR(
        renderer.m_instances[idxs.shell].transform);
    renderer.m_instances[idxs.particle_neutral].transform = nvmath::translation_mat4(nvmath::vec3f(0.0f)) * 
        nvmath::scale_mat4(nvmath::vec3f(0.0f));
    renderer.m_tlas[idxs.particle_neutral].transform = nvvk::toTransformMatrixKHR(
        renderer.m_instances[idxs.particle_neutral].transform);
    renderer.m_instances[idxs.filler].transform = nvmath::translation_mat4(nvmath::vec3f(0.0f)) * 
        nvmath::scale_mat4(nvmath::vec3f(0.0f));
    renderer.m_tlas[idxs.filler].transform = nvvk::toTransformMatrixKHR(
        renderer.m_instances[idxs.filler].transform);
}

void Particle::moveTo(vec3 position, float filler_transition, vec3 filler_scale, float show_transition) {
    if (nvmath::length(position) > MAX_POSITION) {
        throw std::runtime_error("Position too large");
    }
    if (renderer.m_tlas.size() == 0) std::runtime_error("TLAS haven't been built yet");
    if (filler_transition < 0) filler_transition = 0;
    if (filler_transition > 1) filler_transition = 1;
    if (show_transition < 0) show_transition = 0;
    if (show_transition > 1) show_transition = 1;

    renderer.m_instances[idxs.particle_signed].transform = nvmath::translation_mat4(nvmath::vec3f(position)) * 
         nvmath::scale_mat4(nvmath::vec3f((1 - filler_transition) * scale * show_transition));
    renderer.m_tlas[idxs.particle_signed].transform = nvvk::toTransformMatrixKHR(
        renderer.m_instances[idxs.particle_signed].transform);

    renderer.m_instances[idxs.shell].transform = nvmath::translation_mat4(nvmath::vec3f(position)) * 
         nvmath::scale_mat4(nvmath::vec3f((1 - filler_transition) * shell_scale * show_transition));
    renderer.m_tlas[idxs.shell].transform = nvvk::toTransformMatrixKHR(
        renderer.m_instances[idxs.shell].transform);

    if (props.is_splashing) {
        float splash_scale = scale;
        if (filler_transition == 1) splash_scale = 0;
        renderer.m_instances[idxs.particle_neutral].transform = nvmath::translation_mat4(nvmath::vec3f(position)) * 
            nvmath::scale_mat4(filler_transition * vec3(splash_scale));
        renderer.m_tlas[idxs.particle_neutral].transform = nvvk::toTransformMatrixKHR(
            renderer.m_instances[idxs.particle_neutral].transform);
    } else {
        renderer.m_instances[idxs.filler].transform = nvmath::translation_mat4(nvmath::vec3f(position)) * 
            nvmath::scale_mat4(filler_transition * filler_scale);
        renderer.m_tlas[idxs.filler].transform = nvvk::toTransformMatrixKHR(
            renderer.m_instances[idxs.filler].transform);
    }

    renderer.is_rebuild_tlas = true;
}

vec3 BCurve::eval(float t) const {
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    return float(pow(1 - t, 3)) * p1 + 3 * float(pow(1 - t, 2) * t) * p2
            + 3 * float((1 - t) * pow(t, 2)) * p3 + float(pow(t, 3)) * p4;
}

DISet::DISet(Renderer &renderer, vec3 pos) {
    DIProperties props;
    props = {
        .is_has_reference = true,
        .is_construction = false,
        .position = pos,
        .height = 1.5f,
        .scale = 1.0f,
        .scale_ref = 1.0f
    };
    DataItem di(renderer, props, renderer.indices);
    components.push_back(di);

    props.rotation = {3.14159/2, 0};
    props.is_has_reference = false;
    DataItem di2(renderer, props, renderer.indices);
    components.push_back(di2);

    props.rotation = {0, 3.14159/2};
    DataItem di3(renderer, props, renderer.indices);
    components.push_back(di3);

    transform = components[0].transform;
}

void DISet::moveTo(vec3 position, bool is_hidden) {
    for (DataItem &c : components) {
        c.moveTo(position, is_hidden || this->is_hidden);
    }
    transform = components[0].transform;
}

std::vector<vec3> DISet::split(float n, float& w, float& h) {
    return components[0].split(n, w, h);
}

void DISet::setScale(float scale, float scale_ref) {
    if (this->is_hidden) return;
    for (DataItem &c : components) {
        if (c.props.scale != scale || c.props.scale_ref != scale_ref) c.setScale(scale, scale_ref);
    }
    transform = components[0].transform;
}

void DISet::hide(bool isPerm) {
    is_hidden = true;
    if (isPerm) is_hidden_perm = true;
    for (DataItem &c : components) {
        c.hide();
    }
}

void DISet::show(bool isPerm) {
    is_hidden = false;
    if (isPerm) is_hidden_perm = false;
    for (DataItem &c : components) {
        c.show();
    }
}

void DISet::showStatic() {
    for (DataItem &c : components) {
        c.showStatic();
    }
}

void DISet::hideStatic() {
    for (DataItem &c : components) {
        c.hideStatic();
    }
}

void Filter::_Filter(Renderer& renderer, std::vector<unsigned long> weights_shape, std::vector<double> weights_data, float bias, int outLayer) {
    this->bias = bias;
    width = weights_shape[2];
    height = weights_shape[3];
    props.dst = 0;

    int idx = 0;
    int itemsPerOutLayer = width * height * weights_shape[1];
    int itemsPerInLayer = width * height;
    DIProperties props;
    for (double value : weights_data) {
        int layer = idx / itemsPerOutLayer;
        // std::cout << idx << '\t' << layer << '\t' << value << std::endl;
        if (layer % weights_shape[0] == outLayer) {
            int layer_idx = idx % itemsPerOutLayer;
            weights.push_back(value);
            weights_scales.push_back({value, 1});
            weights_positions.push_back(vec3(-1));
            weights_scales_old.push_back({value, 1});
            weights_positions_old.push_back(vec3(-1));
            float pos_x = layer_idx / weights_shape[3];
            float pos_y = layer_idx % weights_shape[3];
            pos_x *= SPACING;
            pos_y *= SPACING;

            DISet diset(renderer, vec3(pos_x, 1.5, pos_y));
            diset.layer = layer_idx / itemsPerInLayer;
            weights_di.push_back(diset);
        }
        idx++;
    }

    props = {
        .is_has_reference = false,
        .is_construction = true,
        .position = vec3(0, LAYER_HEIGHT, 0),
        .height = 0,
        .scale = (float)1.0
    };
    dst = new DataItem(renderer, props, renderer.indices);
}

Filter::Filter(Renderer& renderer, std::vector<unsigned long> weights_shape, std::vector<double> weights_data, float bias, int outLayer) 
        : renderer(renderer), bias(bias) {
    _Filter(renderer, weights_shape, weights_data, bias, outLayer);
}

Filter::Filter(Renderer& renderer, std::string weightsPath, int outLayer) : renderer(renderer) {
    npy::npy_data bias_np = npy::read_npy<double>(weightsPath + "_bias.npy");

    npy::npy_data weights_np = npy::read_npy<double>(weightsPath + "_weights.npy");
    std::vector<unsigned long> weights_shape = {weights_np.shape[0], weights_np.shape[1], weights_np.shape[2], weights_np.shape[3]};
    _Filter(renderer, weights_shape, weights_np.data, bias_np.data[outLayer], outLayer);
}

Filter::~Filter()
{
    if (props.dst) props.dst->show();
    for (Particle* p : particles) {
        delete p;
    }
    delete dst;
}

void Filter::init(FilterProps props, float time_offset) {
    this->time_offset = time_offset;
    if (this->props.dst) this->props.dst->show();
    this->props = props;
    if (this->props.dst) this->props.dst->hide();
    // Clean up previously used particles
    for (Particle* p : particles) {
        delete p;
    }
    particles.clear();
    curves.clear();
    di_curves_start.clear();
    di_curves_mid.clear();
    di_curves_end.clear();
    movement_offsets.clear();
    dst->hide();
    if (props.src.size() != weights.size()) {
        throw std::runtime_error("Data slice and filter sizes does not match");
    }

    result_value = 0;
    float result_x = 0;
    float result_y = 0;
    float result_z = 0;
    // std::cout << std::endl << "Calculating window:" << std::endl;
    for (int i = 0; i < props.src.size(); i++) {
        float applied_value = props.src[i]->props.scale * weights[i];
        // std::cout << props.src[i]->props.scale << ' ' << weights[i] << std::endl;
        result_value += applied_value;
        vec3 target_pos = props.src[i]->props.position;

        result_x += target_pos.x;
        result_y += target_pos.z;
        result_z += target_pos.y;
        movement_offsets.push_back(((float)(rand() % 100) / 100 - 0.5) * TIME_OFFSET_DI_MOVEMENT);

        float target_scale = props.src[i]->props.scale;
        // target_pos.y += 0.5;
        // std::cout << i << '\t' << props.src[i]->props.scale << '\t' << weights[i] << '\t' << applied_value << '\t' << result_value << std::endl;

        weights_scales_old[i] = weights_scales[i];
        weights_positions_old[i] = weights_positions[i].y >= 0 ? weights_positions[i] : target_pos;
        weights_scales[i] = {applied_value, target_scale};
        weights_positions[i] = target_pos;

        weights_di[i].moveTo(weights_positions_old[i]);
        weights_di[i].setScale(weights_scales_old[i].first, std::abs(weights_scales_old[i].second) + 0.001);

    }
    if (std::abs(result_value + bias - props.dst->props.scale) > 1e-6) {
        throw std::runtime_error("Generated and given result won't match");
    }

    dst->setScale(result_value);
    // dst->moveTo(vec3(result_x / weights.size(), result_z / weights.size() + LAYER_HEIGHT, result_y / weights.size()));
    dst->moveTo(props.dst->props.position);
}

void Filter::init_prt_curves() {
    std::vector<vec3> particles_pos;
    std::vector<vec3> particles_neg;
    vec3 *particles_constructing;
    int n_mrg_particles, n_constr_particles;
    int i = 0;

    // Split the filter's weights into particles
    for (DISet &weight_di : weights_di) {
        for (vec3 prt : weight_di.split(props.prts_per_size * std::abs(weights_scales[i].first), prt_w, prt_h)) {
            if (weights_scales[i].first > 0) {
                particles_pos.push_back((vec3)(weight_di.transform * vec4(prt, 1)));
            } else {
                particles_neg.push_back((vec3)(weight_di.transform * vec4(prt, 1)));
            }
        }
        i++;
    }
    
    // Separate the particles into constructing and merging
    particles.reserve(particles_pos.size() + particles_neg.size());
    if (particles_pos.size() > particles_neg.size()) {
        n_constr_particles = particles_pos.size() - particles_neg.size();
        n_mrg_particles = particles_neg.size();
        particles_constructing = &(particles_pos[particles_pos.size() - n_constr_particles]);
    } else {
        n_constr_particles = particles_neg.size() - particles_pos.size();
        n_mrg_particles = particles_pos.size();
        particles_constructing = &(particles_neg[particles_neg.size() - n_constr_particles]);
    }

    // Set up movement of the constructing particles
    std::vector<vec3> prts_end = dst->split(n_constr_particles, prt_w, prt_h);

    for (int i = 0; i < prts_end.size(); i++) {
        if (nvmath::length(particles_constructing[i]) > MAX_POSITION) {
            throw std::runtime_error("Position too large");
        }

        BCurve curve;
        // Duration -0.5 : 1.5 + CONSTRUCTION_DELAY
        curve.time_offset = ((float)(rand() % 100) / 100 - 0.5) * time_offset + (float)i / prts_end.size() - CONSTRUCTION_DELAY;
        curve.p1 = particles_constructing[i];
        curve.p4 = vec3(dst->transform * vec4(prts_end[i], 1));
        curve.p3 = curve.p4 - vec3(0, 0.5, 0);
        curve.p2 = curve.p1 + vec3(0, 1.0, 0);
        curves.push_back(curve);

        PRTProperties prtProps = {
            .is_positive = dst->props.scale > 0,
            .is_splashing = false,
            .position = particles_constructing[i]
        };
        particles.push_back(new Particle(renderer, prtProps, renderer.indices));
    }

    // Set up movement of the merging particles
    for (int i = 0; i < n_mrg_particles; i++) {
        vec3 start_pt1 = particles_pos[i];
        vec3 start_pt2 = particles_neg[i];
        vec3 dist_vector = start_pt2 - start_pt1;
        vec3 mrg_pt = start_pt1 + 0.5f * dist_vector;
        mrg_pt.y = MERGE_HEIGHT;

        BCurve curve;
        curve.time_offset = ((float)(rand() % 100) / 100 - 0.5) * time_offset;
        curve.p1 = start_pt1;
        curve.p4 = mrg_pt;
        curve.p3 = curve.p4 - 0.5f * dist_vector;
        curve.p2 = curve.p1 + vec3(0, 1.0, 0);
        curves.push_back(curve);

        curve.p1 = start_pt2;
        curve.p3 = curve.p4 + 0.5f * dist_vector;
        curve.p2 = curve.p1 + vec3(0, 1.0, 0);
        curves.push_back(curve);

        PRTProperties prtProps = {
            .is_positive = true,
            .is_splashing = true,
            .position = start_pt1
        };
        particles.push_back(new Particle(renderer, prtProps, renderer.indices));

        prtProps.is_positive = false;
        particles.push_back(new Particle(renderer, prtProps, renderer.indices));
    }
}

void Filter::setStage(float value) {
    float move_time = 1.0;
    float unscale_time = 1.0;
    float scale_time = 1.0;
    float bias_time = 1.0;
    float merge_time = ANIMATION_DURATION + TRANSFORM_DURATION + TIME_OFFSET / 2 + CONSTRUCTION_DELAY + TIME_OFFSET / 2;
    float value_unscale     = 1;
    float value_move        = 2;
    float value_scale       = 3;
    float value_merge       = 4;
    float value_bias        = 5;
    float max_value = value_scale;

    // Reset particles, so they don't hang around in a stage they souldn't be involved
    for(int i = 0; i < particles.size(); i++) {
        particles[i]->moveTo(curves[i].eval(0), 0.0, vec3(0.0f), 0.0);
    }

    // DI scale and movement start with random offset, so kept together
    if (di_curves_start.size() == 0) init_di_curves();
    for (int i = 0; i < weights.size(); i++) {
        float value_inner = (value - TIME_OFFSET_DI_MOVEMENT / 2) + di_curves_start[i].time_offset;
        value_inner = value_inner / max_value * (max_value + TIME_OFFSET_DI_MOVEMENT);
        value_inner = std::max(value_inner, 0.0f);
        value_inner = std::min(value_inner, value_scale);
        if (value_inner >= 0 && value_inner <= value_unscale) {
            // Unscale stage
            value_inner = (value_inner - 0) * unscale_time;
            std::pair<float, float> scale = weights_scales_old[i];
            float weighted_scale = (1 - value_inner) * scale.first + value_inner * weights[i];
            float weighted_target_scale = (1 - value_inner) * scale.second + value_inner * 1.0;
            weights_di[i].setScale(weighted_scale, std::abs(weighted_target_scale) + 0.001);
        } else if (value_inner > value_unscale && value_inner <= value_move) {
            // Move stage
            value_inner = (value_inner - value_unscale) * move_time;
            vec3 position = get_di_movement_pos(di_curves_start[i], di_curves_mid[i], di_curves_end[i], value_inner / move_time);
            weights_di[i].moveTo(position);
        } else if (value_inner > value_move && value_inner <= value_scale) {
            // Scale stage
            value_inner = (value_inner - value_move) * scale_time;
            weights_di[i].moveTo(weights_positions[i]);
            std::pair<float, float> scale = weights_scales[i];
            std::pair<float, float> scale_old = {weights[i], 1.0};
            float weighted_scale = value_inner * scale.first + (1 - value_inner) * scale_old.first;
            float weighted_target_scale = value_inner * scale.second + (1 - value_inner) * scale_old.second;
            weights_di[i].setScale(weighted_scale, std::abs(weighted_target_scale) + 0.001);
        }
    }

    // Particles movement
    if (value > value_scale && value <= value_merge) {
        if (curves.size() == 0) init_prt_curves();

        float value_inner = (value - value_scale) * merge_time - TIME_OFFSET / 2; // This value should start at negative
        for(int i = 0; i < particles.size(); i++) {
            float curve_value = value_inner + curves[i].time_offset;
            float stage = (curve_value - ANIMATION_DURATION) / TRANSFORM_DURATION;
            // Only width should be scaled. DI height always remains 1.0
            vec3 scale(prt_w * dst->props.scale, prt_h * dst->props.scale, prt_w * dst->props.scale);
            // Add scale offset. If the filler and DI overlap, weird things happen.
            scale *= 1.01f;
            float show_transition = curve_value / ANIMATION_DURATION * 100;
            particles[i]->moveTo(curves[i].eval(curve_value / ANIMATION_DURATION), stage, scale, show_transition); 
        }
    } else {
        curves.clear();
        for (Particle *p : particles) delete p;
        particles.clear();
    }

    // Showing static part when the construction is complete; showing bias
    if (value > value_merge && value <= value_bias) {
        float value_inner = (value - value_merge) * bias_time;
        float weighted_scale = value_inner * (result_value + bias) + (1 - value_inner) * result_value;
        dst->setScale(weighted_scale);
        dst->showStatic();
    } else {
        dst->hideStatic();
    }

    if (value >= value_bias) {
        dst->hide();
        props.dst->show();
    } else {
        dst->show();
        props.dst->hide();
    }
}

void Filter::init_di_curves() {
    for (int i = 0; i < weights.size(); i++) {
        vec3 dist_vector = nvmath::normalize(weights_positions[i] - weights_positions_old[i]);
        vec3 vertical_offset = vec3(0, 1, 0);
        float lead_length = 0.2;

        BCurve curve;
        curve.time_offset = ((float)(rand() % 100) / 100 - 0.5) * TIME_OFFSET_DI_MOVEMENT;
        curve.p1 = weights_positions_old[i];
        curve.p4 = curve.p1 + vertical_offset;
        curve.p3 = curve.p4 - vertical_offset * lead_length;
        curve.p2 = curve.p1 + vertical_offset * lead_length;
        di_curves_start.push_back(curve);

        curve.p1 = curve.p4;
        curve.p4 = weights_positions[i] + vertical_offset;
        curve.p3 = curve.p4 - dist_vector * lead_length;
        curve.p2 = curve.p1 + dist_vector * lead_length;
        di_curves_mid.push_back(curve);

        curve.p1 = curve.p4;
        curve.p4 = weights_positions[i];
        curve.p3 = curve.p4 + vertical_offset * lead_length;
        curve.p2 = curve.p1 - vertical_offset * lead_length;
        di_curves_end.push_back(curve);
    }
}

vec3 Filter::get_di_movement_pos(const BCurve &start, const BCurve &mid, const BCurve &end, float value) {
    float time_start = 0.2;
    float time_end = 0.2;
    float time_mid = 1 - time_start - time_end;

    if (value < time_start) {
        value = (value - 0) / time_start;
        return start.eval(value);
    } else if (value < (time_start + time_mid)) {
        value = (value - time_start) / time_mid;
        return mid.eval(value);
    } else {
        value = (value - time_start - time_mid) / time_end;
        return end.eval(value);
    }
}

void Filter::hide_layer(int layer) {
    std::cout << "Hiding layer " << layer << std::endl;
    for (auto &w : weights_di) {
        if (layer < 0) w.hide();
        else if (w.layer == layer) w.hide(true);
    }
}

void Filter::show_layer(int layer) {
    std::cout << "Showing layer " << layer << std::endl;
    for (auto &w : weights_di) {
        if (layer < 0 && !w.is_hidden_perm) w.show();
        else if (w.layer == layer) w.show(true);
    }
}

Data::Data(Renderer& renderer, const std::string path, vec3 offset, int layer, float spacing_x, float spacing_y, float spacing_z) {
  npy::npy_data d = npy::read_npy<double>(path);

  depth = layer > 0 ? 1 : d.shape[0];
  width = d.shape[1];
  height = d.shape[2];
  int valsPerLayer = width * height;
  int idx = 0;
  for (double value : d.data) {
    int dataLayer = idx / valsPerLayer;
    if (layer < 0 || dataLayer == layer) {
        float pos_x = (idx - dataLayer * valsPerLayer) / d.shape[2];
        float pos_y = (idx - dataLayer * valsPerLayer) % d.shape[2];
        float pos_z = dataLayer * SPACING * spacing_z;
        // std::cout << pos_x << '\t' << pos_y << '\t';
        pos_x -= width / 2 - 0.5;
        pos_y -= height / 2 - 0.5;
        // std::cout << pos_x << '\t' << pos_y << '\t';
        pos_x *= SPACING * spacing_x;
        pos_y *= SPACING * spacing_y;
        // std::cout << pos_x << '\t' << pos_y << std::endl;

        DIProperties props = {
            .is_has_reference = false,
            .is_construction = false,
            .position = vec3(pos_x, pos_z, pos_y) + offset,
            .height = 0,
            .scale = (float)value
        };
        DataItem di(renderer, props, renderer.indices);
        di.layer = dataLayer;
        items.push_back(di);
    }
    idx++;
  }
  for (int i = 0; i < depth; i++) layersVisibility.push_back(1);
}

std::vector<DataItem*> Data::getRange(int x1, int x2, int y1, int y2) {
    x1 = std::max(0, x1);
    x2 = std::max(x2, x1);
    x2 = std::min(x2, width);
    y1 = std::max(0, y1);
    y2 = std::max(y2, y1);
    y2 = std::min(y2, height);

    std::vector<DataItem*> result;
    result.reserve((x2 - x1) * (y2 - y1) * depth);
    for (int i = 0; i < items.size(); i++) {
        int layer_i = i % (width * height);
        int y = layer_i % height;
        int x = layer_i / height;
        if (x >= x1 && x <= x2 && y >= y1 && y <= y2) result.push_back(&items[i]);
    }

    return result;
}

void Data::hide() {
    for (DataItem &i : items) {
        i.hide();
    }
}

void Data::show() {
    for (DataItem &i : items) {
        i.show();
    }
}

void Data::hide_layer(int layer) {
    for (DataItem &i : items) {
        if (i.layer == layer) i.hide();
    }
}

void Data::show_layer(int layer) {
    for (DataItem &i : items) {
        if (i.layer == layer) i.show();
    }
}
