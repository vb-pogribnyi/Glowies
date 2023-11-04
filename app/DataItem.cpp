#include "DataItem.h"


 float Particle::scale = 0.01;
 float Particle::shell_scale = 0.03;

DataItem::DataItem(Renderer &renderer, DIProperties props, const ModelIndices &indices) : props(props), renderer(renderer) {
    uint32_t instance_id = 0;
    if (props.scale > 0) {
        if (props.is_construction) instance_id = indices.cube_pos_prt_idx;
        else instance_id = indices.cube_pos_idx;
    } else {
        if (props.is_construction) instance_id = indices.cube_neg_prt_idx;
        else instance_id = indices.cube_neg_idx;
    }
    transform = nvmath::translation_mat4(nvmath::vec3f(props.position.x, props.position.y + 0.5, props.position.z)) * 
                    nvmath::scale_mat4(nvmath::vec3f(props.scale, 1, props.scale));
    renderer.m_instances.push_back({transform, instance_id, 0});
    idx_main = renderer.m_instances.size() - 1;
    idx_ref = -1;
    if (props.is_has_reference) {
        renderer.m_instances.push_back({nvmath::translation_mat4(nvmath::vec3f(props.position.x, props.position.y + 0.4, props.position.z)) * 
                        nvmath::scale_mat4(nvmath::vec3f(1.0f, 0.8f, 1.0f)), indices.glass_idx, 0});

        idx_ref = renderer.m_instances.size() - 1;
    }
}

void DataItem::moveTo(vec3 position, Renderer &renderer) {
    transform = nvmath::translation_mat4(nvmath::vec3f(position.x, position.y + 0.5, position.z)) * 
         nvmath::scale_mat4(nvmath::vec3f(props.scale, 1, props.scale));
    renderer.m_instances[idx_main].transform = transform;
    renderer.m_tlas[idx_main].transform = nvvk::toTransformMatrixKHR(transform);
    if (props.is_has_reference) {
        renderer.m_instances[idx_ref].transform = nvmath::translation_mat4(nvmath::vec3f(position.x, position.y + 0.4, position.z)) * 
            nvmath::scale_mat4(nvmath::vec3f(props.scale_ref, 0.8f, props.scale_ref));
        renderer.m_tlas[idx_ref].transform = nvvk::toTransformMatrixKHR(
            renderer.m_instances[idx_ref].transform);
    }

    renderer.is_rebuild_tlas = true;
}

std::vector<vec3> DataItem::split(float n, float& w, float& h) {
    n = std::max(n, 1.f);
    int nrows, ncols, nlrs;                         // The last one is number of layers

    float di_volume = props.scale * props.scale;    // Height is 1. scale x scale x 1
    float di_area = di_volume;
    float point_volume = di_volume / n;
    float point_height = cbrt(point_volume);
    nlrs = floor(1.0 / point_height);
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
    moveTo(props.position, renderer); // This updates transform matrix and triggers TLAS update
}

Particle::Particle(Renderer &renderer, PRTProperties props, const ModelIndices &indices) : props(props), renderer(renderer) {
    idxs = renderer.getParticle(props.is_positive);
    moveTo(props.position, renderer, 0, vec3(0.0f), 0.0);
}

Particle::~Particle()
{
    hide();
    renderer.releaseParticle(props.is_positive, idxs);
}

void Particle::hide()
{
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

void Particle::moveTo(vec3 position, Renderer &renderer, float filler_transition, vec3 filler_scale, float show_transition) {
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

vec3 BCurve::eval(float t) {
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    return float(pow(1 - t, 3)) * p1 + 3 * float(pow(1 - t, 2) * t) * p2
            + 3 * float((1 - t) * pow(t, 2)) * p3 + float(pow(t, 3)) * p4;
}

Filter::Filter(Renderer& renderer, std::string weightsPath) : renderer(renderer) {
    npy::npy_data d = npy::read_npy<double>(weightsPath);

    int idx = 0;
    DIProperties props;
    for (double value : d.data) {
        weights.push_back(value);
        float pos_x = idx / d.shape[0];
        float pos_y = idx % d.shape[1];
        pos_x *= SPACING;
        pos_y *= SPACING;

        props = {
            .is_has_reference = true,
            .is_construction = false,
            .position = vec3(pos_x, 1.5, pos_y),
            .scale = 1.0f,
            .scale_ref = 1.0f
        };
        DataItem di_pos(renderer, props, renderer.indices);
        weights_pos.push_back(di_pos);
        props.scale = -1.0f;
        DataItem di_neg(renderer, props, renderer.indices);
        weights_neg.push_back(di_neg);
        idx++;
    }

    props = {
        .is_has_reference = false,
        .is_construction = true,
        .position = vec3(0, LAYER_HEIGHT, 0),
        .scale = (float)1.0
    };
    dst_pos = new DataItem(renderer, props, renderer.indices);
    props.scale = -1.0;
    dst_neg = new DataItem(renderer, props, renderer.indices);
}

Filter::~Filter()
{
    for (Particle* p : particles) {
        delete p;
    }
    delete dst_pos;
    delete dst_neg;
}

void Filter::init(FilterProps props, float time_offset) {
    this->props = props;
    // Clean up previously used particles
    for (Particle* p : particles) {
        delete p;
    }
    particles.clear();
    curves.clear();
    dst_pos->setScale(0);
    dst_neg->setScale(0);
    if (props.src.size() != weights.size()) {
        throw std::runtime_error("Data slice and filter sizes does not match");
    }

    std::vector<vec3> particles_pos;
    std::vector<vec3> particles_neg;
    vec3 *particles_constructing;
    int n_mrg_particles, n_constr_particles;
    float result_value = 0;
    float result_x = 0;
    float result_y = 0;
    float result_z = 0;
    for (int i = 0; i < props.src.size(); i++) {
        float applied_value = props.src[i]->props.scale * weights[i];
        result_value += applied_value;
        vec3 target_pos = props.src[i]->props.position;

        result_x += target_pos.x;
        result_y += target_pos.z;
        result_z += target_pos.y;

        float target_scale = std::abs(props.src[i]->props.scale) + 0.001;
        target_pos.y += .5;
        if (applied_value > 0) {
            weights_pos[i].setScale(std::abs(applied_value), target_scale);
            weights_pos[i].moveTo(target_pos, renderer);
            weights_neg[i].setScale(0);
            for (vec3 prt : weights_pos[i].split(props.prts_per_size * std::abs(applied_value), prt_w, prt_h)) {
                particles_pos.push_back((vec3)(weights_pos[i].transform * vec4(prt, 1)));
            }
        } else {
            weights_neg[i].setScale(std::abs(applied_value), target_scale);
            weights_neg[i].moveTo(target_pos, renderer);
            weights_pos[i].setScale(0, 0);
            for (vec3 prt : weights_neg[i].split(props.prts_per_size * std::abs(applied_value), prt_w, prt_h)) {
                particles_neg.push_back((vec3)(weights_neg[i].transform * vec4(prt, 1)));
            }
        }
    }
    particles.reserve(particles_pos.size() + particles_neg.size());
    if (particles_pos.size() > particles_neg.size()) {
        n_constr_particles = particles_pos.size() - particles_neg.size();
        n_mrg_particles = particles_neg.size();
        particles_constructing = &(particles_pos[particles_pos.size() - n_constr_particles]);
    } else {
        n_constr_particles = particles_neg.size() - particles_pos.size();
        n_mrg_particles = particles_pos.size();
        particles_constructing = &(particles_neg[particles_pos.size() - n_constr_particles]);
    }

    dst = result_value > 0 ? dst_pos : dst_neg;
    dst->setScale(result_value);
    dst->moveTo(vec3(result_x / weights.size(), result_z / weights.size() + LAYER_HEIGHT, result_y / weights.size()), renderer);

    // Constructing particles
    std::vector<vec3> prts_end = dst->split(n_constr_particles, prt_w, prt_h);
    std::vector<vec3> prts_start(prts_end.size());
    int i = 0;
    for (vec3& startpos : prts_start) {
        startpos = particles_constructing[i];
        i++;
    }

    for (int i = 0; i < prts_end.size(); i++) {
        BCurve curve;
        // Duration -0.5 : 1.5 + CONSTRUCTION_DELAY
        curve.time_offset = ((float)(rand() % 100) / 100 - 0.5) * time_offset + (float)i / prts_end.size() - CONSTRUCTION_DELAY;
        curve.p1 = prts_start[i];
        curve.p4 = vec3(dst->transform * vec4(prts_end[i], 1));
        curve.p3 = curve.p4 - vec3(0, 0.5, 0);
        curve.p2 = curve.p1 + vec3(0, 1.0, 0);
        curves.push_back(curve);
    }

    for (vec3 endpos : prts_start) {
        PRTProperties prtProps = {
            .is_positive = dst->props.scale > 0,
            .is_splashing = false,
            .position = endpos
        };
        particles.push_back(new Particle(renderer, prtProps, renderer.indices));
    }

    // Merging particles
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
    for(int i = 0; i < particles.size(); i++) {
        float curve_value = value + curves[i].time_offset;
        float stage = (curve_value - ANIMATION_DURATION) / TRANSFORM_DURATION;
        vec3 scale(prt_w * dst->props.scale, prt_h, prt_w * dst->props.scale);
        // Add scale offset. If the filler and DI overlap, weird things happen.
        scale *= 1.01f;
        //if (curve_value / ANIMATION_DURATION < 0.01) particles[i]->hide();
        float show_transition = curve_value / ANIMATION_DURATION * 100;
        particles[i]->moveTo(curves[i].eval(curve_value / ANIMATION_DURATION), renderer, stage, scale, show_transition); 
    }
}

Data::Data(Renderer& renderer, const std::string path) {
  npy::npy_data d = npy::read_npy<double>(path);

  width = d.shape[0];
  height = d.shape[1];
  int idx = 0;
  for (double value : d.data) {
    float pos_x = idx / d.shape[0];
    float pos_y = idx % d.shape[1];
    pos_x *= SPACING;
    pos_y *= SPACING;
    
    DIProperties props = {
        .is_has_reference = false,
        .is_construction = false,
        .position = vec3(pos_x, 0, pos_y),
        .scale = (float)value
    };
    DataItem di(renderer, props, renderer.indices);
    items.push_back(di);
    idx++;
  }
}

std::vector<DataItem*> Data::getRange(int x1, int x2, int y1, int y2) {
    x1 = std::max(0, x1);
    x2 = std::max(x2, x1);
    x2 = std::min(x2, width);
    y1 = std::max(0, y1);
    y2 = std::max(y2, y1);
    y2 = std::min(y2, height);

    std::vector<DataItem*> result;
    result.reserve((x2 - x1) * (y2 - y1));
    for (int i = 0; i < items.size(); i++) {
        int x = i % width;
        int y = i / width;
        if (x >= x1 && x <= x2 && y >= y1 && y <= y2) result.push_back(&items[i]);
    }

    return result;
}
