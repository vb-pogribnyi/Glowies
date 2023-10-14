#include "DataItem.h"


 float Particle::scale = 0.01;
 float Particle::shell_scale = 0.03;

DataItem::DataItem(Renderer &renderer, DIProperties props, const ModelIndices &indices) : props(props) {
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
            nvmath::scale_mat4(nvmath::vec3f(1.0f, 0.8f, 1.0f));
        renderer.m_tlas[idx_ref].transform = nvvk::toTransformMatrixKHR(
            renderer.m_instances[idx_ref].transform);
    }

    renderer.is_rebuild_tlas = true;
}

std::vector<vec3> DataItem::split(float n, float& w, float& h) {
    int nrows, ncols, nlrs;                         // The last one is number of layers

    float di_volume = props.scale * props.scale;    // Height is 1. scale x scale x 1
    float di_area = di_volume;
    float point_volume = di_volume / n;
    float point_height = cbrt(point_volume);
    nlrs = floor(1.0 / point_height);
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

Particle::Particle(Renderer &renderer, PRTProperties props, const ModelIndices &indices) : props(props), renderer(renderer) {
    idxs = renderer.getParticle(props.is_positive);
    moveTo(props.position, renderer, 0, vec3(0.0f));
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

void Particle::moveTo(vec3 position, Renderer &renderer, float filler_transition, vec3 filler_scale) {
    if (filler_transition < 0) filler_transition = 0;
    if (filler_transition > 1) filler_transition = 1;

    renderer.m_instances[idxs.particle_signed].transform = nvmath::translation_mat4(nvmath::vec3f(position)) * 
         nvmath::scale_mat4(nvmath::vec3f((1 - filler_transition) * scale));
    renderer.m_tlas[idxs.particle_signed].transform = nvvk::toTransformMatrixKHR(
        renderer.m_instances[idxs.particle_signed].transform);

    renderer.m_instances[idxs.shell].transform = nvmath::translation_mat4(nvmath::vec3f(position)) * 
         nvmath::scale_mat4(nvmath::vec3f((1 - filler_transition) * shell_scale));
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

Filter::Filter(Renderer &renderer, FilterProps props, const ModelIndices &indices, float time_offset) : 
        props(props), renderer(renderer) {
    std::vector<vec3> prts_end = props.result->split(props.prts_per_size * props.result->props.scale, prt_w, prt_h);
    std::vector<vec3> prts_start(prts_end.size());
    for (vec3& startpos : prts_start) {
        startpos.x = ((float)(rand() % 100) / 100 - 0.5) * 5;
        startpos.y = ((float)(rand() % 100) / 100 - 0.5) * 1 + 0.5;
        startpos.z = ((float)(rand() % 100) / 100 - 0.5) * 5;
    }

    for (int i = 0; i < prts_end.size(); i++) {
        BCurve curve;
        curve.time_offset = ((float)(rand() % 100) / 100 - 0.5) * time_offset;
        curve.p1 = prts_start[i];
        curve.p4 = vec3(props.result->transform * vec4(prts_end[i], 1));
        curve.p3 = curve.p4 - vec3(0, 0.5, 0);
        curve.p2 = prts_start[i] + 0.1f * (curve.p4 - prts_start[i]);
        curves.push_back(curve);
    }

    for (vec3 endpos : prts_start) {
        PRTProperties prtProps = {
            .is_positive = props.result->props.scale > 0,
            .is_splashing = false,
            .position = endpos
        };
        particles.push_back(new Particle(renderer, prtProps, indices));
    }
}

Filter::~Filter()
{
    for (Particle* p : particles) {
        delete p;
    }
}

void Filter::setStage(float value) {
    for(int i = 0; i < particles.size(); i++) {
        float curve_value = value + curves[i].time_offset;
        float stage = (curve_value - 1) * 10;
        vec3 scale(prt_w, prt_h, prt_w);
        // Add scale offset. If the filler and DI overlap, weird things happen.
        scale *= 1.01f;
        particles[i]->moveTo(curves[i].eval(curve_value), renderer, stage, scale); 
    }
}