/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

 #define RngStateType uint // Random type

//-------------------------------------------------------------------------------------------------
// This file is the main function for the path tracer.
// * `samplePixel()` is setting a ray from the camera origin through a pixel (jitter)
// * `PathTrace()` will loop until the ray depth is reached or the environment is hit.
// * `DirectLight()` is the contribution at the hit, if the shadow ray is not hitting anything.

#define ENVMAP 1
#define RR 1        // Using russian roulette
#define RR_DEPTH 0  // Minimum depth


#include "pbr_gltf.glsl"

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 DebugInfo(in State state)
{
  switch(rtxState.debugging_mode)
  {
    case eMetallic:
      return vec3(state.mat.metallic);
    case eNormal:
      return (state.normal + vec3(1)) * .5;
    case eBaseColor:
      return state.mat.albedo;
    case eEmissive:
      return state.mat.emission;
    case eAlpha:
      return vec3(state.mat.alpha);
    case eRoughness:
      return vec3(state.mat.roughness);
    case eTexcoord:
      return vec3(state.texCoord, 0);
    case eTangent:
      return vec3(state.tangent.xyz + vec3(1)) * .5;
  };
  return vec3(1000, 0, 0);
}

//-----------------------------------------------------------------------
// Use for light/env contribution
struct VisibilityContribution
{
  vec3  radiance;   // Radiance at the point if light is visible
  vec3  lightDir;   // Direction to the light, to shoot shadow ray
  float lightDist;  // Distance to the light (1e32 for infinite or sky)
  bool  visible;    // true if in front of the face and should shoot shadow ray
};

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------

struct RayState {
  vec3 absorption;
  vec3 throughput;
  vec3 radiance;
};

//-----------------------------------------------------------------------
struct Ray
{
  vec3 origin;
  vec3 direction;
  bool is_straight;
};


//-----------------------------------------------------------------------
struct BsdfSampleRec
{
  vec3  L;
  vec3  f;
  float pdf;
};



// Shading information used by the material
struct ShadeState
{
  vec3 normal;
  vec3 geom_normal;
  vec3 position;
  vec2 text_coords[1];
  vec3 tangent_u[1];
  vec3 tangent_v[1];
  vec3 color;
  uint matIndex;
  WaveFrontMaterial material;
  vec3 modelPosition;
};

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
ShadeState GetShadeState(in hitPayload hstate)
{
  ShadeState sstate;

  // Object data
  ObjDesc    objResource = objDesc.i[hstate.instanceCustomIndex];
  MatIndices matIndices  = MatIndices(objResource.materialIndexAddress);
  Materials  materials   = Materials(objResource.materialAddress);
  Indices    indices     = Indices(objResource.indexAddress);
  Vertices   vertices    = Vertices(objResource.vertexAddress);

  // Indices of the triangle
  ivec3 ind = indices.i[hstate.primitiveID];

  // Vertex of the triangle
  Vertex v0 = vertices.v[ind.x];
  Vertex v1 = vertices.v[ind.y];
  Vertex v2 = vertices.v[ind.z];

  const vec3 barycentrics = vec3(1.0 - hstate.baryCoord.x - hstate.baryCoord.y, hstate.baryCoord.x, hstate.baryCoord.y);

  // Computing the coordinates of the hit position
  const vec3 pos      = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;
  const vec3 worldPos = vec3(hstate.objectToWorld * vec4(pos, 1.0));  // Transforming the position to world space

  // Computing the normal at hit position
  const vec3 nrm      = v0.nrm * barycentrics.x + v1.nrm * barycentrics.y + v2.nrm * barycentrics.z;
  const vec3 worldNrm = normalize(vec3(nrm * hstate.objectToWorld));  // Transforming the normal to world space

  vec3       tangent  = v1.pos - v0.pos;
  tangent.xyz         = normalize(tangent.xyz);
  vec3 world_tangent  = normalize(vec3(mat4(hstate.objectToWorld) * vec4(tangent.xyz, 0)));
  world_tangent       = normalize(world_tangent - dot(world_tangent, worldNrm) * worldNrm);
  vec3 world_binormal = normalize(cross(worldNrm, world_tangent));

  WaveFrontMaterial material = materials.m[matIndices.i[hstate.primitiveID]];
  sstate.material = material;
  sstate.modelPosition = vec3(mat4(hstate.objectToWorld) * vec4(0, 0, 0, 1));

  sstate.normal         = worldNrm;
  sstate.geom_normal    = normalize(nrm);
  sstate.position       = worldPos;
  sstate.tangent_u[0]   = world_tangent;
  sstate.tangent_v[0]   = world_binormal;
  sstate.color          = material.diffuse;
  sstate.matIndex       = matIndices.i[hstate.primitiveID];

  // Move normal to same side as geometric normal
  if(dot(sstate.normal, sstate.geom_normal) <= 0)
  {
    sstate.normal *= -1.0f;
  }

  return sstate;
}

//-----------------------------------------------------------------------
// Retrieve the diffuse and specular color base on the shading model: Metal-Roughness or Specular-Glossiness
//-----------------------------------------------------------------------
void GetMetallicRoughness(inout State state)
{
  // KHR_materials_ior
  float dielectricSpecular = (state.mat.ior - 1) / (state.mat.ior + 1);
  dielectricSpecular *= dielectricSpecular;

  float perceptualRoughness = 0.0;
  float metallic            = 0.0;
  vec4  baseColor           = vec4(1, 1, 1, 1.0);
  vec3  f0                  = vec3(dielectricSpecular);

  // Metallic and Roughness material properties are packed together
  // In glTF, these factors can be specified by fixed scalar values
  // or from a metallic-roughness map
  perceptualRoughness = state.mat.roughness;
  metallic            = state.mat.metallic;

  // baseColor.rgb = mix(baseColor.rgb * (vec3(1.0) - f0), vec3(0), metallic);
  // Specular color (ior 1.4)
  f0 = mix(vec3(dielectricSpecular), baseColor.xyz, metallic);

  state.mat.albedo    = baseColor.xyz;
  state.mat.metallic  = metallic;
  state.mat.roughness = perceptualRoughness;
  state.mat.f0        = f0;
  state.mat.alpha     = baseColor.a;
  state.mat.unlit     = false;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
void GetMaterialsAndTextures(inout State state, in Ray r)
{
  state.mat.specular     = 0.5;
  state.mat.subsurface   = 0;
  state.mat.specularTint = 1;
  state.mat.sheen        = 0;
  state.mat.sheenTint    = vec3(0);

  GetMetallicRoughness(state);

  // Clamping roughness
  state.mat.roughness = max(state.mat.roughness, 0.001);

  // KHR_materials_ior
  state.eta     = dot(state.normal, state.ffnormal) > 0.0 ? (1.0 / state.mat.ior) : state.mat.ior;

  state.mat.anisotropy = 0.0;

  // KHR_materials_volume
  state.mat.attenuationColor    = vec3(0.1);
  state.mat.attenuationDistance = 1.0;
  state.mat.thinwalled          = false;

  //KHR_materials_clearcoat
  state.mat.clearcoat          = 0.001;
  state.mat.clearcoatRoughness = 0.001;
  state.mat.clearcoatRoughness = max(state.mat.clearcoatRoughness, 0.001);

  // KHR_materials_sheen
  state.mat.sheenTint = vec3(0);
  state.mat.sheen     = 0.0;
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 PathTrace(Ray r)
{
  RayState currentRay;
  currentRay.radiance   = vec3(0.0);
  currentRay.throughput = vec3(1.0);
  currentRay.absorption = vec3(0.0);
  RayState prevRay = currentRay;

  int is_shell_hit = 0;
  int is_inner_hit = 0;
  int is_prtcl_hit = 0;

  for(int depth = 0; depth < rtxState.maxDepth; depth++)
  {
    bool is_straight = true;
    uint rayFlags = gl_RayFlagsNoneEXT;
    prd.hitT      = INFINITY;
    prd.test_distance = -1;

    int nhits_shell = 0;
    int nhits_inner = 0;
    ShadeState sstate;
    int iter = 0;
    for (; iter < 55; iter++) {
      traceRayEXT(topLevelAS,   // acceleration structure
                  rayFlags,     // rayFlags
                  0xFF,         // cullMask
                  0,            // sbtRecordOffset
                  0,            // sbtRecordStride
                  0,            // missIndex
                  r.origin,     // ray origin
                  0.0,          // ray min range
                  r.direction,  // ray direction
                  INFINITY,     // ray max range
                  0             // payload (location = 0)
      );

      // Hitting the environment
      if(prd.hitT == INFINITY)
      {
        // if (is_shell_hit != 0) return vec3(1, 1, 0);
        // if (is_inner_hit != 0) return vec3(0, 0, 1);
        if(rtxState.debugging_mode != eNoDebug)
        {
          if(depth != rtxState.maxDepth - 1)
            return vec3(0);
          if(rtxState.debugging_mode == eRadiance)
            return currentRay.radiance;
          else if(rtxState.debugging_mode == eWeight)
            return currentRay.throughput;
          else if(rtxState.debugging_mode == eRayDir)
            return (r.direction + vec3(1)) * 0.5;
        }

        vec3 env = vec3(0);
        // Done sampling return

        return currentRay.radiance + (env * rtxState.hdrMultiplier * currentRay.throughput);
      }

      // Get Position, Normal, Tangents, Texture Coordinates, Color
      sstate = GetShadeState(prd);

      if (sstate.material.illum == 8) {
        if (dot(sstate.normal, r.direction) < 0) nhits_inner++;
        else nhits_inner--;

        // Run another ray to determine if the point of contact is covered
        prd.test_distance = prd.hitT;
        prd.test_position = sstate.position;
        prd.hitT = INFINITY;
        traceRayEXT(topLevelAS,   // acceleration structure
                    rayFlags,     // rayFlags
                    0xFF,         // cullMask
                    0,            // sbtRecordOffset
                    0,            // sbtRecordStride
                    0,            // missIndex
                    r.origin,     // ray origin
                    0.0,          // ray min range
                    r.direction,  // ray direction
                    INFINITY,     // ray max range
                    0             // payload (location = 0)
        );

        prd.test_distance = -1;
        if(prd.hitT == INFINITY) {
          r.origin = sstate.position + r.direction * 0.000001 / abs(dot(normalize(r.direction), sstate.normal));
          continue;
        }
      }
      if (sstate.material.illum == 4) {
        if (dot(sstate.normal, r.direction) < 0) nhits_shell++;
        else nhits_shell--;
        if (nhits_inner > 0) {
          break;
        }
        r.origin = sstate.position + r.direction * 0.000001 / abs(dot(normalize(r.direction), sstate.normal));
        continue;
      }

      if (sstate.material.illum == 5) {
        vec3 modelVector = sstate.modelPosition - r.origin;

        vec3 dist_vector = r.direction - dot(r.direction, modelVector);
        float dist = length(dist_vector);
        float shell_size = 0.5;
        float prob = 1 - dist * dist / shell_size / shell_size;

        if (r.direction != modelVector && !r.is_straight && rnd(prd.seed) > prob) {
          r.direction = modelVector;
          currentRay = prevRay;
        } else {
          r.origin = sstate.position + r.direction * 0.000001 / abs(dot(normalize(r.direction), sstate.normal));
        }
        continue;
      }
      break;
    }
/*
    if (iter > 50) return vec3(1, 0, 0);
    if (nhits_inner != 0) return vec3(0, 1, 0);
    if (nhits_shell != 0) return vec3(0, 0, 1);
*/

    BsdfSampleRec bsdfSampleRec;

    State state;
    state.position       = sstate.position;
    state.normal         = sstate.normal;
    state.tangent        = sstate.tangent_u[0];
    state.bitangent      = sstate.tangent_v[0];
    state.texCoord       = sstate.text_coords[0];
    state.matID          = sstate.matIndex;
    state.isEmitter      = false;
    state.specularBounce = false;
    state.isSubsurface   = false;
    state.ffnormal       = dot(state.normal, r.direction) <= 0.0 ? state.normal : -state.normal;

// if (depth == 1) return r.direction;

    state.mat.emission = sstate.material.emission;
    state.mat.roughness = sstate.material.transmittance.y;
    state.mat.metallic = sstate.material.transmittance.z;
    state.mat.ior = 1.4;
    state.mat.transmission = sstate.material.transmittance.x;

    // Filling material structures
    GetMaterialsAndTextures(state, r);

    // Color at vertices
    state.mat.albedo *= sstate.color;

    // Debugging info
    if(rtxState.debugging_mode != eNoDebug && rtxState.debugging_mode < eRadiance)
      return DebugInfo(state);

    // KHR_materials_unlit
    if(state.mat.unlit)
    {
      return currentRay.radiance + state.mat.albedo * currentRay.throughput;
    }

    // Reset absorption when ray is going out of surface
    if(dot(state.normal, state.ffnormal) > 0.0)
    {
      currentRay.absorption = vec3(0.0);
    }

    // Emissive material
    currentRay.radiance += state.mat.emission * currentRay.throughput;

    // Add absoption (transmission / volume)
    currentRay.throughput *= exp(-currentRay.absorption * prd.hitT);

    // Light and environment contribution
    VisibilityContribution vcontrib;
    vcontrib.radiance = vec3(0);
    vcontrib.visible  = false;
    vcontrib.radiance *= currentRay.throughput;

    // Sampling for the next ray
    bsdfSampleRec.f = PbrSample(state, -r.direction, state.ffnormal, bsdfSampleRec.L, bsdfSampleRec.pdf, prd.seed, r.is_straight);

    // Set absorption only if the ray is currently inside the object.
    if(dot(state.ffnormal, bsdfSampleRec.L) < 0.0)
    {
      currentRay.absorption = -log(state.mat.attenuationColor) / vec3(state.mat.attenuationDistance);
    }

    if(bsdfSampleRec.pdf > 0.0)
    {
      currentRay.throughput *= bsdfSampleRec.f * abs(dot(state.ffnormal, bsdfSampleRec.L)) / bsdfSampleRec.pdf;
    }
    else
    {
      break;
    }

    // Debugging info
    if(rtxState.debugging_mode != eNoDebug && (depth == rtxState.maxDepth - 1))
    {
      if(rtxState.debugging_mode == eRadiance)
        return vcontrib.radiance;
      else if(rtxState.debugging_mode == eWeight)
        return currentRay.throughput;
      else if(rtxState.debugging_mode == eRayDir)
        return (bsdfSampleRec.L + vec3(1)) * 0.5;
    }

    // For Russian-Roulette (minimizing live state)
    float rrPcont = (depth >= RR_DEPTH) ?
                        min(max(currentRay.throughput.x, max(currentRay.throughput.y, currentRay.throughput.z)) * state.eta * state.eta + 0.001, 0.95) :
                        1.0;
    if(rnd(prd.seed) >= rrPcont)
      break;                // paths with low throughput that won't contribute
    currentRay.throughput /= rrPcont;  // boost the energy of the non-terminated paths

    // Next ray
    r.direction = bsdfSampleRec.L;
    r.origin = OffsetRay(sstate.position, dot(bsdfSampleRec.L, state.ffnormal) > 0 ? state.ffnormal : -state.ffnormal);

  }

  return currentRay.radiance;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 samplePixel(ivec2 imageCoords, ivec2 sizeImage)
{
  vec3 pixelColor = vec3(0);

  // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
  vec2 subpixel_jitter = rtxState.frame == 0 ? vec2(0.5f, 0.5f) : vec2(rnd(prd.seed), rnd(prd.seed));

  // Compute sampling position between [-1 .. 1]
  const vec2 pixelCenter = vec2(imageCoords) + subpixel_jitter;
  const vec2 inUV        = pixelCenter / vec2(sizeImage.xy);
  vec2       d           = inUV * 2.0 - 1.0;

  // Compute ray origin and direction
  vec4 origin    = uni.viewInverse * vec4(0, 0, 0, 1);
  vec4 target    = uni.projInverse * vec4(d.x, d.y, 1, 1);
  vec4 direction = uni.viewInverse * vec4(normalize(target.xyz), 0);

  // Depth-of-Field
  vec3  focalPoint        = uni.focalDist * direction.xyz;
  float cam_r1            = rnd(prd.seed) * M_TWO_PI;
  float cam_r2            = rnd(prd.seed) * uni.aperture;
  vec4  cam_right         = uni.viewInverse * vec4(1, 0, 0, 0);
  vec4  cam_up            = uni.viewInverse * vec4(0, 1, 0, 0);
  vec3  randomAperturePos = (cos(cam_r1) * cam_right.xyz + sin(cam_r1) * cam_up.xyz) * sqrt(cam_r2);
  vec3  finalRayDir       = normalize(focalPoint - randomAperturePos);

  Ray ray = Ray(origin.xyz + randomAperturePos, finalRayDir, true);

  vec3 radiance = PathTrace(ray);
/*
  // Removing fireflies
  float lum = dot(radiance, vec3(0.212671f, 0.715160f, 0.072169f));
  if(lum > rtxState.fireflyClampThreshold)
  {
    radiance *= rtxState.fireflyClampThreshold / lum;
  }*/

  return radiance;

}

