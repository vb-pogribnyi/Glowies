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

#extension GL_EXT_ray_query : enable

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
  float tglen = abs(dot(v1.pos - v0.pos, vec3(1, 0, 0)));
  if (tglen == 0) {
    tglen = abs(dot(v2.pos - v0.pos, vec3(1, 0, 0)));
  }
  tangent = tglen * vec3(1, 0, 0);
  if (tglen == 0) {
    tglen = abs(dot(v1.pos - v0.pos, vec3(0, 1, 0)));
    if (tglen == 0) {
      tglen = abs(dot(v2.pos - v0.pos, vec3(0, 1, 0)));
    }
    tangent = vec3(0, 1, 0) * tglen;
  }

  // if (tglen <= 0) tangent = vec3(0, 0, 1);

  tangent.xyz         = normalize(tangent.xyz);
  vec3 world_tangent  = normalize(vec3(mat4(hstate.objectToWorld) * vec4(tangent.xyz, 0)));
  world_tangent       = normalize(world_tangent - dot(world_tangent, worldNrm) * worldNrm);
  vec3 world_binormal = normalize(cross(worldNrm, world_tangent));

  // sstate.position = tangent;
  // return sstate;

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
  state.mat.attenuationColor    = vec3(0.4);
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

bool isInside(rayQueryEXT rayQuery, int dstIllum, vec3 testPoint) {
  rayQueryEXT rayQueryCnt;
  rayQueryInitializeEXT(rayQueryCnt,     //
                        topLevelAS,   // acceleration structure
                        gl_RayFlagsNoneEXT,     // rayFlags
                        0xFF,         // cullMask
                        rayQueryGetWorldRayOriginEXT(rayQuery),     // ray origin
                        0.0,          // ray min range
                        rayQueryGetWorldRayDirectionEXT(rayQuery),  // ray direction
                        rayQueryGetIntersectionTEXT(rayQuery, false));    // ray max range
  
  while(rayQueryProceedEXT(rayQueryCnt))
  {
    if(rayQueryGetIntersectionTypeEXT(rayQueryCnt, false) == gl_RayQueryCandidateIntersectionTriangleEXT)
    {
      ObjDesc    objResource = objDesc.i[rayQueryGetIntersectionInstanceCustomIndexEXT(rayQueryCnt, false)];
      MatIndices matIndices  = MatIndices(objResource.materialIndexAddress);
      Materials  materials   = Materials(objResource.materialAddress);
      int               matIdx = matIndices.i[rayQueryGetIntersectionPrimitiveIndexEXT(rayQueryCnt, false)];
      WaveFrontMaterial mat    = materials.m[matIdx];

      
      if(mat.illum == dstIllum) { 
        vec3 obj_position = vec3(rayQueryGetIntersectionWorldToObjectEXT(rayQueryCnt, false) * vec4(testPoint, 1));

        if (abs(obj_position.x) < 0.5 && abs(obj_position.y) < 0.5 && abs(obj_position.z) < 0.5) {
          return true;
        }
      }
    }
  }
  return false;
}

hitPayload trace(vec3 origin, vec3 direction, rayQueryEXT rayQuery, bool is_straight) {
  hitPayload result;
  result.side_radiance = vec3(0);
  rayQueryInitializeEXT(rayQuery,     //
                        topLevelAS,   // acceleration structure
                        gl_RayFlagsNoneEXT,     // rayFlags
                        0xFF,         // cullMask
                        origin,     // ray origin
                        0.0001,          // ray min range
                        direction,  // ray direction
                        INFINITY);    // ray max range
  
  // Start traversal: return false if traversal is complete
  while(rayQueryProceedEXT(rayQuery))
  {
    if(rayQueryGetIntersectionTypeEXT(rayQuery, false) == gl_RayQueryCandidateIntersectionTriangleEXT)
    {
      // Object data
      ObjDesc    objResource = objDesc.i[rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, false)];
      MatIndices matIndices  = MatIndices(objResource.materialIndexAddress);
      Materials  materials   = Materials(objResource.materialAddress);
      Indices    indices     = Indices(objResource.indexAddress);
      Vertices   vertices    = Vertices(objResource.vertexAddress);
      int               matIdx  = matIndices.i[rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false)];
      ivec3             ind     = indices.i[rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false)];
      WaveFrontMaterial mat    = materials.m[matIdx];

      // Vertex of the triangle
      Vertex v0 = vertices.v[ind.x];
      Vertex v1 = vertices.v[ind.y];
      Vertex v2 = vertices.v[ind.z];
      vec2       bary         = rayQueryGetIntersectionBarycentricsEXT(rayQuery, false);
      const vec3 barycentrics = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);

      // Computing the coordinates of the hit position
      const vec3 pos      = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;
      const vec3 worldPos = vec3(rayQueryGetIntersectionObjectToWorldEXT(rayQuery, false) * vec4(pos, 1.0));

      //////////////////////////////////////
      // Shell cross
      //////////////////////////////////////
      if(mat.illum == 4) {
        if (isInside(rayQuery, 8, worldPos)) {
          rayQueryConfirmIntersectionEXT(rayQuery);
        }
      }

      //////////////////////////////////////
      // Core cross
      //////////////////////////////////////
      else if(mat.illum == 8) {
        if (isInside(rayQuery, 4, worldPos)) {
          rayQueryConfirmIntersectionEXT(rayQuery);
        }
      }

      //////////////////////////////////////
      // Particle cross
      //////////////////////////////////////
      else if(mat.illum == 5) {
        if (is_straight) {
          continue;
        } else {
          result.side_radiance = mat.emission;
          continue;
        }
      }

      else rayQueryConfirmIntersectionEXT(rayQuery);  // The hit was opaque
    }
  }

  result.seed = 5;

  bool hit = (rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT);
  if(hit)
  {
    result.hitT = rayQueryGetIntersectionTEXT(rayQuery, true);
    result.primitiveID = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
    result.instanceID = rayQueryGetIntersectionInstanceIdEXT(rayQuery, true);
    result.instanceCustomIndex = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    result.baryCoord = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
    result.objectToWorld = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, true);
    result.worldToObject = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, true);
  } else {
    result.hitT = INFINITY;
  }
  return result;
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

  for(int depth = 0; depth < rtxState.maxDepth; depth++)
  {
    uint rayFlags = gl_RayFlagsNoneEXT;
    prd.hitT      = INFINITY;

    ShadeState sstate;
    rayQueryEXT rayQuery;
    uint seed = prd.seed;
    prd = trace(r.origin, r.direction, rayQuery, r.is_straight);
    prd.seed = seed;
    currentRay.radiance += prd.side_radiance;

    // Hitting the environment
    if(prd.hitT == INFINITY)
    {
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
    sstate = GetShadeState(prd);

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

