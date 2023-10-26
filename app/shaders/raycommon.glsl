#define INFINITY 1e32

struct hitPayload
{
  uint   seed;
  float  hitT;
  int    primitiveID;
  int    instanceID;
  int    instanceCustomIndex;
  vec2   baryCoord;
  mat4x3 objectToWorld;
  mat4x3 worldToObject;

  vec3 side_radiance;
};

struct hitPayloadSimpli {
  vec3 hitValue;
  bool isShadowed;
  bool isSkipAll;
  int depth;
  int n_core_crosses;
  int n_shell_crosses;
};

struct shadowPayload
{
  bool isHit;
  uint seed;
};
// utility for temperature
float fade(float low, float high, float value)
{
  float mid   = (low + high) * 0.5;
  float range = (high - low) * 0.5;
  float x     = 1.0 - clamp(abs(mid - value) / range, 0.0, 1.0);
  return smoothstep(0.0, 1.0, x);
}


// Return a cold-hot color based on intensity [0-1]
vec3 temperature(float intensity)
{
  const vec3 blue   = vec3(0.0, 0.0, 1.0);
  const vec3 cyan   = vec3(0.0, 1.0, 1.0);
  const vec3 green  = vec3(0.0, 1.0, 0.0);
  const vec3 yellow = vec3(1.0, 1.0, 0.0);
  const vec3 red    = vec3(1.0, 0.0, 0.0);

  vec3 color = (fade(-0.25, 0.25, intensity) * blue    //
                + fade(0.0, 0.5, intensity) * cyan     //
                + fade(0.25, 0.75, intensity) * green  //
                + fade(0.5, 1.0, intensity) * yellow   //
                + smoothstep(0.75, 1.0, intensity) * red);
  return color;
}

// This material is the shading material after applying textures and any
// other operation. This structure is filled in gltfmaterial.glsl
struct Material
{
  vec3  albedo;
  float specular;
  vec3  emission;
  float anisotropy;
  float metallic;
  float roughness;
  float subsurface;
  float specularTint;
  float sheen;
  vec3  sheenTint;
  float clearcoat;
  float clearcoatRoughness;
  float transmission;
  float ior;
  vec3  attenuationColor;
  float attenuationDistance;

  //vec3  texIDs;
  // Roughness calculated from anisotropic
  float ax;
  float ay;
  // ----
  vec3  f0;
  float alpha;
  bool  unlit;
  bool  thinwalled;
};


// From shading state, this is the structure pass to the eval functions
struct State
{
  int   depth;
  float eta;

  vec3 position;
  vec3 normal;
  vec3 ffnormal;
  vec3 tangent;
  vec3 bitangent;
  vec2 texCoord;

  bool isEmitter;
  bool specularBounce;
  bool isSubsurface;

  uint     matID;
  Material mat;
};


//-------------------------------------------------------------------------------------------------
// Avoiding self intersections (see Ray Tracing Gems, Ch. 6)
//-----------------------------------------------------------------------
vec3 OffsetRay(in vec3 p, in vec3 n)
{
  const float intScale   = 256.0f;
  const float floatScale = 1.0f / 65536.0f;
  const float origin     = 1.0f / 32.0f;

  ivec3 of_i = ivec3(intScale * n.x, intScale * n.y, intScale * n.z);

  vec3 p_i = vec3(intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                  intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                  intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

  return vec3(abs(p.x) < origin ? p.x + floatScale * n.x : p_i.x,  //
              abs(p.y) < origin ? p.y + floatScale * n.y : p_i.y,  //
              abs(p.z) < origin ? p.z + floatScale * n.z : p_i.z);
}
