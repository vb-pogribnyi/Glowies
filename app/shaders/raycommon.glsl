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

  vec3 test_position;
  float test_distance;
};

struct shadowPayload
{
  bool isHit;
  uint seed;
};

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
