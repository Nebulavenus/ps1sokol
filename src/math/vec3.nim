import math

type Vec3* = object
  x*: float32
  y*: float32
  z*: float32

proc vec3*(): Vec3 =
  result = Vec3(x: 0f, y: 0f, z: 0f)

proc vec3*(x: float32, y: float32, z: float32): Vec3 =
  result = Vec3(x: x, y: y, z: z)

proc zero*(): Vec3 =
  result = Vec3(x: 0f, y: 0f, z: 0f)

proc up*(): Vec3 =
  result = Vec3(x: 0f, y: 1f, z: 0f)

proc `+`*(v0: Vec3, v1: Vec3): Vec3 =
  result = Vec3(x: v0.x + v1.x, y: v0.y + v1.y, z: v0.z + v1.z)

proc `+=`*(v0: var Vec3, v1: Vec3) =
  v0.x += v1.x
  v0.y += v1.y
  v0.z += v1.z
  #result = Vec3(x: v0.x + v1.x, y: v0.y + v1.y, z: v0.z + v1.z)

proc `-`*(v0: Vec3, v1: Vec3): Vec3 =
  result = Vec3(x: v0.x - v1.x, y: v0.y - v1.y, z: v0.z - v1.z)

proc `-=`*(v0: var Vec3, v1: Vec3) =
  v0.x -= v1.x
  v0.y -= v1.y
  v0.z -= v1.z
  #result = Vec3(x: v0.x - v1.x, y: v0.y - v1.y, z: v0.z - v1.z)

proc `*`*(v0: Vec3, s: float32): Vec3 =
  result = Vec3(x: v0.x * s, y: v0.y * s, z: v0.z * s)

proc `/`*(v0: Vec3, s: float32): Vec3 =
  # TODO: handle division by zero
  result = Vec3(x: v0.x / s, y: v0.y / s, z: v0.z / s)

proc dot*(v0: Vec3, v1: Vec3): float32 =
  result = v0.x * v1.x + v0.y * v1.y + v0.z * v1.z

proc len*(v: Vec3): float32 =
  result = math.sqrt(dot(v, v))

proc lenSqr*(v: Vec3): float32 =
  return dot(v, v)

proc norm*(v: Vec3): Vec3 =
  let l = len(v);
  if l != 0f:
    return Vec3(x: v.x / l, y: v.y / l, z: v.z / l)
  else:
    return zero()

proc cross*(v0: Vec3, v1: Vec3): Vec3 =
  result = Vec3(
    x: (v0.y * v1.z) - (v0.z * v1.y),
    y: (v0.z * v1.x) - (v0.x * v1.z),
    z: (v0.x * v1.y) - (v0.y * v1.x),
  )

proc minV*(a, b: Vec3): Vec3 {.inline.} =
  ## Returns a new Vec3 with the component-wise minimum of two vectors.
  result.x = min(a.x, b.x)
  result.y = min(a.y, b.y)
  result.z = min(a.z, b.z)

proc maxV*(a, b: Vec3): Vec3 {.inline.} =
  ## Returns a new Vec3 with the component-wise maximum of two vectors.
  result.x = max(a.x, b.x)
  result.y = max(a.y, b.y)
  result.z = max(a.z, b.z)
