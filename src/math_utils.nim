import math/vec2
import math/vec3
import std/random

proc lerp*(a, b: float32, t: float32): float32 {.inline.} =
  return a + (b - a) * t

proc vec2ToShort2N*(v: Vec2): (int16, int16) =
  ## Converts a Vec2 float [0.0, 1.0] to two int16s for use with SHORT2N.
  let clampedX = clamp(v.x, 0.0, 1.0)
  let clampedY = clamp(v.y, 0.0, 1.0)
  let normX = (clampedX * 2.0) - 1.0
  let normY = (clampedY * 2.0) - 1.0
  let shortX = cast[int16](normX * 32767.0)
  let shortY = cast[int16](normY * 32767.0)
  return (shortX, shortY)

proc vec2ToUshort2n*(v: Vec2): (uint16, uint16) =
  ## Converts a Vec2 float [0.0, 1.0] to two uint16s for use with USHORT2N.
  let clampedX = clamp(v.x, 0.0, 1.0)
  let clampedY = clamp(v.y, 0.0, 1.0)
  let uvX = cast[uint16](clampedX * 65535.0)
  let uvY = cast[uint16](clampedY * 65535.0)
  return (uvX, uvY)

proc randomHemisphereDirection*(normal: Vec3): Vec3 =
  ## Generates a random direction within a hemisphere oriented by the normal
  var dir = vec3(rand(-1.0..1.0), rand(-1.0..1.0), rand(-1.0..1.0))
  while lenSqr(dir) > 1.0 or lenSqr(dir) == 1.0:
    dir = vec3(rand(-1.0..1.0), rand(-1.0..1.0), rand(-1.0..1.0))
  dir = norm(dir)
  if dot(dir, normal) < 0.0:
    dir = Vec3(x: -dir.x, y: -dir.y, z: -dir.z)
  return dir

proc rayTriangleIntersect*(rayOrigin, rayDir: Vec3, v0, v1, v2: Vec3, maxDist: float): bool =
  ## Check if a ray intersects a triangle using the Möller-Trumbore algorithm.
  const EPSILON = 0.000001
  let edge1 = v1 - v0
  let edge2 = v2 - v0
  let h = cross(rayDir, edge2)
  let a = dot(edge1, h)
  if a > -EPSILON and a < EPSILON:
    return false
  let f = 1.0 / a
  let s = rayOrigin - v0
  let u = f * dot(s, h)
  if u < 0.0 or u > 1.0:
    return false
  let q = cross(s, edge1)
  let v = f * dot(rayDir, q)
  if v < 0.0 or u + v > 1.0:
    return false
  let t = f * dot(edge2, q)
  if t > EPSILON and t < maxDist:
    return true
  else:
    return false

proc rayTriangleIntersectDist*(rayOrigin, rayDir: Vec3, v0, v1, v2: Vec3, maxDist: float): float =
  const EPSILON = 0.000001
  let edge1 = v1 - v0
  let edge2 = v2 - v0
  let h = cross(rayDir, edge2)
  let a = dot(edge1, h)
  if a > -EPSILON and a < EPSILON: return -1.0
  let f = 1.0 / a
  let s = rayOrigin - v0
  let u = f * dot(s, h)
  if u < 0.0 or u > 1.0: return -1.0
  let q = cross(s, edge1)
  let v = f * dot(rayDir, q)
  if v < 0.0 or u + v > 1.0: return -1.0
  let t = f * dot(edge2, q)
  if t > EPSILON and t < maxDist:
    return t
  else:
    return -1.0
