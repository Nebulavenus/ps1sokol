import types
import math/vec3
import math/mat4
import math_utils
import options
import sokol/app as sapp

proc updateCamera*(state: var State, dt: float32) =
  let camOffset = vec3(0.0, state.cameraOffsetY, 8.0)
  const targetOffset = vec3(0.0, 1.0, 0.0)
  const followSpeed = 5.0

  let desiredPos = state.player.position + (state.player.rotation * camOffset)
  let desiredTarget = state.player.position + targetOffset

  let t = clamp(dt * followSpeed, 0.0, 1.0)
  state.cameraPos = vec3.lerpV(state.cameraPos, desiredPos, t)
  state.cameraTarget = vec3.lerpV(state.cameraTarget, desiredTarget, t)

proc getSurfaceInfo*(state: State, carPos: Vec3): Option[SurfaceHit] =
  let rayOrigin = carPos + vec3(0, 1.0, 0)
  let rayDir = vec3(0, -1.0, 0)
  const maxRayDist = 10.0

  var closestHitDist = maxRayDist
  var hitFound = false
  var hitNormal = vec3.up()

  for i in 0 ..< (state.roadCollisionIndices.len div 3):
    let i0 = state.roadCollisionIndices[i * 3 + 0]
    let i1 = state.roadCollisionIndices[i * 3 + 1]
    let i2 = state.roadCollisionIndices[i * 3 + 2]

    let v0 = vec3(state.roadCollisionVertices[i0].x, state.roadCollisionVertices[i0].y, state.roadCollisionVertices[i0].z)
    let v1 = vec3(state.roadCollisionVertices[i1].x, state.roadCollisionVertices[i1].y, state.roadCollisionVertices[i1].z)
    let v2 = vec3(state.roadCollisionVertices[i2].x, state.roadCollisionVertices[i2].y, state.roadCollisionVertices[i2].z)

    let dist = rayTriangleIntersectDist(rayOrigin, rayDir, v0, v1, v2, maxRayDist)

    if dist >= 0 and dist < closestHitDist:
      hitFound = true
      closestHitDist = dist
      let edge1 = v1 - v0
      let edge2 = v2 - v0
      hitNormal = norm(cross(edge1, edge2))
      if hitNormal.y < 0:
        hitNormal = hitNormal * -1.0

  if hitFound:
    let hitPos = rayOrigin + rayDir * closestHitDist
    return some(SurfaceHit(pos: hitPos, normal: hitNormal))
  else:
    return none[SurfaceHit]()

proc checkBarrierCollisions*(state: var State, carPos: Vec3, carRotation: Mat4): CollisionResponse =
  const carWidth = 0.8
  const carLength = 1.5
  const collisionRayLength = 0.5

  let rayOriginsLocal = [
    vec3( carWidth, 0.2, -carLength),
    vec3(-carWidth, 0.2, -carLength),
    vec3( carWidth, 0.2,  carLength),
    vec3(-carWidth, 0.2,  carLength),
  ]

  let carRight = carRotation * vec3(1, 0, 0)
  let carForward = carRotation * vec3(0, 0, -1)
  let rayDirections = [carRight, -carRight, carForward, -carForward]

  for localOrigin in rayOriginsLocal:
    let worldOrigin = carPos + (carRotation * localOrigin)

    for rayDir in rayDirections:
      for i in 0 ..< (state.barrierCollisionIndices.len div 3):
        let i0 = state.barrierCollisionIndices[i * 3 + 0]
        let i1 = state.barrierCollisionIndices[i * 3 + 1]
        let i2 = state.barrierCollisionIndices[i * 3 + 2]

        let v0 = vec3(state.barrierCollisionVertices[i0].x, state.barrierCollisionVertices[i0].y, state.barrierCollisionVertices[i0].z)
        let v1 = vec3(state.barrierCollisionVertices[i1].x, state.barrierCollisionVertices[i1].y, state.barrierCollisionVertices[i1].z)
        let v2 = vec3(state.barrierCollisionVertices[i2].x, state.barrierCollisionVertices[i2].y, state.barrierCollisionVertices[i2].z)

        let dist = rayTriangleIntersectDist(worldOrigin, rayDir, v0, v1, v2, collisionRayLength)

        if dist >= 0:
          var wallNormal = norm(cross(v1 - v0, v2 - v0))
          if dot(wallNormal, rayDir) > 0:
            wallNormal = wallNormal * -1.0

          let penetrationDepth = collisionRayLength - dist
          result.pushOut += wallNormal * penetrationDepth
          result.collided = true

          let velAlongNormal = dot(state.player.velocity, wallNormal)
          if velAlongNormal < 0:
            state.player.velocity = state.player.velocity * 0.98
            state.player.velocity -= wallNormal * velAlongNormal * 1.1

  return result
