import math
import math/vec3
import math/mat4
import types
import physics
import options

proc updateAI*(state: var State, dt: float32) =
  for i in 0 ..< state.aiCars.len:
    var ai = addr state.aiCars[i]
    let targetPos = state.pathNodes[ai.targetNode]
    let toTarget = targetPos - ai.position
    let dist = len(toTarget)
    
    if dist < 5.0:
      ai.targetNode = (ai.targetNode + 1) mod state.pathNodes.len
    
    let dir = norm(toTarget)
    let desiredYaw = (arctan2(dir.x, dir.z) + PI) * (180.0 / PI)
    
    # Smoothly rotate towards target
    var diff = desiredYaw - ai.yaw
    while diff > 180.0: diff -= 360.0
    while diff < -180.0: diff += 360.0
    
    # Difficulty parameters
    var turnSpeed = 2.0f
    var moveSpeed = 20.0f
    case ai.difficulty
    of Difficulty.Easy:
      turnSpeed = 1.0f
      moveSpeed = 15.0f
    of Difficulty.Medium:
      turnSpeed = 2.5f
      moveSpeed = 22.0f
    of Difficulty.Hard:
      turnSpeed = 4.5f
      moveSpeed = 30.0f

    ai.yaw += diff * dt * turnSpeed
    
    ai.rotation = rotate(ai.yaw, vec3(0, 1, 0))
    let forward = ai.rotation * vec3(0, 0, -1)
    
    # Simple movement
    ai.position += forward * moveSpeed * dt
    
    # Surface alignment
    let groundInfo = getSurfaceInfo(state, ai.position)
    if groundInfo.isSome:
      ai.position.y = groundInfo.get().pos.y + 0.9
      let groundNormal = groundInfo.get().normal
      let right = norm(cross(forward, groundNormal))
      let finalForward = norm(cross(groundNormal, right))
      # In our system -Z is forward, so the 3rd column (+Z) must be -finalForward
      ai.rotation = fromCols(right, groundNormal, -finalForward, vec3(0,0,0))
