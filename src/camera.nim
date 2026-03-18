import sokol/app as sapp
import math
import math/vec3
import math/mat4
import types

proc worldToScreen*(worldPos: Vec3, proj, view: Mat4, width, height: float32): (Vec3, bool) =
  let clipPos = proj * view * worldPos
  
  # Check if behind camera using view-space Z
  let viewPos = view * worldPos
  if viewPos.z > 0.0: # In -Z forward, positive Z is behind
    return (vec3(0,0,0), false)

  # ndc to screen
  let screenX = (clipPos.x + 1.0) * 0.5 * width
  let screenY = (1.0 - clipPos.y) * 0.5 * height
  return (vec3(screenX, screenY, 0), true)

proc updateCamera*(state: var State, dt: float32) =
  if state.cameraMode == CameraMode.Follow:
    let targetPos = state.player.position
    let carRot = state.player.rotation
    # In my coordinate system, -Z is forward, so +Z is back
    let backDir = carRot * vec3(0, 0, 1)
    
    let radius = 10.0f
    let height = state.cameraOffsetY # Uses mouse scroll value
    let desiredCamPos = targetPos + (backDir * radius) + vec3(0, height, 0)
    
    # Smooth camera
    state.cameraPos = lerpV(state.cameraPos, desiredCamPos, 0.1)
    state.cameraTarget = lerpV(state.cameraTarget, targetPos + vec3(0, 1.5, 0), 0.2)
  else:
    # Front camera
    let targetPos = state.player.position
    let carRot = state.player.rotation
    let frontDir = carRot * vec3(0, 0, -1)
    state.cameraPos = targetPos + (frontDir * 8.0) + vec3(0, 3.0, 0)
    state.cameraTarget = targetPos
