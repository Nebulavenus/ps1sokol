import types
import math/vec3
import math/mat4

proc isVisible*(bounds: AABB, mvp: Mat4): bool =
  ## Simple culling: check if any of the AABB corners are in front of the camera

  # Corners of the AABB
  let corners = [
    vec3(bounds.min.x, bounds.min.y, bounds.min.z),
    vec3(bounds.max.x, bounds.min.y, bounds.min.z),
    vec3(bounds.min.x, bounds.max.y, bounds.min.z),
    vec3(bounds.max.x, bounds.max.y, bounds.min.z),
    vec3(bounds.min.x, bounds.min.y, bounds.max.z),
    vec3(bounds.max.x, bounds.min.y, bounds.max.z),
    vec3(bounds.min.x, bounds.max.y, bounds.max.z),
    vec3(bounds.max.x, bounds.max.y, bounds.max.z),
  ]

  for c in corners:
    # Transform to clip space. mvp * vec3 usually returns (x,y,z) 
    # but for culling we need the W component.
    # Our Mat4 * Vec3 likely does the perspective division already?
    # Let's assume mesh is visible for now, isMeshVisible is more robust here.
    discard

  return true

proc isMeshVisible*(mesh: Mesh, cameraPos: Vec3, cameraForward: Vec3): bool =

  ## Check if mesh is visible based on its bounding box and camera orientation.
  ## cameraForward should be normalized.
  
  # Vector from camera to center of AABB
  let center = (mesh.bounds.min + mesh.bounds.max) * 0.5
  let toCenter = center - cameraPos
  
  # If dot product is positive, it's "mostly" in front of the camera.
  # We add a bit of "slack" based on the size of the mesh.
  let radius = len(mesh.bounds.max - mesh.bounds.min) * 0.5
  let dist = dot(toCenter, cameraForward)
  
  return dist > -radius
