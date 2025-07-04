import math
import vec3 as vec3

# Coordinate System
# Forward Axis: -Z Forward
# Up Axis: +Y Up

type Mat4* = object
  m*: array[4, array[4, float32]]

proc identity*(): Mat4 =
  result = Mat4(
    m: [
      [1f, 0f, 0f, 0f],
      [0f, 1f, 0f, 0f],
      [0f, 0f, 1f, 0f],
      [0f, 0f, 0f, 1f],
    ]
  )

proc mat4*(): Mat4 =
  result = identity()

proc zero*(): Mat4 =
  result = Mat4(
    m: [
      [0f, 0f, 0f, 0f],
      [0f, 0f, 0f, 0f],
      [0f, 0f, 0f, 0f],
      [0f, 0f, 0f, 0f],
    ]
  )

proc `*`*(m0: Mat4, m1: Mat4): Mat4 =
  result = zero()
  for col in 0..3:
    for row in 0..3:
      result.m[col][row] =
        m0.m[0][row] * m1.m[col][0] +
        m0.m[1][row] * m1.m[col][1] +
        m0.m[2][row] * m1.m[col][2] +
        m0.m[3][row] * m1.m[col][3]

proc `*`*(m: Mat4, v: vec3.Vec3): vec3.Vec3 =
  let w = m.m[0][3] * v.x + m.m[1][3] * v.y + m.m[2][3] * v.z + m.m[3][3]
  result.x = (m.m[0][0] * v.x + m.m[1][0] * v.y + m.m[2][0] * v.z + m.m[3][0]) / w
  result.y = (m.m[0][1] * v.x + m.m[1][1] * v.y + m.m[2][1] * v.z + m.m[3][1]) / w
  result.z = (m.m[0][2] * v.x + m.m[1][2] * v.y + m.m[2][2] * v.z + m.m[3][2]) / w

proc persp*(fov: float32, aspect: float32, near: float32, far: float32): Mat4 =
  result = identity()
  let t = math.tan(fov * (math.PI / 360f))
  result.m[0][0] = 1f / t
  result.m[1][1] = aspect / t
  result.m[2][3] = -1f
  result.m[2][2] = (near + far) / (near - far)
  result.m[3][2] = (2.0 * near * far) / (near - far)
  result.m[3][3] = 0f

proc lookat*(eye: Vec3, center: Vec3, up: Vec3): Mat4 =
  result = zero()
  let f = norm(center - eye)
  let s = norm(cross(f, up))
  let u = cross(s, f)
  result.m[0][0] = s.x
  result.m[0][1] = u.x
  result.m[0][2] = -f.x
  result.m[1][0] = s.y
  result.m[1][1] = u.y
  result.m[1][2] = -f.y
  result.m[2][0] = s.z
  result.m[2][1] = u.z
  result.m[2][2] = -f.z
  result.m[3][0] = -dot(s, eye)
  result.m[3][1] = -dot(u, eye)
  result.m[3][2] = dot(f, eye)
  result.m[3][3] = 1.0

proc radians(deg: float32): float32 =
  result = deg * (math.PI / 180f)

proc rotate*(angle: float32, axis_unorm: Vec3): Mat4 =
  let axis = norm(axis_unorm)
  let sin_theta = math.sin(radians(angle))
  let cos_theta = math.cos(radians(angle))
  let cos_value = 1.0 - cos_theta
  result = identity()
  result.m[0][0] = (axis.x * axis.x * cos_value) + cos_theta
  result.m[0][1] = (axis.x * axis.y * cos_value) + (axis.z * sin_theta)
  result.m[0][2] = (axis.x * axis.z * cos_value) - (axis.y * sin_theta)
  result.m[1][0] = (axis.y * axis.x * cos_value) - (axis.z * sin_theta)
  result.m[1][1] = (axis.y * axis.y * cos_value) + cos_theta
  result.m[1][2] = (axis.y * axis.z * cos_value) + (axis.x * sin_theta)
  result.m[2][0] = (axis.z * axis.x * cos_value) + (axis.y * sin_theta)
  result.m[2][1] = (axis.z * axis.y * cos_value) - (axis.x * sin_theta)
  result.m[2][2] = (axis.z * axis.z * cos_value) + cos_theta

proc translate*(translation: Vec3): Mat4 =
  result = identity()
  result.m[3][0] = translation.x
  result.m[3][1] = translation.y
  result.m[3][2] = translation.z

proc fromCols*(right, up, forward, translation: Vec3): Mat4 =
  ## Constructs a 4x4 matrix from its column vectors (basis axes and translation).
  ## This is useful for creating a rotation/orientation matrix from known direction vectors.
  result = zero()

  # Set the rotation/orientation part (first 3 columns)
  result.m[0][0] = right.x
  result.m[0][1] = right.y
  result.m[0][2] = right.z

  result.m[1][0] = up.x
  result.m[1][1] = up.y
  result.m[1][2] = up.z

  result.m[2][0] = forward.x
  result.m[2][1] = forward.y
  result.m[2][2] = forward.z

  # Set the translation part (4th column)
  result.m[3][0] = translation.x
  result.m[3][1] = translation.y
  result.m[3][2] = translation.z

  # Set the bottom row for a standard affine transformation matrix
  result.m[3][3] = 1.0
