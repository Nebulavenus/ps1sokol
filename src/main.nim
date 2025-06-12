import sokol/log as slog
import sokol/app as sapp
import sokol/gfx as sg
import sokol/glue as sglue
import shaders/default as shd
import math/vec2
import math/vec3
import math/mat4
import qoi
import math
import strutils
import tables
import os
import streams
import std/random
import audio

type Vertex = object
  x, y, z: float32
  xN, yN, zN: float32
  color: uint32
  u, v: float32
  bxN, byN, bzN: float32 # Bent Normal vector

type Mesh = object
  bindings: Bindings
  indexCount: int32

proc vec2ToShort2N(v: Vec2): (int16, int16) =
  ## Converts a Vec2 float [0.0, 1.0] to two int16s for use with SHORT2N.

  # First, ensure the input is within the expected [0, 1] range. This is robust.
  let clampedX = clamp(v.x, 0.0, 1.0)
  let clampedY = clamp(v.y, 0.0, 1.0)

  # --- This is the crucial math ---
  # 1. Remap the [0, 1] range to the [-1, 1] range.
  #    (value * 2.0) - 1.0 does this perfectly.
  let normX = (clampedX * 2.0) - 1.0
  let normY = (clampedY * 2.0) - 1.0

  # 2. Scale the [-1, 1] float to the [ -32767, 32767 ] integer range.
  let shortX = cast[int16](normX * 32767.0)
  let shortY = cast[int16](normY * 32767.0)

  return (shortX, shortY)

proc vec2ToUshort2n(v: Vec2): (uint16, uint16) =
  ## Converts a Vec2 float [0.0, 1.0] to two uint16s for use with USHORT2N.

  # Robustly clamp the input to the expected [0, 1] range.
  let clampedX = clamp(v.x, 0.0, 1.0)
  let clampedY = clamp(v.y, 0.0, 1.0)

  # --- This is the crucial math ---
  # Scale the [0, 1] float directly to the [0, 65535] integer range.
  let uvX = cast[uint16](clampedX * 65535.0)
  let uvY = cast[uint16](clampedY * 65535.0)

  return (uvX, uvY)

proc loadObj(path: string): (seq[Vertex], seq[uint16]) =
  var
    temp_positions: seq[Vec3]
    temp_normals: seq[Vec3]
    temp_uvs: seq[Vec2]
    out_vertices: seq[Vertex]
    out_indices: seq[uint16]
    vertex_cache: Table[string, uint16]

  if not os.fileExists(path):
    echo("Cannot load OBJ file, path not found: " & path)
    return

  echo "Starting OBJ Loading"
  for line in lines(path):
    if line.startsWith("v "):
      let parts = line.split()
      temp_positions.add(vec3(
        parts[1].parseFloat,
        parts[2].parseFloat,
        parts[3].parseFloat,
      ))
    elif line.startsWith("vn "):
      let parts = line.split()
      temp_normals.add(vec3(
        parts[1].parseFloat,
        parts[2].parseFloat,
        parts[3].parseFloat,
      ))
    elif line.startsWith("vt "):
      let parts = line.split()
      temp_uvs.add(vec2(
        parts[1].parseFloat,
        parts[2].parseFloat,
      ))
    elif line.startsWith("f "):
      let face_parts = line.split()
      for i in 1..3: # For each vertex in the triangle face
        let key = face_parts[i]

        if not vertex_cache.haskey(key):
          var
            pos_idx = -1
            uv_idx = -1
            nrm_idx = -1

          let v_parts = key.split('/')

          # Parse based on the number of components found
          case v_parts.len
          of 1: # Format "v"
            pos_idx = v_parts[0].parseInt - 1
          of 2: # Format "v/vt"
            pos_idx = v_parts[0].parseInt - 1
            uv_idx = v_parts[1].parseInt - 1
          of 3: # Format "v/vt/vn" or "v//vn"
            pos_idx = v_parts[0].parseInt - 1
            if v_parts[1].len > 0: # Check if vt is present
              uv_idx = v_parts[1].parseInt - 1
            nrm_idx = v_parts[2].parseInt - 1
          else:
            echo("Unsupported face format component: " & key)
            continue

          # Create the vertex, providing defaults for missing data
          let pos = if pos_idx != -1: temp_positions[pos_idx] else: vec3(0,0,0)
          let nrm = if nrm_idx != -1: temp_normals[nrm_idx] else: vec3(0,1,0) # Default normal points up
          var uv  = if uv_idx != -1: temp_uvs[uv_idx] else: vec2(0,0) # Default UV is 0,0
          # Invert uv, modern rendering convention
          uv.y = 1.0 - uv.y
          let uvS = vec2ToUshort2N(uv)
          # Obj doesn't store vertex colors... by default white
          let new_vert = Vertex(
            x: pos.x, y: pos.y, z: pos.z,
            xN: nrm.x, yN: nrm.y, zN: nrm.z,
            color: 0xFFFFFFFF'u32,
            #u: uvS[0], v: uvS[1]
            u: uv.x, v: uv.y
          )
          out_vertices.add(new_vert)
          let new_idx = (out_vertices.len - 1).uint16
          vertex_cache[key] = new_idx
          out_indices.add(new_idx)
        else:
          # Vertex already exists, just add its index
          out_indices.add(vertex_cache[key])

  echo "Loaded OBJ $1: $2 vertices, $3 indices" % [$path, $out_vertices.len, $out_indices.len]
  return (out_vertices, out_indices)

proc packColor(r, g, b, a: uint8): uint32 {.inline.} =
  ## Packs four 8-bit color channels into a single 32-bit integer.
  ## The byte order (AABBGGRR) is what Sokol's UBYTE4N format expects on little-endian systems
  ## to correctly map to an RGBA vec4 in the shader.
  result = (uint32(a) shl 24) or (uint32(b) shl 16) or (uint32(g) shl 8) or uint32(r)

# Add these new helper procedures
proc unpackColor(c: uint32): (uint8, uint8, uint8, uint8) {.inline.} =
  ## Unpacks a 32-bit AABBGGRR color into four 8-bit channels
  # We extract specific bits with mask `and` and then shift it
  # to their correct position of a simple byte u8
  let r = (c and 0x000000FF'u32).uint8
  let g = ((c and 0x0000FF00'u32) shr 8).uint8
  let b = ((c and 0x00FF0000'u32) shr 16).uint8
  let a = ((c and 0xFF000000'u32) shr 24).uint8
  return (r, g, b, a)

# Standard method to generate a random point within a unit sphere.
proc randomHemisphereDirection(normal: Vec3): Vec3 =
  ## Generates a random direction within a hemisphere oriented by the normal
  var dir = vec3(rand(-1.0..1.0), rand(-1.0..1.0), rand(-1.0..1.0))
  while lenSqr(dir) > 1.0 or lenSqr(dir) == 1.0:
    # Keep generating until we get a point inside the unit sphere (to enusre uniform distribution)
    dir = vec3(rand(-1.0..1.0), rand(-1.0..1.0), rand(-1.0..1.0))

  dir = norm(dir)
  # If the random direction is pointing "into" the surface, flip it
  if dot(dir, normal) < 0.0:
    dir = Vec3(x: -dir.x, y: -dir.y, z: -dir.z)
  return dir

# Slower
proc randomHemisphereDirectionMarsaglia(normal: Vec3): Vec3 =
  var a = rand(0.0..1.0)
  var b = rand(0.0..1.0)
  var theta = arccos(2 * a - 1)
  var phi = 2 * math.PI * b
  var x = sin(theta) * cos(phi)
  var y = sin(theta) * sin(phi)
  var z = cos(theta)

  var dir = vec3(x, y, z)
  # If the random direction is pointing "into" the surface, flip it
  if dot(dir, normal) < 0.0:
    dir = Vec3(x: -dir.x, y: -dir.y, z: -dir.z)
  return dir

proc rayTriangleIntersect(rayOrigin, rayDir: Vec3, v0, v1, v2: Vec3, maxDist: float): bool =
  ## Check if a ray intersects a triangle using the Möller-Trumbore algorithm.
  ## Returns true on intersection within maxDist, false otherwise
  const EPSILON = 0.000001
  let edge1 = v1 - v0
  let edge2 = v2 - v0
  let h = cross(rayDir, edge2)
  let a = dot(edge1, h)

  if a > -EPSILON and a < EPSILON:
    return false # Ray is parallel to the triangle

  let f = 1.0 / a
  let s = rayOrigin - v0
  let u = f * dot(s, h)

  if u < 0.0 or u > 1.0:
    return false

  let q = cross(s, edge1)
  let v = f * dot(rayDir, q)

  if v < 0.0 or u + v > 1.0:
    return false

  # At this point we have an intersection. Check if it's within the max distance
  let t = f * dot(edge2, q)
  if t > EPSILON and t < maxDist:
    return true # Ray intersection
  else:
    return false # Intersection is too far away or behind the ray origin

# Axis-Aligned Bounding Box
type AABB = object
  min, max: Vec3

# Uniform Grid acceleration structure
type UniformGrid = object
  bounds: AABB
  dims: (int, int, int) # Number of cells in each dimension (x, y, z)
  cellSize: Vec3        # Size of a single cell in world units
  cells: seq[seq[int]]  # 1D list of cells, each cell is a list of triangle

# Helper to create uniform grid which encompass the mesh
proc initUniformGrid(vertices: seq[Vertex], resolution: int): UniformGrid =
  echo "Initializing Uniform Grid..."
  # 1. Calculate the mesh's overall bounding box (AABB)
  result.bounds.min = vec3(Inf, Inf, Inf)
  result.bounds.max = vec3(-Inf, -Inf, -Inf)
  for v in vertices:
    result.bounds.min.x = min(result.bounds.min.x, v.x)
    result.bounds.min.y = min(result.bounds.min.y, v.y)
    result.bounds.min.z = min(result.bounds.min.z, v.z)
    result.bounds.max.x = max(result.bounds.max.x, v.x)
    result.bounds.max.y = max(result.bounds.max.y, v.y)
    result.bounds.max.z = max(result.bounds.max.z, v.z)

  # 2. Determine grid dimensions based on resolution and aspect ratio
  let size = result.bounds.max - result.bounds.min
  let maxDim = max(size.x, max(size.y, size.z))
  result.dims = (
    max(1, int(size.x / maxDim * resolution.float)),
    max(1, int(size.y / maxDim * resolution.float)),
    max(1, int(size.z / maxDim * resolution.float))
  )
  echo "  Grid dimensions: ", result.dims

  # 3. Calculate cell size and initialize cell storage
  result.cellSize = vec3(
    size.x / result.dims[0].float,
    size.y / result.dims[1].float,
    size.z / result.dims[2].float,
  )
  let totalCells = result.dims[0] * result.dims[1] * result.dims[2]
  result.cells = newSeq[seq[int]](totalCells)

proc worldToCell(grid: UniformGrid, pos: Vec3): (int, int, int) =
  ## Converts a world-space position to grid cell coordinates
  let ix = clamp(int((pos.x - grid.bounds.min.x) / grid.cellSize.x), 0, grid.dims[0] - 1)
  let iy = clamp(int((pos.y - grid.bounds.min.y) / grid.cellSize.y), 0, grid.dims[1] - 1)
  let iz = clamp(int((pos.z - grid.bounds.min.z) / grid.cellSize.z), 0, grid.dims[2] - 1)
  return (ix, iy, iz)

## Populates the grid by placing triangle indices into the cells they overlap
proc populateGrid(grid: var UniformGrid, vertices: seq[Vertex], indices: seq[uint16]) =
  echo "Populating Grid with ", (indices.len div 3), " triangles..."
  # For each triangle...
  for i in 0 ..< (indices.len div 3):
    let i0 = indices[i * 3 + 0]
    let i1 = indices[i * 3 + 1]
    let i2 = indices[i * 3 + 2]
    let v0 = vec3(vertices[i0].x, vertices[i0].y, vertices[i0].z)
    let v1 = vec3(vertices[i1].x, vertices[i1].y, vertices[i1].z)
    let v2 = vec3(vertices[i2].x, vertices[i2].y, vertices[i2].z)

    # Find the AABB of the triangle itself
    var triBounds: AABB
    triBounds.min = minV(v0, minV(v1, v2))
    triBounds.max = maxV(v0, maxV(v1, v2))

    # Convert the triangle's AABB to grid cell index ranges
    let (minX, minY, minZ) = worldToCell(grid, triBounds.min)
    let (maxX, maxY, maxZ) = worldToCell(grid, triBounds.max)

    # Insert the triangle index into all cells it overlaps
    for z in minZ..maxZ:
      for y in minY..maxY:
        for x in minX..maxX:
          let cellIndex = z * grid.dims[0] * grid.dims[1] + y * grid.dims[0] + x
          grid.cells[cellIndex].add(i)

type AOBakeParams = object
  # Number of rays to cast per vertex. More is better but slower. (e.g., 64, 128)
  numRays: int
  # How far a ray can travel to cause occlusion. Prevents distant geometry from affecting local AO.
  maxDistance: float
  # How strong the darkening effect is. (e.g., 1.0)
  intensity: float
  # A small offset to push the ray origin away from the vertex to prevent self-intersection. (e.g., 0.001)
  bias: float

# This function calculates bent normals instead of a simple AO float.
proc bakeBentNormalWithGrid(vertices: var seq[Vertex], indices: seq[uint16], params: AOBakeParams, gridResolution: int) =
  ## Bakes a bent normal vector into each vertex using a Uniform Grid for acceleration.
  var grid = initUniformGrid(vertices, gridResolution)
  populateGrid(grid, vertices, indices)

  echo "Starting Bent Normal AO bake for ", vertices.len, " vertices..."
  let totalVerts = vertices.len
  var progressCounter = 0

  for i in 0 ..< vertices.len:
    let vert = vertices[i]
    let geoNormal = norm(vec3(vert.xN, vert.yN, vert.zN))
    let origin = vec3(vert.x, vert.y, vert.z) + geoNormal * params.bias

    var sumOfUnoccludedDirections = vec3(0, 0, 0)

    for r in 0 ..< params.numRays:
      let rayDir = randomHemisphereDirection(geoNormal)

      # Grid traversal setup (DDA-like algorithm)
      var (ix, iy, iz) = worldToCell(grid, origin)
      let stepX = if rayDir.x > 0: 1 else: -1
      let stepY = if rayDir.y > 0: 1 else: -1
      let stepZ = if rayDir.z > 0: 1 else: -1

      # Handle division by zero for axis-aligned rays
      let tiny = 1.0e-6
      let tDeltaX = if abs(rayDir.x) < tiny: 1.0e38 else: abs(grid.cellSize.x / rayDir.x)
      let tDeltaY = if abs(rayDir.y) < tiny: 1.0e38 else: abs(grid.cellSize.y / rayDir.y)
      let tDeltaZ = if abs(rayDir.z) < tiny: 1.0e38 else: abs(grid.cellSize.z / rayDir.z)

      let nextBx = grid.bounds.min.x + (ix.float + (if stepX > 0: 1 else: 0)) * grid.cellSize.x
      let nextBy = grid.bounds.min.y + (ix.float + (if stepY > 0: 1 else: 0)) * grid.cellSize.y
      let nextBz = grid.bounds.min.z + (ix.float + (if stepZ > 0: 1 else: 0)) * grid.cellSize.z

      var tMaxX = if abs(rayDir.x) < tiny: 1.0e38 else: (nextBx - origin.x) / rayDir.x
      var tMaxY = if abs(rayDir.y) < tiny: 1.0e38 else: (nextBy - origin.y) / rayDir.y
      var tMaxZ = if abs(rayDir.z) < tiny: 1.0e38 else: (nextBz - origin.z) / rayDir.z

      var rayIsOccluded = false
      while not rayIsOccluded:
        # Check for intersection with triangles in the current cell
        let cellIndex = iz * grid.dims[0] * grid.dims[1] + iy * grid.dims[0] + ix

        for triIndex in grid.cells[cellIndex]:
          let i0 = indices[triIndex * 3 + 0]
          let i1 = indices[triIndex * 3 + 1]
          let i2 = indices[triIndex * 3 + 2]
          if i0 == i.uint16 or i1 == i.uint16 or i2 == i.uint16: continue

          let v0 = vec3(vertices[i0].x, vertices[i0].y, vertices[i0].z)
          let v1 = vec3(vertices[i1].x, vertices[i1].y, vertices[i1].z)
          let v2 = vec3(vertices[i2].x, vertices[i2].y, vertices[i2].z)

          if rayTriangleIntersect(origin, rayDir, v0, v1, v2, params.maxDistance):
            rayIsOccluded = true
            break # Found hit, stop checking triangles in this cell

          if rayIsOccluded: break

        # Step to the next cell
        if tMaxX < tMaxY:
          if tMaxX < tMaxZ:
            ix += stepX; tMaxX += tDeltaX
          else:
            iz += stepZ; tMaxZ += tDeltaZ
        else:
          if tMaxY < tMaxZ:
            iy += stepY; tMaxY += tDeltaY
          else:
            iz += stepZ; tMaxZ += tDeltaZ

        # Exit if ray leaves the grid
        if ix < 0 or ix >= grid.dims[0] or
            iy < 0 or iy >= grid.dims[1] or
            iz < 0 or iz >= grid.dims[2]:
              break

      if not rayIsOccluded:
        sumOfUnoccludedDirections += rayDir

    # After all rays are cast, average the sum of directions.
    # The resulting vector is the Bent Normal. Its length naturally encodes occlusion.
    let bentNormal = sumOfUnoccludedDirections / params.numRays.float
    vertices[i].bxN = bentNormal.x
    vertices[i].byN = bentNormal.y
    vertices[i].bzN = bentNormal.z

    # Progress report
    progressCounter += 1
    if progressCounter mod (totalVerts div 10) == 0:
      echo "  AO Bake progress: ", round(float(progressCounter) / float(totalVerts) * 100, 2), "%"

  echo "Accelerated Ambient Occlusion bake complete."

proc bakeAmbientOcclusionWithGrid(vertices: var seq[Vertex], indices: seq[uint16], params: AOBakeParams, gridResolution: int) =
  ## Bakes a raw ambient occlusion value using a Unfiform Grid for acceleration
  var grid = initUniformGrid(vertices, gridResolution)
  populateGrid(grid, vertices, indices)

  echo "Starting Accelerated Ambient Occlusion bake for ", vertices.len, " vertices..."
  let totalVerts = vertices.len
  var progressCounter = 0

  for i in 0 ..< vertices.len:
    let vert = vertices[i]
    let normal = norm(vec3(vert.xN, vert.yN, vert.zN))
    let origin = vec3(vert.x, vert.y, vert.z) + normal * params.bias
    var occludedCount = 0

    for r in 0 ..< params.numRays:
      let rayDir = randomHemisphereDirection(normal)

      # Grid traversal setup (DDA-like algorithm)
      var (ix, iy, iz) = worldToCell(grid, origin)
      let stepX = if rayDir.x > 0: 1 else: -1
      let stepY = if rayDir.y > 0: 1 else: -1
      let stepZ = if rayDir.z > 0: 1 else: -1

      # Handle division by zero for axis-aligned rays
      let tiny = 1.0e-6
      let tDeltaX = if abs(rayDir.x) < tiny: 1.0e38 else: abs(grid.cellSize.x / rayDir.x)
      let tDeltaY = if abs(rayDir.y) < tiny: 1.0e38 else: abs(grid.cellSize.y / rayDir.y)
      let tDeltaZ = if abs(rayDir.z) < tiny: 1.0e38 else: abs(grid.cellSize.z / rayDir.z)

      let nextBx = grid.bounds.min.x + (ix.float + (if stepX > 0: 1 else: 0)) * grid.cellSize.x
      let nextBy = grid.bounds.min.y + (ix.float + (if stepY > 0: 1 else: 0)) * grid.cellSize.y
      let nextBz = grid.bounds.min.z + (ix.float + (if stepZ > 0: 1 else: 0)) * grid.cellSize.z

      var tMaxX = if abs(rayDir.x) < tiny: 1.0e38 else: (nextBx - origin.x) / rayDir.x
      var tMaxY = if abs(rayDir.y) < tiny: 1.0e38 else: (nextBy - origin.y) / rayDir.y
      var tMaxZ = if abs(rayDir.z) < tiny: 1.0e38 else: (nextBz - origin.z) / rayDir.z

      var rayIsOccluded = false
      while not rayIsOccluded:
        # Check for intersection with triangles in the current cell
        let cellIndex = iz * grid.dims[0] * grid.dims[1] + iy * grid.dims[0] + ix

        for triIndex in grid.cells[cellIndex]:
          let i0 = indices[triIndex * 3 + 0]
          let i1 = indices[triIndex * 3 + 1]
          let i2 = indices[triIndex * 3 + 2]
          if i0 == i.uint16 or i1 == i.uint16 or i2 == i.uint16: continue

          let v0 = vec3(vertices[i0].x, vertices[i0].y, vertices[i0].z)
          let v1 = vec3(vertices[i1].x, vertices[i1].y, vertices[i1].z)
          let v2 = vec3(vertices[i2].x, vertices[i2].y, vertices[i2].z)

          if rayTriangleIntersect(origin, rayDir, v0, v1, v2, params.maxDistance):
            occludedCount += 1
            rayIsOccluded = true
            break # Found hit, stop checking triangles in this cell

          if rayIsOccluded: break

        # Step to the next cell
        if tMaxX < tMaxY:
          if tMaxX < tMaxZ:
            ix += stepX; tMaxX += tDeltaX
          else:
            iz += stepZ; tMaxZ += tDeltaZ
        else:
          if tMaxY < tMaxZ:
            iy += stepY; tMaxY += tDeltaY
          else:
            iz += stepZ; tMaxZ += tDeltaZ

        # Exit if ray leaves the grid
        if ix < 0 or ix >= grid.dims[0] or
            iy < 0 or iy >= grid.dims[1] or
            iz < 0 or iz >= grid.dims[2]:
              break

    # Calculate AO value
    # TODO: disabled, because removed it from Vertex format. ao: float
    #vertices[i].ao = float(occludedCount) / float(params.numRays)

    # Progress report
    progressCounter += 1
    if progressCounter mod (totalVerts div 10) == 0:
      echo "  AO Bake progress: ", round(float(progressCounter) / float(totalVerts) * 100, 2), "%"

  echo "Accelerated Ambient Occlusion bake complete."

proc bakeAmbientOcclusion(vertices: var seq[Vertex], indices: seq[uint16], params: AOBakeParams) =
  ## Bakes ambient occlusion into vertex colors by raycasting.
  ## This is a slow, one-time operation on the CPU
  echo "Starting Ambient Occlusion bake for ", vertices.len, " vertices..."
  let totalVerts = vertices.len
  var progressCounter = 0

  for i in 0 ..< vertices.len:
    let vert = vertices[i]
    let normal = norm(vec3(vert.xN, vert.yN, vert.zN))
    let origin = vec3(vert.x, vert.y, vert.z) + normal * params.bias
    var occludedCount = 0

    for r in 0 ..< params.numRays:
      let rayDir = randomHemisphereDirection(normal)
      #let rayDir = randomHemisphereDirectionMarsaglia(normal)

      # Check this ray against all triangles in the mesh
      for f in 0 ..< (indices.len div 3):
        let i0 = indices[f * 3]
        let i1 = indices[f * 3 + 1]
        let i2 = indices[f * 3 + 2]

        # Don't check for intersection with triangles connected to the current vertex
        if i0 == i.uint16 or i1 == i.uint16 or i2 == i.uint16:
          continue

        let v0 = vec3(vertices[i0].x, vertices[i0].y, vertices[i0].z)
        let v1 = vec3(vertices[i1].x, vertices[i1].y, vertices[i1].z)
        let v2 = vec3(vertices[i2].x, vertices[i2].y, vertices[i2].z)

        if rayTriangleIntersect(origin, rayDir, v0, v1, v2, params.maxDistance):
          occludedCount += 1
          break # This ray is occluded, no need to check other triangles. Move to the next ray

    # Calculate occlusion factor (0.0 = fully lit, 1.0 = fully occluded)
    let occlusionFactor = float(occludedCount) / float(params.numRays)

    # Save it into vertices
    # TODO: disabled, because removed it from Vertex format. ao: float
    #vertices[i].ao = occlusionFactor

    # Progress report
    progressCounter += 1
    if progressCounter mod (totalVerts div 10) == 0:
      echo "  AO Bake progress: ", round(float(progressCounter) / float(totalVerts) * 100, 2), "%"

  echo "Ambient Occlusion bake complete."


proc loadPly(path: string): (seq[Vertex], seq[uint16]) =
  ## Loads a 3D model from an ASCII PLY file.
  ##
  ## Features:
  ## - Parses vertex properties: x, y, z, nx, ny, nz, s, t (or u, v).
  ## - Parses vertex colors: red, green, blue, alpha (as uchar).
  ## - Provides sane defaults for missing properties (normals, uvs, colors).
  ## - Supports triangular (3) and quadrilateral (4) faces, triangulating quads automatically.
  ## - Robustly handles any property order in the header.

  var
    out_vertices: seq[Vertex]
    out_indices: seq[uint16]

  if not os.fileExists(path):
    echo "loadPly: Cannot load PLY file, path not found: " & path
    return

  # --- 1. Header Parsing ---
  var
    vertexCount = 0
    faceCount = 0
    inHeader = true
    # Maps property name ("x", "nx") to its column index
    # as they appear after element order, int stores its column index
    vertexPropertyMap: Table[string, int]
    vertexPropertyCount = 0
    parsingVertex = false

  let fileLines = readFile(path).splitLines()
  var bodyStartIndex = -1

  for i, line in fileLines:
    if not inHeader: continue

    let parts = line.split()
    if parts.len == 0: continue

    case parts[0]
    of "ply": discard # Standard magic number
    of "format":
      if parts.len > 1 and parts[1] == "ascii":
        echo "Ply is in ASCII format"
      else:
        echo "Unsupported or invalid PLY format"
        return
    of "comment": discard
    of "element":
      parsingVertex = false # Reset state when a new element is found
      if parts.len == 3 and parts[1] == "vertex":
        vertexCount = parts[2].parseInt
        parsingVertex = true
        vertexPropertyCount = 0
      elif parts.len == 3 and parts[1] == "face":
        faceCount = parts[2].parseInt
    of "property":
      if parsingVertex and parts.len == 3:
        # We only care about vertex properties for now
        var propName = parts[^1]
        # Allow both "s, t" and "u, v" for texture coords
        if propName == "u": propName = "s"
        if propName == "v": propName = "t"
        vertexPropertyMap[propName] = vertexPropertyCount
        vertexPropertyCount += 1
    of "end_header":
      inHeader = false
      bodyStartIndex = i + 1
      break # Exit header parsing loop
    else: discard

  if bodyStartIndex == -1:
    echo "loadPly, Failed to parse PLY header"
    return

  echo "loadPly, Header parsed. Vertices: $1, Faces: $2" % [$vertexCount, $faceCount]

  # --- 2. Body Parsing (Vertices) ---
  let vertexLinesEnd = bodyStartIndex + vertexCount
  out_vertices.setLen(vertexCount)
  for i in bodyStartIndex ..< vertexLinesEnd:
    let parts = fileLines[i].split()
    if parts.len != vertexPropertyCount:
      echo "loadPly, Vertex line has incorrect number of properties, skipping."
      continue

    # Helper to safely get a property value or a default
    proc getProp(name: string, default: float): float =
      if vertexPropertyMap.haskey(name):
        result = parts[vertexPropertyMap[name]].parseFloat
      else:
        result = default

    # Position (required)
    let x = getProp("x", 0.0)
    let y = getProp("y", 0.0)
    let z = getProp("z", 0.0)

    # Normals (optional) # by default points up
    let nx = getProp("nx", 0.0)
    let ny = getProp("ny", 1.0)
    let nz = getProp("nz", 0.0)

    # UVs (optional)
    let u = getProp("s", 0.0)
    var v = getProp("t", 0.0)
    # Uncomment if textures appear upside down
    v = 1.0 - v

    # Colors (optional)
    let r = getProp("red", 255.0).uint8
    let g = getProp("green", 255.0).uint8
    let b = getProp("blue", 255.0).uint8
    let a = getProp("alpha", 255.0).uint8
    let color = packColor(r, g, b, a)

    out_vertices[i - bodyStartIndex] = Vertex(
      x: x.float32, y: y.float32, z: z.float32,
      xN: nx.float32, yN: ny.float32, zN: nz.float32,
      color: color,
      u: u.float32, v: v.float32
    )

  # --- 3. Body Parsing (Faces) ---
  let faceLinesEnd = vertexLinesEnd + faceCount
  for i in vertexLinesEnd ..< faceLinesEnd:
    let parts = fileLines[i].split()
    let numVertsInFace = parts[0].parseInt

    case numVertsInFace
    of 3: # Triangle
      let i0 = parts[1].parseInt.uint16
      let i1 = parts[2].parseInt.uint16
      let i2 = parts[3].parseInt.uint16
      # Clockwise order ABC-i0i1i2
      #out_indices.add([i0, i1, i2])
      # Counter-clockwise order - that what uses obj right?
      out_indices.add([i0, i2, i1]) # CBA
    of 4: # Quad - triangulate it
      let i0 = parts[1].parseInt.uint16
      let i1 = parts[2].parseInt.uint16
      let i2 = parts[3].parseInt.uint16
      let i3 = parts[4].parseInt.uint16
      # First triangle (0, 1, 2)
      out_indices.add([i0, i1, i2])
      # Second triangle (0, 2, 3) - common for convex quads
      out_indices.add([i0, i2, i3])
    else:
      echo "loadPly, Unsupported face with $1 vertices. Only triangles (3) and quads (4) are supported." % $numVertsInFace

  # --- 4. Create Sokol Buffers and Final Mesh ---
  if out_vertices.len == 0 or out_indices.len == 0:
    echo "loadPly, No vertices or indices were loaded from the PLY file"
    return

  echo "loadPly, Loaded PLY: $1 vertices, $2 indices" % [$out_vertices.len, $out_indices.len]
  return (out_vertices, out_indices)

# --- Caching and mesh processing procedures ---
proc saveMeshToCache(path: string, vertices: seq[Vertex], indices: seq[uint16]) =
  ## Saves the processed vertex and index data to a fast binary cache file
  echo "Saving baked mesh to cache: ", path
  var stream = newFileStream(path, fmWrite)
  if stream == nil:
    echo "Error: Could not open cache file for writing: ", path
    return

  try:
    # Write a simple header: vertex count and index count
    stream.write(vertices.len)
    stream.write(indices.len)
    # Write the raw data blobs
    stream.writeData(vertices[0].addr, vertices.len * sizeof(Vertex))
    stream.writeData(indices[0].addr, indices.len * sizeof(uint16))
  finally:
    stream.close()

proc loadMeshFromCache(path: string): (seq[Vertex], seq[uint16]) =
  ## Loads vertex and index data from a binary cache file.
  echo "Loading baked mesh from cache: ", path
  var stream = newFileStream(path, fmRead)
  if stream == nil:
    echo "Error: Could not open cache file for reading: ", path
    return

  var vertices: seq[Vertex]
  var indices: seq[uint16]

  try:
    var vertCount, idxCount: int
    stream.read(vertCount)
    stream.read(idxCount)

    vertices.setLen(vertCount)
    indices.setLen(idxCount)

    discard stream.readData(vertices[0].addr, vertCount * sizeof(Vertex))
    discard stream.readData(indices[0].addr, idxCount * sizeof(uint16))
  finally:
    stream.close()

  return (vertices, indices)

proc loadAndProcessMesh(modelPath: string, aoParams: AOBakeParams, texture: Image, sampler: Sampler): Mesh =
  ## High-level procedure to load a mesh.
  ## It will use a cached version if available, otherwise it will load,
  ## bake AO, and save a new version to the cache.
  var
    cpuVertices: seq[Vertex]
    cpuIndices: seq[uint16]

  let cachePath = modelPath & ".baked_ao.bin"

  if os.fileExists(cachePath):
    # Load directly from the fast binary cache
    (cpuVertices, cpuIndices) = loadMeshFromCache(cachePath)
  else:
    # Cache not found, do the full loading and baking process
    let fileExt = modelPath.splitFile.ext
    case fileExt.toLower()
    of ".ply":
      (cpuVertices, cpuIndices) = loadPly(modelPath)
    of ".obj":
      (cpuVertices, cpuIndices) = loadObj(modelPath)
    else:
      echo "Unsupported model format: ", fileExt
      return

    if cpuVertices.len > 0:
      # Bake Ambient Occlusion with GRID
      #bakeAmbientOcclusion(cpuVertices, cpuIndices, aoParams)
      #bakeAmbientOcclusionWithGrid(cpuVertices, cpuIndices, aoParams, gridResolution = 64)
      bakeBentNormalWithGrid(cpuVertices, cpuIndices, aoParams, gridResolution = 64)
      # Save the result to cache for the next run
      saveMeshToCache(cachePath, cpuVertices, cpuIndices)

  # --- GPU Upload ---
  if cpuVertices.len > 0 and cpuIndices.len > 0:
    let vbuf = sg.makeBuffer(BufferDesc(
      usage: BufferUsage(vertexBuffer: true),
      data: sg.Range(addr: cpuVertices[0].addr, size: cpuVertices.len * sizeof(Vertex))
    ))
    let ibuf = sg.makeBuffer(BufferDesc(
      usage: BufferUsage(indexBuffer: true),
      data: sg.Range(addr: cpuIndices[0].addr, size: cpuIndices.len * sizeof(uint16))
    ))
    result.indexCount = cpuIndices.len.int32
    result.bindings = Bindings(vertexBuffers: [vbuf], indexBuffer: ibuf)
    result.bindings.images[shd.imgUTexture] = texture
    result.bindings.samplers[shd.smpUSampler] = sampler
  else:
    echo "Error: No vertex data to upload to GPU."

proc loadTexture(textureNamePath: string): sg.Image =
  # load qoi texture
  var qoiImage: QOIF
  try:
    qoiImage = readQOI(textureNamePath)
    #qoiImage = readQOI("assets/malenia.qoi")
    echo "Success loaded qoi: " & textureNamePath, qoiImage.header.width, "-", qoiImage.header.height
  except Exception as e:
    echo "Error loading qoi"
    requestQuit()

  var finalPixelData: seq[byte]
  var finalPixelFormat: sg.PixelFormat
  if qoiImage.header.channels == qoi.RGBA:
    finalPixelData = qoiImage.data
    finalPixelFormat = sg.PixelFormat.pixelFormatRgba8
  else:
    # Conversion required
    finalPixelFormat = sg.PixelFormat.pixelFormatRgba8
    let numPixels = qoiImage.header.width.int * qoiImage.header.height.int
    finalPixelData = newSeq[byte](numPixels * 4)
    # Write data
    var srcIndex = 0
    var dstIndex = 0
    for i in 0 ..< numPixels:
      # Copy R, G, B
      finalPixelData[dstIndex]   = qoiImage.data[srcIndex]     # R
      finalPixelData[dstIndex+1] = qoiImage.data[srcIndex+1]   # G
      finalPixelData[dstIndex+2] = qoiImage.data[srcIndex+2]   # B
      # Add the Alpha channel
      finalPixelData[dstIndex+3] = 255.byte                   # A (fully opaque)

      srcIndex += 3
      dstIndex += 4

  result = sg.makeImage(sg.ImageDesc(
    width: qoiImage.header.width.int32,
    height: qoiImage.header.height.int32,
    pixelFormat: finalPixelFormat,
    data: ImageData(
      subimage: [ [ sg.Range(addr: finalPixelData[0].addr, size: qoiImage.header.width.int32 * qoiImage.header.height.int32 * 4) ] ]
    )
  ))

type PlayerVehicle = object
  position: Vec3
  velocity: Vec3
  rotation: Mat4
  yaw: float32
  angularVelocity: float32

type InputState = object
  accelerate: bool
  brake: bool
  turnLeft: bool
  turnRight: bool
  drift: bool

type State = object
  pip: Pipeline
  passAction: sg.PassAction
  mesh: Mesh # Track mesh
  ocean: Mesh # Ocean mesh
  oceanFrame: float
  carMesh: Mesh # Player's car mesh
  input: InputState
  player: PlayerVehicle
  cameraOffsetY: float
  cameraPos: Vec3 # Camera's actual world position
  cameraTarget: Vec3 # Point the camera is looking at
  vsParams: VsParams
  fsParams: FsParams
  # -- Controlling AO Multi-Layered --
  aoShadowStrength: float32
  skyLightColor: Vec3
  skyLightIntensity: float32
  groundLightColor: Vec3
  groundLightIntensity: float32


var state: State

const
  passAction = PassAction(
    colors: [
      ColorAttachmentAction(
        loadAction: loadActionClear,
        clearValue: (0.25, 0.5, 0.75, 1) # same for fog color
      )
    ]
  )

proc init() {.cdecl.} =
  sg.setup(sg.Desc(
    environment: sglue.environment(),
    logger: sg.Logger(fn: slog.fn),
  ))
  audioInit()

  case sg.queryBackend():
    of backendGlcore: echo "using GLCORE33 backend"
    of backendD3d11: echo "using D3D11 backend"
    of backendMetalMacos: echo "using Metal backend"
    else: echo "using untested backend"

  # create a checker board texture
  let pixels = [
    0xFFFFFFFF'u32, 0xFF000000'u32, 0xFFFFFFFF'u32, 0xFF000000'u32,
    0xFF000000'u32, 0xFFFFFFFF'u32, 0xFF000000'u32, 0xFFFFFFFF'u32,
    0xFFFFFFFF'u32, 0xFF000000'u32, 0xFFFFFFFF'u32, 0xFF000000'u32,
    0xFF000000'u32, 0xFFFFFFFF'u32, 0xFF000000'u32, 0xFFFFFFFF'u32,
  ]
  #bindings.images[shd.imgTex] = sg.makeImage(sg.ImageDesc(
  let checkboardImg = sg.makeImage(sg.ImageDesc(
    width: 4,
    height: 4,
    data: ImageData(
      subimage: [ [ sg.Range(addr: pixels.addr, size: pixels.sizeof) ] ]
    )
  ))

  # create shader and pipeline object
  state.pip = sg.makePipeline(PipelineDesc(
    shader: sg.makeShader(shd.texcubeShaderDesc(sg.queryBackend())),
    layout: VertexLayoutState(
      attrs: [
        VertexAttrState(format: vertexFormatFloat3),  # position
        VertexAttrState(format: vertexFormatFloat3),  # normal
        VertexAttrState(format: vertexFormatUbyte4n), # color0
        VertexAttrState(format: vertexFormatFloat2),  # texcoord0
        VertexAttrState(format: vertexFormatFloat3),  # bentNormal
      ],
    ),
    indexType: indexTypeUint16,
    #faceWinding: faceWindingCcw,
    #cullMode: cullModeBack,
    cullMode: cullModeNone,
    depth: DepthState(
      compare: compareFuncLessEqual,
      writeEnabled: true,
    )
  ))

  # Load mesh & also preprocessing if needed
  let assetDir = getAppDir() & DirSep
  let trackPath = assetDir & "track.ply"
  let carPath = assetDir & "car.ply"
  let oceanPath = assetDir & "ocean.ply"

  # Define AO parameters
  let aoParams = AOBakeParams(
    numRays: 32,
    maxDistance: 2.0,
    intensity: 1.0,
    bias: 0.001,
  )
  # Also store real-time AO variables
  state.aoShadowStrength = 1.0
  state.skyLightColor = vec3(0.4, 0.5, 0.8) # Nice light blue sky
  state.skyLightIntensity = 0.0
  state.groundLightColor = vec3(0.6, 0.4, 0.3) # Warm earthy ground bounce
  state.groundLightIntensity = 0.0

  # create a matching sampler - for everything
  let pointSmp = sg.makeSampler(sg.SamplerDesc(
    minFilter: filterNearest,
    magFilter: filterNearest,
  ));

  # Load the mesh. One function handles everything
  #let trackTexture = loadTexture("assets/diffuse.qoi")
  let oceanTexture = loadTexture("assets/ocean.qoi")
  var ocean = loadAndProcessMesh(oceanPath, aoParams, oceanTexture, pointSmp)

  let trackTexture = loadTexture("assets/track1.qoi")
  var mesh = loadAndProcessMesh(trackPath, aoParams, trackTexture, pointSmp)

  let carTexture = loadTexture("assets/car2.qoi")
  var carMesh = loadAndProcessMesh(carPath, aoParams, carTexture, pointSmp)

  # Don't forget to save it in state
  state.mesh = mesh
  state.ocean = ocean
  state.carMesh = carMesh

  # --- Setup camera & player ---
  state.player.position = vec3(0.0, 0.2, 0.0)
  state.player.velocity = vec3(0, 0, 0)
  state.player.yaw = 0.0
  state.player.angularVelocity = 0.0
  state.player.rotation = rotate(state.player.yaw, vec3(0, 1, 0))

  # Camera a bit behind the player
  state.cameraPos = vec3(0.0, 10.0, 2.0)
  state.cameraOffsetY = 5.0
  state.cameraTarget = state.player.position

proc computeVsParams(): shd.VsParams =
  let proj = persp(60.0f, sapp.widthf() / sapp.heightf(), 0.01f, 150.0f)
  let view = lookat(state.cameraPos, state.cameraTarget, vec3.up())

  # This model matrix is for the static geometry (the track)
  # If the track is already at the world origin, this is just the identity matrix
  let model = mat4.identity()

  result = shd.VsParams(
    u_mvp: proj * view * model,
    u_model: model,
    u_camPos: state.cameraPos,
    u_jitterAmount: 240.0, # Simulate a 240p vertical resolution
  )

proc computeFsParams(): shd.FsParams =
  result = shd.FsParams(
    u_fogColor: vec3(0.25f, 0.5f, 0.75f),
    u_fogNear: 4.0f,
    u_fogFar: 150.0f,
    u_ditherSize: vec2(320.0, 240.0), # Should be equal to window size
    # -- AO uniforms --
    u_aoShadowStrength: state.aoShadowStrength,
    u_skyLightColor: state.skyLightColor,
    u_skyLightIntensity: state.skyLightIntensity,
    u_groundLightColor: state.groundLightColor,
    u_groundLightIntensity: state.groundLightIntensity
  )

proc updateCamera(dt: float32) =
  # -- Constants to tweak the camera feel --
  #const camOffset = vec3(0.0, 5.0, 8.0) # How far behind and above the car
  let camOffset = vec3(0.0, state.cameraOffsetY, 8.0) # How far behind and above the car
  const targetOffset = vec3(0.0, 1.0, 0.0)  # Look slightly above the car's pivot
  const followSpeed = 5.0 # How quickly the camera catches up (higher is tighter)

  # 1. Calculate the desired camera position in world space.
  # We take the camera offset and rotate it by the car's rotation.
  let desiredPos = state.player.position + (state.player.rotation * camOffset)

  # 2. Calculate the desired look-at target.
  let desiredTarget = state.player.position + targetOffset

  # 3. Smoothly interpolate the camera's actual position towards the desired one.
  # The `dt * followSpeed` makes the interpolation frame-rate independent.
  let t = clamp(dt * followSpeed, 0.0, 1.0)
  state.cameraPos = vec3.lerpV(state.cameraPos, desiredPos, t)
  state.cameraTarget = vec3.lerpV(state.cameraTarget, desiredTarget, t)

proc frame() {.cdecl.} =
  #let dt = sapp.frameDuration() * 60f
  let dt = sapp.frameDuration()

  # --- Logic ---
  # 1. Update player rotation matrix based on yaw from input
  block VehiclePhysics:
    # --- Constants to Tweak ---
      const engineForce = 15.0    # How much power the engine has
      const turningTorque = 60.0   # How quickly the car can start to turn
      const brakeForce = 5.0    # How powerful the brakes are
      const drag = 0.9            # Air resistance, slows down at high speed
      const angularDrag = 1.3     # Stops the car from spinning forever
      const baseGrip = 2.0        # Renamed for clarity: Base grip strength
      const driftGripMultiplier = 0.2 # How much grip is reduced when drifting (e.g., 0.2 means 80% less grip)
      const driftTurningMultiplier = 1.5 # How much more torque you get when drifting

      # Store previous velocity to calculate acceleration later
      let prevVelocity = state.player.velocity

      # --- APPLY FORCES FROM INPUT ---
      let forwardDir = state.player.rotation * vec3(0, 0, -1)

      # Determine current grip and turning torque based on drift state
      var currentGrip = baseGrip
      var currentTurningTorque = turningTorque

      if state.input.drift:
        currentGrip *= driftGripMultiplier
        currentTurningTorque *= driftTurningMultiplier

      if state.input.accelerate:
        state.player.velocity += forwardDir * engineForce * dt

      if state.input.brake:
        # Brakes are more effective if they oppose the current velocity
        if len(state.player.velocity) > 0.1:
          state.player.velocity -= norm(state.player.velocity) * brakeForce * dt

      # Use the currentTurningTorque here
      let turnFactor = pow(len(state.player.velocity) / 10, 2)
      if state.input.turnLeft:
        state.player.angularVelocity += currentTurningTorque * dt * turnFactor

      if state.input.turnRight:
        state.player.angularVelocity -= currentTurningTorque * dt * turnFactor
      # --- END INPUT FORCES ---

      # 1. Apply Drag/Friction
      state.player.velocity = state.player.velocity * (1.0 - (drag * dt))
      state.player.angularVelocity = state.player.angularVelocity * (1.0 - (angularDrag * dt))

      # 2. Update Rotation and Position from Velocities
      state.player.yaw += state.player.angularVelocity * dt
      state.player.position += state.player.velocity * dt
      state.player.rotation = rotate(state.player.yaw, vec3(0, 1, 0))

      # 3. Align Velocity with Forward Direction (The "Grip" part)
      # This is the magic that makes the car feel like it's driving, not just floating.
      # It gradually turns the car's momentum vector to match the way the car is pointing.
      let currentSpeed = len(state.player.velocity)

      # --- Calculate Speed and Acceleration for Audio ---
      let carAccel = (currentSpeed - len(prevVelocity)) / dt
      updateEngineSound(currentSpeed, carAccel)

      # Handle zero velocity for norm() safely for lerp's first argument
      # If speed is very low, assume velocity direction is forward to avoid NaN from norm(zero_vec)
      let velocityDirection = if currentSpeed > 0.01: norm(state.player.velocity) else: forwardDir

      # Use the calculated currentGrip for the lerp factor
      let newVelocityDir = norm(lerpV(velocityDirection, forwardDir, clamp(currentGrip * dt, 0.0, 1.0)))

      if currentSpeed > 0.01: # Avoid issues with normalizing a zero vector when assigning
          state.player.velocity = newVelocityDir * currentSpeed
      else:
          # If speed is zero, ensure velocity stays zero or aligns without movement.
          # This prevents tiny residual velocities from causing issues when stopped.
          state.player.velocity = vec3(0,0,0)

  # 2. Update the camera's position to follow the player
  updateCamera(dt)

  # --- Call Audio Sample Generation ---
  audioGenerateSamples()

  # --- Rendering ---
  # 3. Common matrices and fragment shader uniforms
  let fsParams = computeFsParams()
  let proj = persp(60.0f, sapp.widthf() / sapp.heightf(), 0.01f, 150.0f)
  let view = lookat(state.cameraPos, state.cameraTarget, vec3.up())

  # -- Similar models uses similar pipelines
  sg.beginPass(Pass(action: passAction, swapchain: sglue.swapchain()))
  sg.applyPipeline(state.pip)
  sg.applyUniforms(shd.ubFsParams, sg.Range(addr: fsParams.addr, size: fsParams.sizeof))

  # --- Draw ocean ---
  var oceanModel = identity()
  state.oceanFrame += dt / 1.5
  let offset = (sin(state.oceanFrame) * 0.4) # adjust the frequency and amplitude as needed
  let translationMatrix = translate(vec3(0, offset, 0))
  oceanModel = oceanModel * translationMatrix
  var oceanVsParams = shd.VsParams(
    u_mvp: proj * view * oceanModel,
    u_model: oceanModel,
    u_camPos: state.cameraPos,
    u_jitterAmount: 240.0,
  )
  sg.applyBindings(state.ocean.bindings)
  sg.applyUniforms(shd.ubVsParams, sg.Range(addr: oceanVsParams.addr, size: oceanVsParams.sizeof))
  sg.draw(0, state.ocean.indexCount, 1)

  # --- Draw track ---
  let trackModel = identity()
  var trackVsParams = shd.VsParams(
    u_mvp: proj * view * trackModel,
    u_model: trackModel,
    u_camPos: state.cameraPos,
    u_jitterAmount: 240.0,
  )
  sg.applyBindings(state.mesh.bindings)
  sg.applyUniforms(shd.ubVsParams, sg.Range(addr: trackVsParams.addr, size: trackVsParams.sizeof))
  sg.draw(0, state.mesh.indexCount, 1)

  # --- Draw car ---
  let carModel = translate(state.player.position) * state.player.rotation
  var carVsParams = shd.VsParams(
    u_mvp: proj * view * carModel,
    u_model: carModel,
    u_camPos: state.cameraPos,
    u_jitterAmount: 240.0,
  )
  sg.applyBindings(state.carMesh.bindings)
  sg.applyUniforms(shd.ubVsParams, sg.Range(addr: carVsParams.addr, size: carVsParams.sizeof))
  sg.draw(0, state.carMesh.indexCount, 1)

  sg.endPass()
  sg.commit()

proc cleanup() {.cdecl.} =
  audioShutdown()
  sg.shutdown()

proc event(e: ptr sapp.Event) {.cdecl.} =
  # Mouse
  #[
  if e.`type` == EventType.eventTypeMouseMove:
    let mouseSensitivity = 0.005 # Adjust this value to your liking
    state.camYaw   += e.mouseDx * mouseSensitivity
    state.camPitch -= e.mouseDy * mouseSensitivity # Subtract because positive dy is mouse down

    # Clamp pitch to prevent the camera from flipping over
    const pitchLimit = 1.55 # ~89 degrees in radians
    if state.camPitch > pitchLimit: state.camPitch = pitchLimit
    if state.camPitch < -pitchLimit: state.camPitch = -pitchLimit
  ]#
  if e.`type` == EventType.eventTypeMouseScroll:
    state.cameraOffsetY += e.scrollY
    state.cameraOffsetY = max(state.cameraOffsetY, 0.0)

  # Keyboard
  if e.`type` == EventType.eventTypeKeyDown or e.`type` == EventType.eventTypeKeyUp:
    # AO control
    let step: float32 = 0.05 # Use a smaller step for finer control

    let isDown = e.`type` == EventType.eventTypeKeyDown

    case e.keyCode
    of keyCodeEscape:
      sapp.requestQuit()
    of keyCodeW: # Accelerate
      state.input.accelerate = isDown
    of keyCodeS: # Brake/Rverese
      state.input.brake = isDown
    of keyCodeA: # Steer Left
      state.input.turnLeft = isDown
    of keyCodeD: # Steer Right
      state.input.turnRight = isDown
    of keyCodeSpace: # Drift
      state.input.drift = isDown
    # -- AO realtime controlling --
    of keyCode1: state.aoShadowStrength = max(0.0, state.aoShadowStrength - step); echo "Shadow Str: ", state.aoShadowStrength
    of keyCode2: state.aoShadowStrength += step; echo "Shadow Str: ", state.aoShadowStrength
    of keyCode3: state.skyLightIntensity = max(0.0, state.skyLightIntensity - step); echo "Sky Light Str: ", state.skyLightIntensity
    of keyCode4: state.skyLightIntensity += step; echo "Sky Light Str: ", state.skyLightIntensity
    of keyCode5: state.groundLightIntensity = max(0.0, state.groundLightIntensity - step); echo "Ground Light Str: ", state.groundLightIntensity
    of keyCode6: state.groundLightIntensity += step; echo "Ground Light Str: ", state.groundLightIntensity
    else: discard

# main

sapp.run(sapp.Desc(
  initCb: init,
  frameCb: frame,
  eventCb: event,
  cleanupCb: cleanup,
  windowTitle: "Game",
  width: 640,
  height: 480,
  sampleCount: 4,
  icon: IconDesc(sokol_default: true),
  logger: sapp.Logger(fn: slog.fn)
))
