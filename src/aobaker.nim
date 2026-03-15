import types
import math/vec3
import math
import math_utils

# Helper to create uniform grid which encompass the mesh
proc initUniformGrid*(vertices: seq[Vertex], resolution: int): UniformGrid =
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

proc worldToCell*(grid: UniformGrid, pos: Vec3): (int, int, int) =
  ## Converts a world-space position to grid cell coordinates
  let ix = clamp(int((pos.x - grid.bounds.min.x) / grid.cellSize.x), 0, grid.dims[0] - 1)
  let iy = clamp(int((pos.y - grid.bounds.min.y) / grid.cellSize.y), 0, grid.dims[1] - 1)
  let iz = clamp(int((pos.z - grid.bounds.min.z) / grid.cellSize.z), 0, grid.dims[2] - 1)
  return (ix, iy, iz)

## Populates the grid by placing triangle indices into the cells they overlap
proc populateGrid*(grid: var UniformGrid, vertices: seq[Vertex], indices: seq[uint16]) =
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

# This function calculates bent normals instead of a simple AO float.
proc bakeBentNormalWithGrid*(vertices: var seq[Vertex], indices: seq[uint16], params: AOBakeParams, gridResolution: int) =
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
      let nextBy = grid.bounds.min.y + (iy.float + (if stepY > 0: 1 else: 0)) * grid.cellSize.y
      let nextBz = grid.bounds.min.z + (iz.float + (if stepZ > 0: 1 else: 0)) * grid.cellSize.z

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

proc bakeAmbientOcclusionWithGrid*(vertices: var seq[Vertex], indices: seq[uint16], params: AOBakeParams, gridResolution: int) =
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
      let nextBy = grid.bounds.min.y + (iy.float + (if stepY > 0: 1 else: 0)) * grid.cellSize.y
      let nextBz = grid.bounds.min.z + (iz.float + (if stepZ > 0: 1 else: 0)) * grid.cellSize.z

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

    # Progress report
    progressCounter += 1
    if progressCounter mod (totalVerts div 10) == 0:
      echo "  AO Bake progress: ", round(float(progressCounter) / float(totalVerts) * 100, 2), "%"

  echo "Accelerated Ambient Occlusion bake complete."

proc bakeAmbientOcclusion*(vertices: var seq[Vertex], indices: seq[uint16], params: AOBakeParams) =
  ## Bakes ambient occlusion into vertex colors by raycasting.
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

      # Check this ray against all triangles in the mesh
      for f in 0 ..< (indices.len div 3):
        let i0 = indices[f * 3]
        let i1 = indices[f * 3 + 1]
        let i2 = indices[f * 3 + 2]

        if i0 == i.uint16 or i1 == i.uint16 or i2 == i.uint16:
          continue

        let v0 = vec3(vertices[i0].x, vertices[i0].y, vertices[i0].z)
        let v1 = vec3(vertices[i1].x, vertices[i1].y, vertices[i1].z)
        let v2 = vec3(vertices[i2].x, vertices[i2].y, vertices[i2].z)

        if rayTriangleIntersect(origin, rayDir, v0, v1, v2, params.maxDistance):
          occludedCount += 1
          break

    # Progress report
    progressCounter += 1
    if progressCounter mod (totalVerts div 10) == 0:
      echo "  AO Bake progress: ", round(float(progressCounter) / float(totalVerts) * 100, 2), "%"

  echo "Ambient Occlusion bake complete."
