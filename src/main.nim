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

type Vertex = object
  x, y, z: float32
  xN, yN, zN: float32
  color: uint32
  u, v: float32
  ao: float32 # Raw baked AO value [0.0, 1.0]

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

type AOBakeParams = object
  # Number of rays to cast per vertex. More is better but slower. (e.g., 64, 128)
  numRays: int
  # How far a ray can travel to cause occlusion. Prevents distant geometry from affecting local AO.
  maxDistance: float
  # How strong the darkening effect is. (e.g., 1.0)
  intensity: float
  # A small offset to push the ray origin away from the vertex to prevent self-intersection. (e.g., 0.001)
  bias: float

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
    vertices[i].ao = occlusionFactor

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
      # Bake Ambient Occlusion
      bakeAmbientOcclusion(cpuVertices, cpuIndices, aoParams)
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

type State = object
  pip: Pipeline
  passAction: sg.PassAction
  mesh: Mesh
  camTime: float32
  camPos: Vec3
  camYaw: float32
  camPitch: float32
  vsParams: VsParams
  fsParams: FsParams
  rx, ry: float32
  # -- Controlling AO --
  aoMode: int
  aoIntensity: float32
  aoDetailColor: Vec3
  aoBaseColor: Vec3

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
  let texcubeImg = sg.makeImage(sg.ImageDesc(
    width: 4,
    height: 4,
    data: ImageData(
      subimage: [ [ sg.Range(addr: pixels.addr, size: pixels.sizeof) ] ]
    )
  ))

  # create a matching sampler
  #bindings.samplers[shd.smpSmp] = sg.makeSampler(sg.SamplerDesc(
  let texcubeSmp = sg.makeSampler(sg.SamplerDesc(
    minFilter: filterNearest,
    magFilter: filterNearest,
  ));

  # load qoi texture
  var qoiImage: QOIF
  try:
    #qoiImage = readQOI("assets/diffuse.qoi")
    #qoiImage = readQOI("diffuse.qoi")
    qoiImage = readQOI("assets/malenia.qoi")
    #qoiImage = readQOI("malenia.qoi")
    echo "Success loaded qoi: diffuse.qoi ", qoiImage.header.width, "-", qoiImage.header.height
    echo "First byte is not null! ", qoiImage.data[160]
    echo "Data is not null! ", qoiImage.data.len
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

  let qoiTexture = sg.makeImage(sg.ImageDesc(
    width: qoiImage.header.width.int32,
    height: qoiImage.header.height.int32,
    pixelFormat: finalPixelFormat,
    data: ImageData(
      #subimage: [ [ sg.Range(addr: qoiImage.data[0].addr, size: qoiImage.data.sizeof) ] ]
      #subimage: [ [ sg.Range(addr: qoiImage.data[0].addr, size: 16) ] ]
      #subimage: [ [ sg.Range(addr: qoiImage.data[0].addr, size: qoiImage.header.width.int32 * qoiImage.header.height.int32 * 4) ] ]
      #subimage: [ [ sg.Range(addr: finalPixelData[0].addr, size: qoiImage.header.width.int32 * qoiImage.header.height.int32 * 4) ] ]
      #subimage: [ [ sg.Range(addr: finalPixelData.addr, size: qoiImage.header.width.int32 * qoiImage.header.height.int32 * 4) ] ]
      subimage: [ [ sg.Range(addr: finalPixelData[0].addr, size: qoiImage.header.width.int32 * qoiImage.header.height.int32 * 4) ] ]
      #subimage: [ [ sg.Range(addr: qoiImage.data[0].addr, size: qoiImage.data.sizeof) ] ]
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
        VertexAttrState(format: vertexFormatFloat),  # ao
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
  # save everything in mesh after processing it
  var mesh: Mesh

  let assetDir = getAppDir() & DirSep
  let modelPath = assetDir & "malenia.ply"

  # Define AO parameters
  let aoParams = AOBakeParams(
    numRays: 64,
    maxDistance: 1.0,
    intensity: 1.0,
    bias: 0.001,
  )
  # Also store real-time AO variables
  #state.aoMode = AOBakeMode.Multiply
  state.aoMode = 0
  state.aoIntensity = 1.5
  state.aoDetailColor = vec3(0.2, 0.1, 0.05) # "dirt"
  state.aoBaseColor = vec3(0.1, 0.1, 0.2) # Cool ambient

  # Load the mesh. One function handles everything
  mesh = loadAndProcessMesh(modelPath, aoParams, qoiTexture, texcubeSmp)

  #mesh.bindings.images[shd.imgUTexture] = texcubeImg
  #mesh.bindings.images[shd.imgUTexture] = qoiTexture
  #mesh.bindings.samplers[shd.smpUSampler] = texcubeSmp
  state.mesh = mesh

proc computeVsParams(): shd.VsParams =
  let camStart = state.camPos + vec3(0.0, 2.5, 4.0)
  let camEnd = state.camPos + vec3(0.0, 0.5, 12.0)
  let dt = sapp.frameDuration()
  let speed = 0.3 # cycles per second
  state.camTime += dt
  #let t = (1.0 + math.sin(2.0 * 3.14159 * speed * state.camTime)) / 2.0 # move sin [-1,1] to [0, 1]
  #let camPos = camStart + (camEnd - camStart) * t
  let camPos = state.camPos

  let proj = persp(60.0f, sapp.widthf() / sapp.heightf(), 0.01f, 50.0f)

  # Calculate the camera's forward direction vector using spherical coordinates
  let forwardVec = norm(vec3(
    cos(state.camPitch) * sin(state.camYaw),
    sin(state.camPitch),
    cos(state.camPitch) * -cos(state.camYaw)
  ))
  let lookAtPoint = camPos + forwardVec
  let view = lookat(camPos, lookAtPoint, vec3.up())

  let rxm = rotate(state.rx, vec3(1f, 0f, 0f))
  let rym = rotate(state.ry, vec3(0f, 1f, 0f))
  let model = rxm * rym
  result = shd.VsParams(
    u_mvp: proj * view * model,
    u_model: model,
    u_camPos: camPos,
    u_jitterAmount: 240.0, # Simulate a 240p vertical resolution
    #u_jitterAmount: 480.0, # Simulate a 240p vertical resolution
  )

proc computeFsParams(): shd.FsParams =
  result = shd.FsParams(
    u_fogColor: vec3(0.25f, 0.5f, 0.75f),
    u_fogNear: 4.0f,
    u_fogFar: 20.0f,
    u_ditherSize: vec2(320.0, 240.0), # Should be equal to window size
    # -- AO uniforms --
    u_aoMode: state.aoMode.int32,
    u_aoIntensity: state.aoIntensity,
    u_aoDetailColor: state.aoDetailColor,
    u_aoBaseColor: state.aoBaseColor
  )

proc frame() {.cdecl.} =
  let dt = sapp.frameDuration() * 60f
  #state.rx += 1f * dt
  #state.ry += 2f * dt

  let vsParams = computeVsParams()
  let fsParams = computeFsParams()

  sg.beginPass(Pass(action: passAction, swapchain: sglue.swapchain()))
  sg.applyPipeline(state.pip)
  sg.applyBindings(state.mesh.bindings)
  sg.applyUniforms(shd.ubVsParams, sg.Range(addr: vsParams.addr, size: vsParams.sizeof))
  sg.applyUniforms(shd.ubFsParams, sg.Range(addr: fsParams.addr, size: fsParams.sizeof))
  sg.draw(0, state.mesh.indexCount, 1)
  sg.endPass()
  sg.commit()

proc cleanup() {.cdecl.} =
  sg.shutdown()

proc event(e: ptr sapp.Event) {.cdecl.} =
  # Mouse
  if e.`type` == EventType.eventTypeMouseMove:
    let mouseSensitivity = 0.005 # Adjust this value to your liking
    state.camYaw   += e.mouseDx * mouseSensitivity
    state.camPitch -= e.mouseDy * mouseSensitivity # Subtract because positive dy is mouse down

    # Clamp pitch to prevent the camera from flipping over
    const pitchLimit = 1.55 # ~89 degrees in radians
    if state.camPitch > pitchLimit: state.camPitch = pitchLimit
    if state.camPitch < -pitchLimit: state.camPitch = -pitchLimit

  # Keyboard
  if e.`type` == EventType.eventTypeKeyDown:
    let moveSpeed = 0.5
    let rotSpeed = 0.3
    let forwardVec = vec3(sin(state.camYaw), 0.0, -cos(state.camYaw))
    let rightVec = vec3(forwardVec.z, 0.0, -forwardVec.x)

    case e.keyCode
    of keyCodeEscape:
      sapp.requestQuit()
    of keyCodeW:
      state.camPos += (forwardVec * moveSpeed)
    of keyCodeS:
      state.camPos -= forwardVec * moveSpeed
    of keyCodeA:
      state.camPos -= rightVec * moveSpeed
    of keyCodeD:
      state.camPos += rightVec * moveSpeed
    of keyCodeQ:
      state.camPos.y += moveSpeed
    of keyCodeE:
      state.camPos.y -= moveSpeed
    # -- AO realtime controlling --
    of keyCodeM:
      # Cycle through AO modes
      let nextMode = (state.aoMode.int + 1) mod 3
      state.aoMode = nextMode
      echo "Set AO Mode to: ", state.aoMode
    of keyCodeUp:
      state.aoIntensity += 0.1
      echo "AO Intensity: ", state.aoIntensity
    of keyCodeDown:
      state.aoIntensity = max(0.0, state.aoIntensity - 0.1)
      echo "AO Intensity: ", state.aoIntensity
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
