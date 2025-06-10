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

type Vertex = object
  x, y, z: float32
  xN, yN, zN: float32
  color: uint32
  #u, v: uint16
  u, v: float32

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

proc loadObj(path: string): Mesh =
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
  let vbuf = sg.makeBuffer(BufferDesc(
    usage: BufferUsage(vertexBuffer: true),
    data: sg.Range(addr: out_vertices[0].addr, size: out_vertices.len * sizeof(Vertex))
  ))
  let ibuf = sg.makeBuffer(BufferDesc(
    usage: BufferUsage(indexBuffer: true),
    data: sg.Range(addr: out_indices[0].addr, size: out_indices.len * sizeof(uint16))
  ))
  result.indexCount = out_indices.len.int32
  result.bindings = Bindings(vertexBuffers: [vbuf], indexBuffer: ibuf)

proc packColor(r, g, b, a: uint8): uint32 {.inline.} =
  ## Packs four 8-bit color channels into a single 32-bit integer.
  ## The byte order (AABBGGRR) is what Sokol's UBYTE4N format expects on little-endian systems
  ## to correctly map to an RGBA vec4 in the shader.
  result = (uint32(a) shl 24) or (uint32(b) shl 16) or (uint32(g) shl 8) or uint32(r)

proc loadPly(path: string): Mesh =
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

  let vbuf = sg.makeBuffer(BufferDesc(
    usage: BufferUsage(vertexBuffer: true),
    data: sg.Range(addr: out_vertices[0].addr, size: out_vertices.len * sizeof(Vertex))
  ))
  let ibuf = sg.makeBuffer(BufferDesc(
    usage: BufferUsage(indexBuffer: true),
    data: sg.Range(addr: out_indices[0].addr, size: out_indices.len * sizeof(uint16))
  ))
  result.indexCount = out_indices.len.int32
  result.bindings = Bindings(vertexBuffers: [vbuf], indexBuffer: ibuf)

type State = object
  pip: Pipeline
  passAction: sg.PassAction
  mesh: Mesh
  camTime: float32
  camPos: Vec3
  camYaw: float32
  camPitch: float32
  vsParams: VSParams
  fsParams: FSParams
  rx, ry: float32

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

  #sapp.lockMouse(true)
  #[
    Cube vertex buffer with packed vertex formats for color and texture coords.
    Note that a vertex format which must be portable across all
    backends must only use the normalized integer formats
    (BYTE4N, UBYTE4N, SHORT2N, SHORT4N), which can be converted
    to floating point formats in the vertex shader inputs.
  ]#
  #[
  const vertices = [
    Vertex( x: -1.0, y: -1.0, z: -1.0,  xN: 0.0, yN: 0.0, zN: -1.0,  color: 0xFF0000FF'u32, u:     0, v:     0 ),
    Vertex( x:  1.0, y: -1.0, z: -1.0,  xN: 0.0, yN: 0.0, zN: -1.0,  color: 0xFF0000FF'u32, u: 32767, v:     0 ),
    Vertex( x:  1.0, y:  1.0, z: -1.0,  xN: 0.0, yN: 0.0, zN: -1.0,  color: 0xFF0000FF'u32, u: 32767, v: 32767 ),
    Vertex( x: -1.0, y:  1.0, z: -1.0,  xN: 0.0, yN: 0.0, zN: -1.0,  color: 0xFF0000FF'u32, u:     0, v: 32767 ),
    Vertex( x: -1.0, y: -1.0, z:  1.0,  xN: 0.0, yN: 0.0, zN: 1.0,  color: 0xFF00FF00'u32, u:     0, v:     0 ),
    Vertex( x:  1.0, y: -1.0, z:  1.0,  xN: 0.0, yN: 0.0, zN: 1.0,  color: 0xFF00FF00'u32, u: 32767, v:     0 ),
    Vertex( x:  1.0, y:  1.0, z:  1.0,  xN: 0.0, yN: 0.0, zN: 1.0,  color: 0xFF00FF00'u32, u: 32767, v: 32767 ),
    Vertex( x: -1.0, y:  1.0, z:  1.0,  xN: 0.0, yN: 0.0, zN: 1.0,  color: 0xFF00FF00'u32, u:     0, v: 32767 ),
    Vertex( x: -1.0, y: -1.0, z: -1.0,  xN: -1.0, yN: 0.0, zN: 0.0,  color: 0xFFFF0000'u32, u:     0, v:     0 ),
    Vertex( x: -1.0, y:  1.0, z: -1.0,  xN: -1.0, yN: 0.0, zN: 0.0,  color: 0xFFFF0000'u32, u: 32767, v:     0 ),
    Vertex( x: -1.0, y:  1.0, z:  1.0,  xN: -1.0, yN: 0.0, zN: 0.0,  color: 0xFFFF0000'u32, u: 32767, v: 32767 ),
    Vertex( x: -1.0, y: -1.0, z:  1.0,  xN: -1.0, yN: 0.0, zN: 0.0,  color: 0xFFFF0000'u32, u:     0, v: 32767 ),
    Vertex( x:  1.0, y: -1.0, z: -1.0,  xN: 1.0, yN: 0.0, zN: 0.0,  color: 0xFFFF007F'u32, u:     0, v:     0 ),
    Vertex( x:  1.0, y:  1.0, z: -1.0,  xN: 1.0, yN: 0.0, zN: 0.0,  color: 0xFFFF007F'u32, u: 32767, v:     0 ),
    Vertex( x:  1.0, y:  1.0, z:  1.0,  xN: 1.0, yN: 0.0, zN: 0.0,  color: 0xFFFF007F'u32, u: 32767, v: 32767 ),
    Vertex( x:  1.0, y: -1.0, z:  1.0,  xN: 1.0, yN: 0.0, zN: 0.0,  color: 0xFFFF007F'u32, u:     0, v: 32767 ),
    Vertex( x: -1.0, y: -1.0, z: -1.0,  xN: 0.0, yN: -1.0, zN: 0.0,  color: 0xFFFF7F00'u32, u:     0, v:     0 ),
    Vertex( x: -1.0, y: -1.0, z:  1.0,  xN: 0.0, yN: -1.0, zN: 0.0,  color: 0xFFFF7F00'u32, u: 32767, v:     0 ),
    Vertex( x:  1.0, y: -1.0, z:  1.0,  xN: 0.0, yN: -1.0, zN: 0.0,  color: 0xFFFF7F00'u32, u: 32767, v: 32767 ),
    Vertex( x:  1.0, y: -1.0, z: -1.0,  xN: 0.0, yN: -1.0, zN: 0.0,  color: 0xFFFF7F00'u32, u:     0, v: 32767 ),
    Vertex( x: -1.0, y:  1.0, z: -1.0,  xN: 0.0, yN: 1.0, zN: 0.0,  color: 0xFF007FFF'u32, u:     0, v:     0 ),
    Vertex( x: -1.0, y:  1.0, z:  1.0,  xN: 0.0, yN: 1.0, zN: 0.0,  color: 0xFF007FFF'u32, u: 32767, v:     0 ),
    Vertex( x:  1.0, y:  1.0, z:  1.0,  xN: 0.0, yN: 1.0, zN: 0.0,  color: 0xFF007FFF'u32, u: 32767, v: 32767 ),
    Vertex( x:  1.0, y:  1.0, z: -1.0,  xN: 0.0, yN: 1.0, zN: 0.0,  color: 0xFF007FFF'u32, u:     0, v: 32767 ),
  ]
  let vbuf = sg.makeBuffer(BufferDesc(
    usage: BufferUsage(vertexBuffer: true),
    data: sg.Range(addr: vertices.addr, size: vertices.sizeof)
  ))

  # an index buffer
  const indices = [
    0'u16, 1, 2,  0, 2, 3,
    6, 5, 4,      7, 6, 4,
    8, 9, 10,     8, 10, 11,
    14, 13, 12,   15, 14, 12,
    16, 17, 18,   16, 18, 19,
    22, 21, 20,   23, 22, 20,
  ]
  let ibuf = sg.makeBuffer(BufferDesc(
    usage: BufferUsage(indexBuffer: true),
    data: sg.Range(addr: indices.addr, size: indices.sizeof)
  ))
  ]#

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
    qoiImage = readQOI("assets/diffuse.qoi")
    #qoiImage = readQOI("diffuse.qoi")
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
        #VertexAttrState(format: vertexFormatShort2n), # texcoord0
        VertexAttrState(format: vertexFormatFloat2), # texcoord0
      ],
    ),
    indexType: indexTypeUint16,
    #faceWinding: faceWindingCcw,
    cullMode: cullModeBack,
    depth: DepthState(
      compare: compareFuncLessEqual,
      writeEnabled: true,
    )
  ))
  # save everything in bindings
  var mesh: Mesh
  #mesh.bindings = Bindings(vertexBuffers: [vbuf], indexbuffer: ibuf)
  #mesh.bindings.images[shd.imgUTexture] = texcubeImg
  #mesh.bindings.samplers[shd.smpUSampler] = texcubeSmp
  #mesh.indexCount = indices.sizeof
  #state.mesh = mesh

  let assetDir = getAppDir() & DirSep
  #let modelPath = assetDir & "bs_rest.obj"

  #let modelPath = assetDir & "teapot.ply"
  let modelPath = assetDir & "bs_rest.ply"
  mesh = loadPly(modelPath)
  #let modelPath = assetDir & "bs_rest.obj"
  #mesh = loadObj(modelPath)
  #mesh.bindings.images[shd.imgUTexture] = texcubeImg
  mesh.bindings.images[shd.imgUTexture] = qoiTexture
  mesh.bindings.samplers[shd.smpUSampler] = texcubeSmp
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
    #u_jitterAmount: 240.0, # Simulate a 240p vertical resolution
    u_jitterAmount: 480.0, # Simulate a 240p vertical resolution
  )

proc computeFsParams(): shd.FsParams =
  result = shd.FsParams(
    u_fogColor: vec3(0.25f, 0.5f, 0.75f),
    u_fogNear: 4.0f,
    u_fogFar: 12.0f,
    #u_ditherSize: vec2(800.0, 600.0)
    u_ditherSize: vec2(sapp.widthf(), sapp.heightf()),
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
