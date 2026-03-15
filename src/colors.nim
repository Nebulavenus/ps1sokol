proc packColor*(r, g, b, a: uint8): uint32 {.inline.} =
  ## Packs four 8-bit color channels into a single 32-bit integer.
  ## The byte order (AABBGGRR) is what Sokol's UBYTE4N format expects on little-endian systems
  ## to correctly map to an RGBA vec4 in the shader.
  result = (uint32(a) shl 24) or (uint32(b) shl 16) or (uint32(g) shl 8) or uint32(r)

proc unpackColor*(c: uint32): (uint8, uint8, uint8, uint8) {.inline.} =
  ## Unpacks a 32-bit AABBGGRR color into four 8-bit channels
  let r = (c and 0x000000FF'u32).uint8
  let g = ((c and 0x0000FF00'u32) shr 8).uint8
  let b = ((c and 0x00FF0000'u32) shr 16).uint8
  let a = ((c and 0xFF000000'u32) shr 24).uint8
  return (r, g, b, a)
