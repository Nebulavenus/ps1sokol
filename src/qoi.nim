##
## .. image:: qoi_logo.png
##
## =============================================================
## **The Quite OK Image Format for Fast, Lossless Compression**
## =============================================================
##
## **About**
##
## QOI encodes and decodes images in a lossless format. Compared to stb_image and
## stb_image_write QOI offers 20x-50x faster encoding, 3x-4x faster decoding and
## 20% better compression. The `QOI specification document<https://qoiformat.org/qoi-specification.pdf>`_
## outlines the data format.
##
## **Available Functions**
##  * readQOI    -- read and decode a QOI file
##  * decodeQOI  -- decode the raw bytes of a QOI image from memory
##  * writeQOI   -- encode and write a QOI file
##  * encodeQOI  -- encode an rgba buffer into a QOI image in memory
##
## **QOI Format Specification**
##
## A QOI file has a 14 byte `header<#Header>`_, followed by any number of data "chunks" and an
## 8-byte `end<#QOI_END>`_ marker. Images are encoded row by row, left to right, top to bottom.
## The decoder and encoder start with ``{r:0, g:0, b:0, a:255}`` as the previous pixel value.
## An image is complete when all pixels specified by ``width * height`` have been covered.
##
## Pixels are encoded as:
##  * a run of the previous pixel
##  * an index into an array of previously seen pixels
##  * a difference to the previous pixel value in r,g,b
##  * full r,g,b or r,g,b,a values
##
## The color channels are assumed to not be premultiplied with the alpha channel
## ("un-premultiplied alpha").
##
## A running ``array[64]`` (zero-initialized) of previously seen pixel values is
## maintained by the encoder and decoder. Each pixel that is seen by the encoder
## and decoder is put into this array at the position formed by a hash function of
## the color value. In the encoder, if the pixel value at the index matches the
## current pixel, this index position is written to the stream as QOI_OP_INDEX.
## The hash function for the index is:
##
## .. code-block::
## 	 index_position = (r * 3 + g * 5 + b * 7 + a * 11) % 64
##
## Each chunk starts with a 2-bit or 8-bit tag, followed by a number of data bits. The
## bit length of chunks is divisible by 8 - i.e. all chunks are byte aligned. All
## values encoded in these data bits have the most significant bit on the left.
##
## The 8-bit tags have precedence over the 2-bit tags. A decoder must check for the
## presence of an 8-bit tag first.
##
## The byte stream's end is marked with 7 0x00 bytes followed a single 0x01 byte.
##
## The possible chunks are:
##
## ``QOI_OP_INDEX``:
##  * ``Byte[0]``
##
##    - ``bits[6..7]``  2-bit tag b00
##    - ``bits[0..5]``  6-bit index into the color index array: 0..63
##
##  * A valid encoder must not issue 2 or more consecutive QOI_OP_INDEX chunks to the
##    same index. QOI_OP_RUN should be used instead.
##
## ``QOI_OP_DIFF``:
##  * ``Byte[0]``
##
##    - ``bits[6..7]``  2-bit tag b01
##    - ``bits[4..5]``  red channel difference from the previous pixel between -2..1
##    - ``bits[2..3]``  green channel difference from the previous pixel between -2..1
##    - ``bits[0..1]``  blue channel difference from the previous pixel between -2..1
##
##  - The difference to the current channel values are using a wraparound operation,
##    so "1 - 2" will result in 255, while "255 + 1" will result in 0.
##
##  - Values are stored as unsigned integers with a bias of 2. E.g. -2 is stored as
##    0 (b00). 1 is stored as 3 (b11).
##
##  - The alpha value remains unchanged from the previous pixel.
#
##
## ``QOI_OP_LUMA``:
##  * ``Byte[0]``
##
##    - ``bits[6..7]``  2-bit tag b10
##    - ``bits[0..5]``  6-bit green channel difference from the previous pixel -32..31
##
##  * ``Byte[1]``
##
##    - ``bits[4..7]``  4-bit red channel difference minus green channel difference -8..7
##    - ``bits[0..3]``  4-bit blue channel difference minus green channel difference -8..7
##
##  * The green channel is used to indicate the general direction of change and is
##    encoded in 6 bits. The red and blue channels (dr and db) base their diffs off
##    of the green channel difference and are encoded in 4 bits. I.e.:
##
##    - ``dr_dg = (cur_px.r - prev_px.r) - (cur_px.g - prev_px.g)``
##    - ``db_dg = (cur_px.b - prev_px.b) - (cur_px.g - prev_px.g)``
##
##  * The difference to the current channel values are using a wraparound operation,
##    so "10 - 13" will result in 253, while "250 + 7" will result in 1.
##
##  * Values are stored as unsigned integers with a bias of 32 for the green channel
##    and a bias of 8 for the red and blue channel.
##
##  * The alpha value remains unchanged from the previous pixel.
##
## ``QOI_OP_RUN``:
##  * ``Byte[0]``
##
##    - ``bits[6..7]``  2-bit tag b11
##    - ``bits[0..5]``  6-bit run-length repeating the previous pixel: 1..62
##
##  * The run-length is stored with a bias of -1. Note that the run-lengths 63 and 64
##    (b111110 and b111111) are illegal as they are occupied by the QOI_OP_RGB and
##    QOI_OP_RGBA tags.
##
## ``QOI_OP_RGB``:
##  * ``Byte[0]``
##
##    - ``bits[0..7]``  8-bit tag b11111110
##
##  * ``Byte[1]``
##    - ``bits[0..7]``  8-bit red channel value
##
##  * ``Byte[2]``
##    - ``bits[0..7]``  8-bit green channel value
##
##  * ``Byte[3]``
##    - ``bits[0..7]``  8-bit blue channel value
##
##  * The alpha value remains unchanged from the previous pixel.
##
##
## ``QOI_OP_RGBA``:
##  * ``Byte[0]``
##
##    - ``bits[0..7]``  8-bit tag b11111111
##
##  * ``Byte[1]``
##    - ``bits[0..7]``  8-bit red channel value
##
##  * ``Byte[2]``
##    - ``bits[0..7]``  8-bit green channel value
##
##  * ``Byte[3]``
##    - ``bits[0..7]``  8-bit blue channel value
##
##  * ``Byte[4]``
##    - ``bits[0..7]``  8-bit alpha channel value
##
##
##

import std/streams
import std/strutils

type
  Channels* = enum
    ## The channel byte in the QOI file header is an enum where:
    ##  * 3 = RGB
    ##  * 4 = RGBA
    RGB = 3, RGBA = 4

  Colorspace* = enum
    ## The colorspace byte in the QOI file header is an enum where:
    ##	* 0 = sRGB, i.e. gamma scaled RGB channels and a linear alpha channel
    ##	* 1 = all channels are linear
    ## The colorspace is purely
    ## informative. It will be saved to the file header, but does not affect
    ## how chunks are encoded/decoded.
    sRGB = 0, linear = 1

  Header* = object
    ## A QOI file has a 14 byte header, whose fields are defined as follows:
    ##  * magic       [char, 4] - magic bytes "qoif" (not stored in the data object but checked
    ##    when decoding a data stream)
    ##  * width       [uint32]  - image width in pixels (BE)
    ##  * height      [uint32]  - image height in pixels (BE)
    ##  * channels    [byte]    - 3 = RGB, 4 = RGBA
    ##  * colorspace  [byte]    - 0 = sRGB with linear alpha, 1 = all channels linear
    width*: uint32
    height*: uint32
    channels*: Channels
    colorspace*: Colorspace

  Pixel = object
    ## The QOI encoder/decoder tends to work with RGBA data, where the A channel is ignored for RGB images.
    r, g, b, a: byte

  QOIF* = object
    ## The object that stores QOI image data for use in programs.
    header*: Header
    data*: seq[byte]

  ByteStream = object
    ## A simple stream wrapper around a seq[byte] to handle reading and writing.
    data: seq[byte]
    pos: int

const
  QOI_MAGIC* = "qoif"

  QOI_2BIT_MASK       = 0b11000000.byte
  QOI_RUN_TAG_MASK    = 0b11000000.byte
  QOI_INDEX_TAG_MASK  = 0b00000000.byte
  QOI_DIFF_TAG_MASK   = 0b01000000.byte
  QOI_LUMA_TAG_MASK   = 0b10000000.byte

  QOI_RGB_TAG         = 0b11111110.byte
  QOI_RGBA_TAG        = 0b11111111.byte

  QOI_RUN_VAL_MASK    = 0b00111111.byte
  QOI_INDEX_VAL_MASK  = 0b00111111.byte

  QOI_2BIT_LOWER_MASK = 0b00000011.byte
  QOI_LUMA_DG_MASK    = 0b00111111.byte
  QOI_4BIT_LOWER_MASK = 0b00001111.byte

  QOI_END* = [0.byte, 0, 0, 0, 0, 0, 0, 1]

# -----------------------------------------------------------------
#                   HELPER FUNCTIONS
# -----------------------------------------------------------------

# ByteStream helper procedures
proc newByteStream(data: seq[byte] = @[]): ByteStream =
  ## Creates a new ByteStream, optionally from an existing seq of bytes.
  result.data = data
  result.pos = 0

proc readByte(stream: var ByteStream): byte =
  ## Reads a single byte from the stream and advances the position.
  if stream.pos >= stream.data.len:
    raise newException(IOError, "Attempted to read past the end of the stream.")
  result = stream.data[stream.pos]
  inc stream.pos

proc writeByte(stream: var ByteStream, b: byte) =
  ## Writes a single byte to the stream's data sequence.
  stream.data.add(b)

proc readUint32(stream: var ByteStream): uint32 =
  ## Reads a 32-bit big-endian unsigned integer from the stream.
  let b1 = stream.readByte()
  let b2 = stream.readByte()
  let b3 = stream.readByte()
  let b4 = stream.readByte()
  result = (uint32(b1) shl 24) or (uint32(b2) shl 16) or (uint32(b3) shl 8) or uint32(b4)

proc writeUint32(stream: var ByteStream, val: uint32) =
  ## Writes a 32-bit unsigned integer in big-endian format to the stream.
  stream.writeByte(byte(val shr 24))
  stream.writeByte(byte(val shr 16))
  stream.writeByte(byte(val shr 8))
  stream.writeByte(byte(val))

func init*(width, height: uint32; channels: Channels, colorspace: Colorspace): Header =
  ## Creates a new Header object using the provided variables
  result.width = width
  result.height = height
  result.channels = channels
  result.colorspace = colorspace

func hash(pixel: Pixel): byte =
  ## The hash function as defined by the QOI spec for the seen pixel array.
  return (pixel.r * 3 + pixel.g * 5 + pixel.b * 7 + pixel.a * 11) mod 64

func `==`(p1, p2: Pixel): bool =
  ## Checks if two pixels are identical.
  p1.r == p2.r and p1.g == p2.g and p1.b == p2.b and p1.a == p2.a

func `-`(p1, p2: Pixel): Pixel =
  ## Returns a pixel representing the wrapping difference between two pixels.
  result.r = p1.r - p2.r
  result.g = p1.g - p2.g
  result.b = p1.b - p2.b
  result.a = p1.a - p2.a

proc getPixel(stream: var ByteStream, channels: Channels): Pixel =
  ## Reads 3 or 4 bytes from the raw pixel stream.
  result.r = stream.readByte()
  result.g = stream.readByte()
  result.b = stream.readByte()
  result.a = if channels == RGBA: stream.readByte() else: 255

proc opRun(output: var ByteStream, runs: var byte, index, lastPixel: int) =
  ## Writes a QOI_OP_RUN chunk if conditions are met.
  inc runs
  if (runs == 62) or (index == lastPixel):
    output.writeByte(QOI_RUN_TAG_MASK or (runs - 1).byte)
    runs = 0

proc opDiff(output: var ByteStream, diff: Pixel, flag: var bool) =
  ## Writes a QOI_OP_DIFF or QOI_OP_LUMA chunk to the output stream.
  flag = true
  if (diff.r.int in 254..1) and (diff.g.int in 254..1) and (diff.b.int in 254..1): # -2..1
    output.writeByte(QOI_DIFF_TAG_MASK or ((diff.r + 2).byte shl 4) or ((diff.g + 2).byte shl 2) or (diff.b + 2).byte)

  elif (diff.g.int in 224..31) and # -32..31
        ((diff.r - diff.g).int in 248..7) and # -8..7
        ((diff.b - diff.g).int in 248..7): # -8..7
    output.writeByte(QOI_LUMA_TAG_MASK or (diff.g + 32).byte)
    output.writeByte(((diff.r - diff.g + 8).byte shl 4) or (diff.b - diff.g + 8).byte)
  else:
    flag = false

proc opRGB(output: var ByteStream, pixel: Pixel) =
  ## Writes a 4-byte QOI_OP_RGB chunk to the output stream.
  output.writeByte(QOI_RGB_TAG)
  output.writeByte(pixel.r)
  output.writeByte(pixel.g)
  output.writeByte(pixel.b)

proc opRGBA(output: var ByteStream, pixel: Pixel) =
  ## Writes a 5-byte QOI_OP_RGBA chunk to the output stream.
  output.writeByte(QOI_RGBA_TAG)
  output.writeByte(pixel.r)
  output.writeByte(pixel.g)
  output.writeByte(pixel.b)
  output.writeByte(pixel.a)

proc writeHeader(output: var ByteStream; header: Header) =
  ## Writes the 14-byte QOI file header to the output stream.
  for c in QOI_MAGIC:
    output.writeByte(c.byte)

  output.writeUint32(header.width)
  output.writeUint32(header.height)
  output.writeByte(uint8(header.channels))
  output.writeByte(uint8(header.colorspace))

proc writeData(output: var ByteStream; input: var ByteStream; hdr: Header) =
  ## Reads raw pixel data from the input stream and writes compressed QOI chunks to the output stream.
  let
    imageSize = hdr.width.int * hdr.height.int
    pixelSize = hdr.channels.int
    lastPixelIndex = imageSize - 1

  var
    prevPixel = Pixel(r: 0, g: 0, b: 0, a: 255)
    runs = 0.byte
    seenWindow: array[64, Pixel]

  for i in 0 ..< imageSize:
    let currPixel = input.getPixel(hdr.channels)

    if currPixel == prevPixel:
      opRun(output, runs, i, lastPixelIndex)
    else:
      if runs > 0:
        output.writeByte(QOI_RUN_TAG_MASK or (runs - 1).byte)
        runs = 0

      let indexPos = currPixel.hash()
      if seenWindow[indexPos] == currPixel:
        output.writeByte(QOI_INDEX_TAG_MASK or indexPos)
      else:
        seenWindow[indexPos] = currPixel
        let diffPixel = currPixel - prevPixel

        if diffPixel.a == 0:
          var diffWritten: bool
          opDiff(output, diffPixel, diffWritten)
          if not diffWritten:
            opRGB(output, currPixel)
        else:
          opRGBA(output, currPixel)

    prevPixel = currPixel

proc readSignature(stream: var ByteStream): bool =
  ## Reads 4 bytes and checks if they match the QOI magic string "qoif".
  for c in QOI_MAGIC:
    if stream.readByte() != c.byte: return false
  return true

proc readData(stream: var ByteStream, hdr: Header): seq[byte] =
  ## Decodes QOI data chunks from the input stream into a raw pixel data sequence.
  let pixelCount = hdr.width.int * hdr.height.int
  var
    pixel = Pixel(r: 0, g: 0, b: 0, a: 255)
    run = 0
    seenWindow: array[64, Pixel]

  result = newSeq[byte](pixelCount * hdr.channels.int)
  var p = 0 # pointer into result seq

  for i in 0 ..< pixelCount:
    if run > 0:
      dec run
    else:
      let b1 = stream.readByte()
      if b1 == QOI_RGB_TAG:
        pixel.r = stream.readByte()
        pixel.g = stream.readByte()
        pixel.b = stream.readByte()
      elif b1 == QOI_RGBA_TAG:
        pixel.r = stream.readByte()
        pixel.g = stream.readByte()
        pixel.b = stream.readByte()
        pixel.a = stream.readByte()
      elif (b1 and QOI_2BIT_MASK) == QOI_INDEX_TAG_MASK:
        pixel = seenWindow[b1]
      elif (b1 and QOI_2BIT_MASK) == QOI_DIFF_TAG_MASK:
        pixel.r += (b1 shr 4 and QOI_2BIT_LOWER_MASK) - 2
        pixel.g += (b1 shr 2 and QOI_2BIT_LOWER_MASK) - 2
        pixel.b += (b1 and QOI_2BIT_LOWER_MASK) - 2
      elif (b1 and QOI_2BIT_MASK) == QOI_LUMA_TAG_MASK:
        let dg = (b1 and QOI_LUMA_DG_MASK) - 32
        let b2 = stream.readByte()
        let drdg = (b2 shr 4 and QOI_4BIT_LOWER_MASK) - 8
        let dbdg = (b2 and QOI_4BIT_LOWER_MASK) - 8
        pixel.r += dg + drdg
        pixel.g += dg
        pixel.b += dg + dbdg
      elif (b1 and QOI_2BIT_MASK) == QOI_RUN_TAG_MASK:
        run = (b1 and QOI_RUN_VAL_MASK).int

      seenWindow[pixel.hash()] = pixel

    result[p] = pixel.r; inc p
    result[p] = pixel.g; inc p
    result[p] = pixel.b; inc p
    if hdr.channels == RGBA:
      result[p] = pixel.a; inc p

# -----------------------------------------------------------------
#                   MAIN PUBLIC API
# -----------------------------------------------------------------

proc decodeQOI*(qoiData: seq[byte]): QOIF =
  ## Decode a QOI image from a sequence of bytes.
  var stream = newByteStream(qoiData)

  if not readSignature(stream):
    raise newException(ValueError, "Invalid QOI file signature.")

  let header = init(
    width = stream.readUint32(),
    height = stream.readUint32(),
    channels = cast[Channels](stream.readByte()),
    colorspace = cast[Colorspace](stream.readByte())
  )
  result.header = header
  result.data = readData(stream, header)

  let endMarker = qoiData[^8 .. ^1]
  if endMarker != QOI_END:
    raise newException(ValueError, "QOI file is truncated or has an invalid end marker.")

proc encodeQOI*(header: Header, pixelData: seq[byte]): seq[byte] =
  ## Encode raw RGB/RGBA pixels into a QOI image byte sequence.
  var output = newByteStream()
  var input = newByteStream(pixelData)

  writeHeader(output, header)
  writeData(output, input, header)

  for b in QOI_END:
    output.writeByte(b)

  return output.data

proc readQOI*(filename: string): QOIF =
  ## Read and decode a QOI image from the file system.
  let fileStream = newFileStream(filename, fmRead)
  if fileStream == nil:
    raise newException(IOError, "Cannot open file: " & filename)
  defer: fileStream.close()

  #let qoiData = fileStream.readAll().toSeqByte()
  let qoiData = cast[seq[byte]](fileStream.readAll())
  return decodeQOI(qoiData)

proc writeQOI*(filename: string, header: Header, pixelData: seq[byte]) =
  ## Encode raw pixel data and write it to a QOI file on the file system.
  let qoiData = encodeQOI(header, pixelData)
  #writeFile(filename, qoiData.mapIt(it.char).join)
  writeFile(filename, cast[string](qoiData))
