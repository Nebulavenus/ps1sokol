# QOA - The "Quite OK Audio" format for fast, lossy audio compression
#
# Copyright (c) 2023, Dominic Szablewski - https://phoboslab.org
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2023-2024 Piotr Fusik
# https://github.com/pfusik/qoa-fu/blob/master/transpiled/QOA.swift
#
# Nim port of QOA by AI based on the original C code
# and transpiled version of Swift with Fusion lang.
#
import std/math

# -----------------------------------------------------------------------------
# LMS Filter (Least Mean Squares)
# -----------------------------------------------------------------------------

type
  LMS* = ref object
    history*: array[4, int32]
    weights*: array[4, int32]

proc newLMS(): LMS =
  ## Creates a new LMS filter state. Arrays are zero-initialized by default.
  new(result)

proc assign*(self: LMS, source: LMS) =
  ## Copies the state from another LMS instance.
  self.history = source.history
  self.weights = source.weights

proc predict*(self: LMS): int =
  ## Predicts the next sample.
  var p: int64 = 0
  for i in 0..3:
    p += int64(self.history[i]) * int64(self.weights[i])
  return int(p shr 13)

proc update*(self: LMS, sample: int, residual: int) =
  ## Updates the filter weights and history.
  let delta = residual shr 4
  self.weights[0] += int32(if self.history[0] < 0: -delta else: delta)
  self.weights[1] += int32(if self.history[1] < 0: -delta else: delta)
  self.weights[2] += int32(if self.history[2] < 0: -delta else: delta)
  self.weights[3] += int32(if self.history[3] < 0: -delta else: delta)

  self.history[0] = self.history[1]
  self.history[1] = self.history[2]
  self.history[2] = self.history[3]
  self.history[3] = int32(sample)

# -----------------------------------------------------------------------------
# QOA Base (Common constants and helpers)
# -----------------------------------------------------------------------------

type
  QOABase* = ref object of RootObj
    frameHeader*: int

const
  qoaMaxChannels* = 8
  qoaSliceSamples* = 20
  qoaMaxFrameSlices* = 256
  qoaMaxFrameSamples* = qoaSliceSamples * qoaMaxFrameSlices # 5120

proc getChannels*(self: QOABase): int =
  ## Returns the number of audio channels.
  return self.frameHeader shr 24

proc getSampleRate*(self: QOABase): int =
  ## Returns the sample rate in Hz.
  return self.frameHeader and 0xFFFFFF # 16777215

proc getFrameBytes*(self: QOABase, sampleCount: int): int =
  let slices = (sampleCount + 19) div 20
  return 8 + self.getChannels() * (16 + slices * 8)

const
  qoaScaleFactors*: array[16, int16] = [
    1, 7, 21, 45, 84, 138, 211, 304, 421, 562, 731, 928, 1157, 1419, 1715, 2048
  ]

proc dequantize*(quantized: int, scaleFactor: int): int =
  var dequantized: int
  case quantized shr 1
  of 0: dequantized = (scaleFactor * 3 + 2) shr 2
  of 1: dequantized = (scaleFactor * 5 + 1) shr 1
  of 2: dequantized = (scaleFactor * 9 + 1) shr 1
  else: dequantized = scaleFactor * 7

  if (quantized and 1) != 0:
    return -dequantized
  else:
    return dequantized

# -----------------------------------------------------------------------------
# QOA Encoder
# -----------------------------------------------------------------------------

type
  QOAEncoder* = ref object of QOABase
    writeLongProc*: proc(l: int64): bool {.closure.}
    lmses: array[qoaMaxChannels, LMS]

proc newQOAEncoder*(writeLongProc: proc(l: int64): bool): QOAEncoder =
  ## Creates a new QOA encoder.
  new(result)
  result.writeLongProc = writeLongProc
  for i in 0 ..< qoaMaxChannels:
    result.lmses[i] = newLMS()

proc writeLong(self: QOAEncoder, l: int64): bool =
  if self.writeLongProc.isNil:
    raise newException(AssertionDefect, "Abstract method called: writeLong")
  return self.writeLongProc(l)

proc writeHeader*(self: QOAEncoder, totalSamples: int, channels: int, sampleRate: int): bool =
  ## Writes the file header. Returns `true` on success.
  if totalSamples <= 0 or channels <= 0 or channels > qoaMaxChannels or sampleRate <= 0 or sampleRate >= 0x1000000:
    return false

  self.frameHeader = (channels shl 24) or sampleRate
  for c in 0 ..< channels:
    self.lmses[c].weights[0] = 0
    self.lmses[c].weights[1] = 0
    self.lmses[c].weights[2] = -8192
    self.lmses[c].weights[3] = 16384

  const magic = 1903124838'i64 # "qoaf"
  return self.writeLong((magic shl 32) or int64(totalSamples))

proc writeLMS(self: QOAEncoder, a: openarray[int32]): bool =
  let a0 = int64(a[0])
  let a1 = int64(a[1])
  let a2 = int64(a[2])
  let a3 = int64(a[3])
  let val = (a0 shl 48) or ((a1 and 0xFFFF) shl 32) or ((a2 and 0xFFFF) shl 16) or (a3 and 0xFFFF)
  return self.writeLong(val)

const
  qoaWriteFrameReciprocals: array[16, int32] = [
    65536, 9363, 3121, 1457, 781, 475, 311, 216, 156, 117, 90, 71, 57, 47, 39, 32
  ]
  qoaWriteFrameQuantTab: array[17, uint8] = [
    7, 7, 7, 5, 5, 3, 3, 1, 0, 0, 2, 2, 4, 4, 6, 6, 6
  ]

proc writeFrame*(self: QOAEncoder, samples: openarray[int16], samplesCount: int): bool =
  ## Encodes and writes a frame.
  if samplesCount <= 0 or samplesCount > qoaMaxFrameSamples:
    return false

  let header = int64(self.frameHeader)
  if not self.writeLong((header shl 32) or int64(samplesCount shl 16) or int64(self.getFrameBytes(samplesCount))):
    return false

  let channels = self.getChannels()
  for c in 0 ..< channels:
    if not self.writeLMS(self.lmses[c].history) or not self.writeLMS(self.lmses[c].weights):
      return false

  let lms = newLMS()
  let bestLMS = newLMS()
  var lastScaleFactors = newSeq[uint8](qoaMaxChannels)

  for sampleIndex in countup(0, samplesCount - 1, qoaSliceSamples):
    let sliceSamples = min(samplesCount - sampleIndex, qoaSliceSamples)
    for c in 0 ..< channels:
      var bestRank: int64 = high(int64)
      var bestSlice: int64 = 0

      for scaleFactorDelta in 0 ..< 16:
        let scaleFactor = (int(lastScaleFactors[c]) + scaleFactorDelta) and 15
        lms.assign(self.lmses[c])
        let reciprocal = int(qoaWriteFrameReciprocals[scaleFactor])
        var slice: int64 = int64(scaleFactor)
        var currentRank: int64 = 0

        var s = 0
        while s < sliceSamples:
          let sample = int(samples[(sampleIndex + s) * channels + c])
          let predicted = lms.predict()
          let residual = sample - predicted

          var scaled = (residual * reciprocal + 32768) shr 16
          if scaled != 0:
            if scaled < 0:
              scaled += 1
            else:
              scaled += -1
          if residual != 0:
            if residual > 0:
              scaled += 1
            else:
              scaled += -1

          let quantized = int(qoaWriteFrameQuantTab[8 + clamp(scaled, -8, 8)])
          let dequantized = dequantize(quantized, int(qoaScaleFactors[scaleFactor]))
          let reconstructed = clamp(predicted + dequantized, -32768, 32767)

          let error = int64(sample - reconstructed)
          currentRank += error * error

          var weightsPenalty: int = 0
          for i in 0..3: weightsPenalty += int(lms.weights[i]) * int(lms.weights[i])
          weightsPenalty = (weightsPenalty shr 18) - 2303

          if weightsPenalty > 0: currentRank += int64(weightsPenalty)
          if currentRank >= bestRank: break

          lms.update(reconstructed, dequantized)
          slice = (slice shl 3) or int64(quantized)
          inc s

        if currentRank < bestRank:
          bestRank = currentRank
          bestSlice = slice
          bestLMS.assign(lms)

      self.lmses[c].assign(bestLMS)
      bestSlice = bestSlice shl int64((20 - sliceSamples) * 3)
      lastScaleFactors[c] = uint8(bestSlice shr 60)
      if not self.writeLong(bestSlice):
        return false
  return true

# -----------------------------------------------------------------------------
# QOA Decoder
# -----------------------------------------------------------------------------

type
  QOADecoder* = ref object of QOABase
    readByteProc*: proc(): int {.closure.}
    seekToByteProc*: proc(position: int) {.closure.}

    buffer: int
    bufferBits: int
    totalSamples: int
    positionSamples: int

proc newQOADecoder*(readByteProc: proc(): int, seekToByteProc: proc(pos: int)): QOADecoder =
  ## Creates a new QOA decoder.
  new(result)
  result.readByteProc = readByteProc
  result.seekToByteProc = seekToByteProc

proc readByte(self: QOADecoder): int =
  if self.readByteProc.isNil:
    raise newException(AssertionDefect, "Abstract method called: readByte")
  return self.readByteProc()

proc seekToByte*(self: QOADecoder, pos: int) =
  if self.seekToByteProc.isNil:
    raise newException(AssertionDefect, "Abstract method called: seekToByte")
  self.seekToByteProc(pos)
  self.buffer = 0
  self.bufferBits = 0

proc readBits(self: QOADecoder, bits: int): int =
  while self.bufferBits < bits:
    let b = self.readByte()
    if b < 0: return -1
    self.buffer = (self.buffer shl 8) or b
    self.bufferBits += 8

  self.bufferBits -= bits
  result = self.buffer shr self.bufferBits
  self.buffer = self.buffer and ((1 shl self.bufferBits) - 1)
  return result

proc readHeader*(self: QOADecoder): bool =
  ## Reads the file header. Returns `true` if the header is valid.
  if self.readByte() != 'q'.int or self.readByte() != 'o'.int or self.readByte() != 'a'.int or self.readByte() != 'f'.int:
    return false

  self.buffer = 0
  self.bufferBits = 0

  var temp = self.readBits(32)
  if temp < 0: return false
  self.totalSamples = temp

  if self.totalSamples <= 0: return false

  temp = self.readBits(32)
  if temp < 0: return false
  self.frameHeader = temp

  if self.frameHeader <= 0: return false

  self.positionSamples = 0
  let channels = self.getChannels()
  return channels > 0 and channels <= qoaMaxChannels and self.getSampleRate() > 0

proc getTotalSamples*(self: QOADecoder): int =
  ## Returns the file length in samples per channel.
  self.totalSamples

proc getMaxFrameBytes(self: QOADecoder): int =
  8 + self.getChannels() * (16 + qoaMaxFrameSlices * 8)

proc readLMS(self: QOADecoder, result1: var openarray[int32]): bool =
  for i in 0 ..< 4:
    let hi = self.readByte()
    if hi < 0: return false
    let lo = self.readByte()
    if lo < 0: return false
    #result1[i] = int32(int16(hi shl 8 or lo)) # Reconstruct signed 16-bit big-endian value
    # Combine bytes into a 16-bit unsigned word first
    let combined_word = uint16((hi shl 8) or lo)
    # Then, 'cast' the bits to a signed int16, and then to int32 for storage
    result1[i] = int32(cast[int16](combined_word)) # <-- This is the fix
  return true

proc readFrame*(self: QOADecoder, samples: var openarray[int16]): int =
  ## Reads and decodes a frame. Returns the number of samples per channel.
  if self.positionSamples > 0:
    let header = self.readBits(32)
    if header == -1 or header != self.frameHeader: return -1

  let samplesCount = self.readBits(16)
  if samplesCount <= 0 or samplesCount > qoaMaxFrameSamples or samplesCount > self.totalSamples - self.positionSamples:
    return -1

  let channels = self.getChannels()
  let slices = (samplesCount + 19) div 20

  let frameSize = self.readBits(16)
  if frameSize == -1 or frameSize != 8 + channels * (16 + slices * 8):
    return -1

  var lmses: array[qoaMaxChannels, LMS]
  for i in 0 ..< qoaMaxChannels: lmses[i] = newLMS()

  for c in 0 ..< channels:
    if not self.readLMS(lmses[c].history) or not self.readLMS(lmses[c].weights):
      return -1

  for sampleIndex in countup(0, samplesCount - 1, qoaSliceSamples):
    for c in 0 ..< channels:
      var scaleFactorVal = self.readBits(4)
      if scaleFactorVal < 0: return -1

      let scaleFactor = int(qoaScaleFactors[scaleFactorVal])
      var sampleOffset = sampleIndex * channels + c

      for s in 0 ..< qoaSliceSamples:
        let quantized = self.readBits(3)
        if quantized < 0: return -1

        if sampleIndex + s >= samplesCount: continue

        let dequantized = dequantize(quantized, scaleFactor)
        let reconstructed = clamp(lmses[c].predict() + dequantized, -32768, 32767)
        lmses[c].update(reconstructed, dequantized)
        samples[sampleOffset] = int16(reconstructed)
        sampleOffset += channels

  self.positionSamples += samplesCount
  return samplesCount

proc seekToSample*(self: QOADecoder, position: int) =
  ## Seeks to the given time offset.
  let frame = position div qoaMaxFrameSamples
  self.seekToByte(if frame == 0: 12 else: 8 + frame * self.getMaxFrameBytes())
  self.positionSamples = frame * qoaMaxFrameSamples

proc isEnd*(self: QOADecoder): bool =
  ## Returns `true` if all frames have been read.
  return self.positionSamples >= self.totalSamples
