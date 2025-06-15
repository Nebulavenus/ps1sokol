# tests/test_qoa.nim

import unittest
import os
import math

# Assuming your QOA implementation is in the parent directory
# This allows the test to find the qoa module
#[
when defined(windows):
  const ParentDir = "..\\"
else:
  const ParentDir = "../"
addPath(ParentDir)
]#

import qoa

# -----------------------------------------------------------------------------
# Test Configuration
# -----------------------------------------------------------------------------

const
  testSampleRate = 44100
  testChannels = 2 # Stereo
  testTotalSamples = 6000 # A little more than one max frame

# -----------------------------------------------------------------------------
# In-Memory Stream for Testing
# -----------------------------------------------------------------------------

var
  qoaData: seq[byte]
  streamPos: int

proc resetStream() =
  qoaData = @[]
  streamPos = 0

proc testWriteLong(l: int64): bool =
  ## Big-endian write to our in-memory byte sequence.
  for i in countdown(7, 0):
    qoaData.add(byte((l shr (i * 8)) and 0xFF))
  return true

proc testReadByte(): int =
  ## Read from our in-memory byte sequence.
  if streamPos >= qoaData.len:
    return -1 # EOF
  result = int(qoaData[streamPos])
  inc streamPos

proc testSeekToByte(pos: int) =
  ## Seek in our in-memory byte sequence.
  if pos >= 0 and pos < qoaData.len:
    streamPos = pos
  elif pos >= qoaData.len:
    streamPos = qoaData.len
  else:
    streamPos = 0

# -----------------------------------------------------------------------------
# Test Data Generation
# -----------------------------------------------------------------------------

proc generateTestPcm(numSamples, channels, sampleRate: int): seq[int16] =
  ## Generates a stereo sine wave PCM sample buffer.
  result = newSeq[int16](numSamples * channels)
  let freq1 = 440.0 # A4
  let freq2 = 659.25 # E5
  let amplitude = 32000.0

  for i in 0 ..< numSamples:
    let time = float(i) / float(sampleRate)

    # Channel 1
    let val1 = sin(2 * PI * freq1 * time) * amplitude
    result[i * channels + 0] = int16(val1)

    # Channel 2
    if channels > 1:
      let val2 = sin(2 * PI * freq2 * time) * amplitude
      result[i * channels + 1] = int16(val2)

# -----------------------------------------------------------------------------
# Unit Test Suite
# -----------------------------------------------------------------------------

suite "QOA Audio Codec Tests":
  test "Encoder/Decoder Round-trip":
    # 1. SETUP
    resetStream()
    let originalPcm = generateTestPcm(testTotalSamples, testChannels, testSampleRate)

    let encoder = newQOAEncoder(testWriteLong)
    let decoder = newQOADecoder(testReadByte, testSeekToByte)

    # 2. ENCODE
    echo "Encoding ", testTotalSamples, " samples..."
    require encoder.writeHeader(testTotalSamples, testChannels, testSampleRate)

    var samplesWritten = 0
    while samplesWritten < testTotalSamples:
      let chunkSize = min(qoaMaxFrameSamples, testTotalSamples - samplesWritten)
      require encoder.writeFrame(originalPcm.toOpenArray(
        samplesWritten * testChannels,
        (samplesWritten + chunkSize) * testChannels - 1
      ), chunkSize)
      inc samplesWritten, chunkSize

    check samplesWritten == testTotalSamples
    echo "Encoded data size: ", qoaData.len, " bytes"

    # 3. DECODE
    echo "Decoding..."
    decoder.seekToByte(0) # Rewind stream

    require decoder.readHeader()

    # Verify header metadata
    check decoder.getChannels() == testChannels
    check decoder.getSampleRate() == testSampleRate
    check decoder.getTotalSamples() == testTotalSamples

    var decodedPcm = newSeq[int16](testTotalSamples * testChannels)
    var samplesRead = 0
    while not decoder.isEnd():
      let frameSamples = decoder.readFrame(decodedPcm.toOpenArray(
        samplesRead * testChannels,
        decodedPcm.len - 1
      ))
      require frameSamples > 0
      inc samplesRead, frameSamples

    check samplesRead == testTotalSamples

    # 4. VERIFY (using Root Mean Squared Error for lossy comparison)
    echo "Verifying decoded data..."
    var totalError: float64 = 0.0
    for i in 0 ..< originalPcm.len:
      let error = float(originalPcm[i]) - float(decodedPcm[i])
      totalError += error * error

    let mse = totalError / float(originalPcm.len)
    let rmse = sqrt(mse)

    echo "Root Mean Squared Error (RMSE): ", rmse

    # QOA is "Quite OK", not transparent. An error in this range is expected.
    # If the RMSE is very high, the codec is likely broken.
    # If it's 0, it might mean nothing was decoded.
    let errorThreshold = 350.0
    check rmse > 0.0
    check rmse < errorThreshold
