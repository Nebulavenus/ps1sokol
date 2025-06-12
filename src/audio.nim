# audio.nim
import sokol/app as sapp
import sokol/audio as saudio
import sokol/log as slog
import math
import math/vec3 # For len() and dot() if needed for more complex sound logic
import std/random # For noise generation

const
  SAUDIO_SAMPLE_RATE = 44100.0   # Standard sample rate
  SAUDIO_NUM_SAMPLES = 128       # Number of samples to push per chunk (power of 2 is good)

type
  AudioState = object
    oscillatorPhase: float32
    # For a driving sound:
    targetRpm: float32       # Current target RPM (derived from car speed/acceleration)
    currentRpm: float32      # Smoothly interpolated RPM
    engineFrequency: float32 # Current base frequency of the engine sound
    engineVolume: float32    # Current volume of the engine sound

var audioState: AudioState
var audioSamples: array[SAUDIO_NUM_SAMPLES, float32] # Buffer for samples

proc audioInit*() =
  saudio.setup(saudio.Desc(
    logger: saudio.Logger(fn: slog.fn),
    sampleRate: SAUDIO_SAMPLE_RATE.int32
  ))
  audioState.oscillatorPhase = 0.0
  audioState.targetRpm = 0.0
  audioState.currentRpm = 0.0
  audioState.engineFrequency = 0.0
  audioState.engineVolume = 0.0

proc audioShutdown*() =
  saudio.shutdown()

proc lerp(a, b: float32, t: float32): float32 {.inline.} =
  ## Linearly interpolates between two floats
  return a + (b - a) * t

proc updateEngineSound*(carSpeed: float32, carAccel: float32) =
  # Map car speed to target RPM (adjust these ranges to taste)
  const minRpm = 1000.0 # Idle RPM
  const maxRpm = 6000.0 # Max RPM
  const maxSpeed = 10.0 # Max speed of the car for full RPM

  # A simple mapping of speed to a target RPM
  audioState.targetRpm = minRpm + (maxRpm - minRpm) * clamp(abs(carSpeed) / maxSpeed, 0.0, 1.0)

  # Smoothly interpolate current RPM towards target RPM for less jarring changes
  const rpmSmoothingFactor = 2.0 # How fast currentRpm catches up to targetRpm
  audioState.currentRpm = lerp(audioState.currentRpm, audioState.targetRpm, clamp(sapp.frameDuration() * rpmSmoothingFactor, 0.0, 1.0))

  # Convert RPM to a base frequency (e.g., 2-stroke engine fires every revolution, 4-stroke every 2)
  # A simple 4-stroke engine sound might fire twice per rotation (crankshaft)
  # Or you can just pick a mapping that sounds good.
  const freqPerRpm = 0.015 # Tune this to get desired pitch range
  audioState.engineFrequency = audioState.currentRpm * freqPerRpm

  # Modulate volume based on acceleration (or just speed)
  const baseVolume = 0.2
  const accelVolumeBoost = 0.5 # How much acceleration impacts volume
  # If carAccel is positive (accelerating) volume goes up, if negative (braking) it might go down
  audioState.engineVolume = baseVolume + clamp(carAccel * accelVolumeBoost, 0.0, 0.8) # Clamp to avoid too loud
  audioState.engineVolume = clamp(audioState.engineVolume, 0.0, 1.0)

#[
proc audioGenerateSamples*() =
  let numFrames = saudio.expect()
  if numFrames == 0: return

  for i in 0..<numFrames:
    # Generate the waveform
    # We'll use a simple sawtooth wave for a buzzy engine sound
    # A sine wave is too smooth, a square wave too harsh. Sawtooth is a good compromise.
    let sample = audioState.engineVolume * (2.0 * (audioState.oscillatorPhase - floor(audioState.oscillatorPhase)) - 1.0)

    # Add a little high-frequency noise for texture (optional)
    # let noise = rand(-0.1..0.1) * audioState.engineVolume * 0.2
    # audioSamples[i] = sample + noise

    audioSamples[i] = sample

    # Update phase for next sample
    audioState.oscillatorPhase += audioState.engineFrequency / SAUDIO_SAMPLE_RATE

    # Wrap phase around 1.0
    if audioState.oscillatorPhase >= 1.0:
      audioState.oscillatorPhase -= 1.0

  discard saudio.push(addr(audioSamples[0]), numFrames)
]#

proc audioGenerateSamples*() =
  let expectedFrames = saudio.expect()
  if expectedFrames == 0: return

  # This is the crucial change: only generate samples up to the size of your buffer,
  # or as many as are expected, whichever is smaller.
  # The audio system will typically call saudio.expect() multiple times if it needs more data.
  let framesToGenerate = min(expectedFrames, SAUDIO_NUM_SAMPLES)

  for i in 0..<framesToGenerate: # Loop only up to framesToGenerate
    # Generate the waveform
    let sample = audioState.engineVolume * (2.0 * (audioState.oscillatorPhase - floor(audioState.oscillatorPhase)) - 1.0)

    audioSamples[i] = sample # No out-of-bounds access here now

    # Update phase for next sample
    audioState.oscillatorPhase += audioState.engineFrequency / SAUDIO_SAMPLE_RATE

    # Wrap phase around 1.0
    if audioState.oscillatorPhase >= 1.0:
      audioState.oscillatorPhase -= 1.0

  # Push exactly the number of frames you generated
  discard saudio.push(addr(audioSamples[0]), framesToGenerate)
