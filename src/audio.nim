import sokol/app as sapp
import sokol/audio as saudio
import sokol/log as slog
import math
import math/vec3 # For len() and dot() if needed for more complex sound logic
import std/random # For noise generation

const
  SAUDIO_SAMPLE_RATE = 44100.0   # Standard sample rate
  SAUDIO_NUM_SAMPLES = 128       # Number of samples to push per chunk (power of 2 is good)

  # --- Engine Sound Parameters ---
  MAX_HARMONICS = 3              # Number of harmonic oscillators
  # Multipliers for each harmonic (e.g., 1st, 2nd, 3rd multiples of fundamental)
  HARMONIC_FREQS = [1.0, 2.0, 3.0]

  # Base volume for each harmonic at low RPM (idle)
  # Tune these to shape the idle sound
  HARMONIC_LEVELS_LOW_RPM = [0.8, 0.4, 0.2]
  # Base volume for each harmonic at high RPM (revving)
  # Tune these to shape the high-rev sound
  HARMONIC_LEVELS_HIGH_RPM = [0.6, 0.7, 0.8]

  # --- New Drift Sound Parameters ---
  DRIFT_RPM_BOOST = 800.0        # How much RPM boosts when drifting
  MAX_SCREECH_VOLUME = 0.6       # Max volume of the tire screech
  SCREECH_ATTACK_SPEED = 10.0    # How fast screech volume ramps up
  SCREECH_DECAY_SPEED = 2.0      # How fast screech volume ramps down

type
  AudioState = object
    oscillatorPhases: array[MAX_HARMONICS, float32] # Each harmonic needs its own phase
    # For a driving sound:
    targetRpm: float32       # Current target RPM (derived from car speed/acceleration)
    currentRpm: float32      # Smoothly interpolated RPM
    engineFrequency: float32 # Current base frequency of the engine sound (fundamental)
    engineVolume: float32    # Overall volume of the engine sound

    # Per-harmonic volumes, interpolated dynamically
    harmonicsVolumes: array[MAX_HARMONICS, float32]
    engineNoiseVolume: float32     # Volume of the noise component
    currentScreechVolume: float32  # ADD THIS: Current volume of the tire screech
    # Optional: Perlin noise state for smoother noise
    # PerlinPhase: float32

var audioState: AudioState
var audioSamples: array[SAUDIO_NUM_SAMPLES, float32] # Buffer for samples


proc lerp*(a, b: float32, t: float32): float32 {.inline.} =
  return a + (b - a) * t

proc audioInit*() =
  saudio.setup(saudio.Desc(
    logger: saudio.Logger(fn: slog.fn),
    sampleRate: SAUDIO_SAMPLE_RATE.int32
  ))
  for i in 0..<MAX_HARMONICS:
    audioState.oscillatorPhases[i] = 0.0
  audioState.targetRpm = 0.0
  audioState.currentRpm = 0.0
  audioState.engineFrequency = 0.0
  audioState.engineVolume = 0.0
  audioState.engineNoiseVolume = 0.0
  audioState.currentScreechVolume = 0.0

proc audioShutdown*() =
  saudio.shutdown()

proc updateEngineSound*(carSpeed: float32, carAccel: float32, isDrifting: bool) =
  # Map car speed to target RPM (adjust these ranges to taste)
  const minRpm = 1000.0 # Idle RPM
  const maxRpm = 6000.0 # Max RPM
  const maxSpeed = 10.0 # Max speed of the car for full RPM

  # A simple mapping of speed to a target RPM
  var baseTargetRpm = minRpm + (maxRpm - minRpm) * clamp(abs(carSpeed) / maxSpeed, 0.0, 1.0)

  # --- ADD DRIFT RPM BOOST ---
  if isDrifting:
    baseTargetRpm = min(maxRpm, baseTargetRpm + DRIFT_RPM_BOOST) # Boost RPM, but don't exceed maxRpm
  audioState.targetRpm = baseTargetRpm
  # --- END DRIFT RPM BOOST ---

  # Smoothly interpolate current RPM towards target RPM for less jarring changes
  const rpmSmoothingFactor = 2.0 # How fast currentRpm catches up to targetRpm
  audioState.currentRpm = lerp(audioState.currentRpm, audioState.targetRpm, clamp(sapp.frameDuration() * rpmSmoothingFactor, 0.0, 1.0))

  # Convert RPM to a base frequency (fundamental)
  const freqPerRpm = 0.005 # Tune this to get desired pitch range
  audioState.engineFrequency = audioState.currentRpm * freqPerRpm

  # --- Dynamic Sound Shaping ---
  # Normalize RPM to a 0-1 range for blending harmonic levels
  let rpmNormalized = (audioState.currentRpm - minRpm) / (maxRpm - minRpm)
  let blendFactor = clamp(rpmNormalized, 0.0, 1.0)

  # Interpolate harmonic volumes between low and high RPM states
  for i in 0..<MAX_HARMONICS:
    audioState.harmonicsVolumes[i] = lerp(HARMONIC_LEVELS_LOW_RPM[i], HARMONIC_LEVELS_HIGH_RPM[i], blendFactor)

  # Noise increases with RPM (more mechanical noise at higher revs)
  audioState.engineNoiseVolume = 0.05 + 0.15 * blendFactor
  audioState.engineNoiseVolume = clamp(audioState.engineNoiseVolume, 0.0, 0.2) # Clamp noise to prevent it dominating

  # Modulate overall volume based on acceleration and speed
  const baseVolume = 0.2
  const accelVolumeBoost = 0.5 # How much acceleration impacts volume
  const minSpeedVolume = 0.1 # Minimum volume even when stopped

  # Basic volume scales with speed
  let speedVolume = minSpeedVolume + (1.0 - minSpeedVolume) * clamp(abs(carSpeed) / maxSpeed, 0.0, 1.0)

  # Boost volume on acceleration
  let finalVolume = speedVolume + clamp(carAccel * accelVolumeBoost, 0.0, 0.8)
  audioState.engineVolume = clamp(finalVolume, 0.0, 1.0)

  # --- ADD DRIFTING SCREECH VOLUME LOGIC ---
  let targetScreechVolume = if isDrifting and abs(carSpeed) > 0.5: MAX_SCREECH_VOLUME else: 0.0
  let smoothingSpeed = if isDrifting: SCREECH_ATTACK_SPEED else: SCREECH_DECAY_SPEED
  audioState.currentScreechVolume = lerp(audioState.currentScreechVolume, targetScreechVolume, clamp(sapp.frameDuration() * smoothingSpeed, 0.0, 1.0))
  # --- END DRIFTING SCREECH VOLUME LOGIC ---

proc audioGenerateSamples*() =
  let expectedFrames = saudio.expect()
  if expectedFrames == 0: return

  let framesToGenerate = min(expectedFrames, SAUDIO_NUM_SAMPLES)

  for i in 0..<framesToGenerate:
    var totalSample = 0.0

    # Sum up harmonic components
    for h in 0..<MAX_HARMONICS:
      # Generate a sawtooth wave for each harmonic
      let harmonicSample = (2.0 * (audioState.oscillatorPhases[h] - floor(audioState.oscillatorPhases[h])) - 1.0)
      totalSample += harmonicSample * audioState.harmonicsVolumes[h]

      # Update phase for next sample for this harmonic
      audioState.oscillatorPhases[h] += (audioState.engineFrequency * HARMONIC_FREQS[h]) / SAUDIO_SAMPLE_RATE

      # Wrap phase around 1.0
      if audioState.oscillatorPhases[h] >= 1.0:
        audioState.oscillatorPhases[h] -= 1.0
      elif audioState.oscillatorPhases[h] < 0.0: # In case of negative frequency or other anomalies
        audioState.oscillatorPhases[h] += 1.0

    # Add a white noise component
    totalSample += (rand(-1.0..1.0) * audioState.engineNoiseVolume)

    # --- ADD SCREECH NOISE COMPONENT ---
    # Tire screech is pure noise, its volume controlled by currentScreechVolume
    totalSample += (rand(-1.0..1.0) * audioState.currentScreechVolume)
    # --- END SCREECH NOISE ---

    # Apply overall engine volume
    totalSample *= audioState.engineVolume

    # Clamp final sample to prevent clipping
    audioSamples[i] = clamp(totalSample, -1.0, 1.0)

  discard saudio.push(addr(audioSamples[0]), framesToGenerate)
