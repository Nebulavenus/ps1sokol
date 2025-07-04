# Package

version       = "0.1.0"
author        = "Nebula Venus"
description   = "A new awesome nimble package"
license       = "The Unlicense"
srcDir        = "src"
installExt    = @["nim"]
bin           = @["main"]


# Dependencies

requires "nim >= 2.2.4"
requires "sokol"

# Compilation

import strformat

let shaders = [
    "cube",
    "texcube",
    "default"
]

let audios = [
    "1", "2", "3",
    "4", "5", "6"
]

proc compilerSwitch(): string =
  when defined(windows):
      #return "--cc:vcc"
      # Disable microsoft compiler, compile instead with nim's gcc
      return ""
  else:
      return ""

proc backendSwitch(): string =
  when defined gl:
      return "-d:gl"
  else:
      return ""

proc build() =
  exec &"nim c -d:release --outdir:build/ {compilerSwitch()} {backendSwitch()} src/main"

proc buildProfile() =
  exec &"nim c --profiler:on --stackTrace:on -d:release --outdir:build/ {compilerSwitch()} {backendSwitch()} src/main"

proc buildDebug() =
  exec &"nim c -d:debug --debugger:native --outdir:build/ {compilerSwitch()} {backendSwitch()} src/main"

# Tasks

task convertPng, "Convert PNG to QOI":
    exec &"nim c --outdir:build/ src/converterqoi"
    exec &"build/converterqoi"

task convertAudio, "Convert audio to QOA":
  let binDir = "tools/goqoa/bin/"
  let goqoaPath =
    when defined(windows):
      &"{binDir}win32/goqoa"
    elif defined(macosx) and defined(arm64):
      &"{binDir}osx_arm64/goqoa"
    elif defined(macosx):
      &"{binDir}osx/goqoa"
    else:
      &"{binDir}linux/goqoa"
  for audio in audios:
    # I hate this, lets rename all tracks to be digits..
    #let cmd = fmt"""{goqoaPath} convert "\assets/music/{audio}.ogg"\ "\assets/music/{audio}.qoa"\"""
    #let cmd = fmt"""{goqoaPath} convert \"assets/music/{audio}\".ogg \"assets/music/{audio}\".qoa"""
    let cmd = fmt"""{goqoaPath} convert assets/music/{audio}.ogg assets/music/{audio}.qoa"""
    echo &"   {cmd}"
    exec cmd

task debuggame, "Debug the game":
  buildDebug()
  exec &"build/main"

task profilegame, "Profile the game":
  buildProfile()
  exec &"build/main"

task game, "Runs the game":
  build()
  exec &"build/main"

task shaders, "Compile all shaders (launched from tools/sokol-tools-bin)":
  let binDir = "tools/sokol-tools-bin/bin/"
  let shdcPath =
    when defined(windows):
      &"{binDir}win32/sokol-shdc"
    elif defined(macosx) and defined(arm64):
      &"{binDir}osx_arm64/sokol-shdc"
    elif defined(macosx):
      &"{binDir}osx/sokol-shdc"
    else:
      &"{binDir}linux/sokol-shdc"
  for shader in shaders:
    let cmd = &"{shdcPath} -i src/shaders/{shader}.glsl -o src/shaders/{shader}.nim -l glsl430:metal_macos:hlsl5:glsl300es -f sokol_nim"
    echo &"   {cmd}"
    exec cmd

task docgen, "Generate documentation":
  exec "nim doc --project --index:on --outdir:docs/ src/main.nim"
