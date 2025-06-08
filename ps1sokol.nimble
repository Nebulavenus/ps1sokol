# Package

version       = "0.1.0"
author        = "Nebula Venus"
description   = "A new awesome nimble package"
license       = "MIT"
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
]

proc compilerSwitch(): string =
  when defined(windows):
      return "--cc:vcc"
  else:
      return ""

proc backendSwitch(): string =
  when defined gl:
      return "-d:gl"
  else:
      return ""

proc build() =
  exec &"nim c --outdir:build/ {compilerSwitch()} {backendSwitch()} src/main"

# Tasks

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
