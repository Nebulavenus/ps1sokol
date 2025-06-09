import os, osproc, strutils

const
  assetsDir = "assets"
  sourceExt = ".png"
  targetExt = ".qoi"

# Check for magick command
let magickPath = findExe("magick")
if magickPath == "":
  echo "Error: 'magick' command not found."
  echo "Please install ImageMagick (v7+ is recommended for 'magick' command):"
  echo "  https://imagemagick.org/index.php"
  quit(1)

if not assetsDir.dirExists():
  echo "Error: Directory '" & assetsDir & "' not found."
  echo "Please create it and place your .png files inside."
  quit(1)

echo "--- Starting PNG to QOI Conversion ---"
echo "Source Directory: " & assetsDir
echo "Source Extension: " & sourceExt
echo "Target Extension: " & targetExt
echo ""

var convertedCount = 0
var failedCount = 0

# Use walkDir to traverse all files and subdirectories
for fullPath in walkDir(assetsDir):
  if fullPath[1].endsWith(sourceExt):
    # Construct the output path by changing the extension
    let outputPath = fullPath[1].changeFileExt(targetExt)

    echo "Converting " & fullPath[1] & " -> " & outputPath

    let
      #cmd = ["magick", fullPath[1], outputPath]
      args = ["convert", $fullPath[1], "-colorspace RGBA", $outputPath]
      # poStdErrToStdOut ensures magick's error messages are visible
      exitCode = execProcess(command = "magick", args = args, options = {poStdErrToStdOut})

    if true:
      echo "  OK"
      convertedCount.inc
    else:
      echo "  FAILED with exit code: " & $exitCode
      failedCount.inc

echo ""
echo "--- Conversion Summary ---"
echo "Converted: " & $convertedCount & " files"
echo "Failed:    " & $failedCount & " files"

if failedCount > 0:
  quit(1) # Indicate failure to the calling process (nimble)
