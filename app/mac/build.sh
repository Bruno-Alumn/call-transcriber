#!/bin/zsh
# Builds "Call Transcriber.app" and installs it to ~/Applications.
set -e
cd "$(dirname "$0")"
APP="build/Call Transcriber.app"
rm -rf build && mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" build/icon.iconset
swiftc -O -o "$APP/Contents/MacOS/CallTranscriber" main.swift -framework AppKit -framework WebKit
swiftc -O -o build/icon icon.swift -framework AppKit && ./build/icon build/icon-1024.png
for s in 16 32 128 256 512; do
  sips -z $s $s build/icon-1024.png --out build/icon.iconset/icon_${s}x${s}.png >/dev/null
  sips -z $((s*2)) $((s*2)) build/icon-1024.png --out build/icon.iconset/icon_${s}x${s}@2x.png >/dev/null
done
iconutil -c icns build/icon.iconset -o "$APP/Contents/Resources/AppIcon.icns"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Call Transcriber</string>
  <key>CFBundleDisplayName</key><string>Call Transcriber</string>
  <key>CFBundleIdentifier</key><string>local.calltranscriber</string>
  <key>CFBundleExecutable</key><string>CallTranscriber</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
</dict></plist>
PLIST
codesign --force --deep -s - "$APP"
mkdir -p ~/Applications
rm -rf ~/Applications/"Call Transcriber.app"
cp -R "$APP" ~/Applications/
echo "Installed: ~/Applications/Call Transcriber.app"
