#!/bin/zsh
# Builds "Call Transcriber.app" (the desktop app in app/desktop, on Electron)
# and installs it to ~/Applications.
#
# The bundle is Electron's own app with our name, icon and permissions; it
# loads the app code from this repository, so code updates need no rebuild.
# A rebuild is only needed when the Electron version changes.
set -e
cd "$(dirname "$0")"
MAC="$PWD"
REPO="$(cd ../.. && pwd)"
DESKTOP="$REPO/app/desktop"

# 1. Electron (pinned in app/desktop/package.json)
want=$(node -p "require('$DESKTOP/package.json').devDependencies.electron")
have=$(cat "$DESKTOP/node_modules/electron/dist/version" 2>/dev/null || true)
if [[ "$have" != "$want" ]]; then
  echo "Installing Electron $want…"
  (cd "$DESKTOP" && npm ci --no-audit --no-fund --loglevel=error && node node_modules/electron/install.js)
fi

# 2. Bundle
APP="build/Call Transcriber.app"
rm -rf build && mkdir -p build/icon.iconset
cp -R "$DESKTOP/node_modules/electron/dist/Electron.app" "$APP"
swiftc -O -o build/icon icon.swift -framework AppKit && ./build/icon build/icon-1024.png
for s in 16 32 128 256 512; do
  sips -z $s $s build/icon-1024.png --out build/icon.iconset/icon_${s}x${s}.png >/dev/null
  sips -z $((s*2)) $((s*2)) build/icon-1024.png --out build/icon.iconset/icon_${s}x${s}@2x.png >/dev/null
done
iconutil -c icns build/icon.iconset -o "$APP/Contents/Resources/electron.icns"

PL="$APP/Contents/Info.plist"
pb() { /usr/libexec/PlistBuddy -c "$1" "$PL" 2>/dev/null || true; }
VERSION=$(cat "$REPO/VERSION")
pb "Set :CFBundleName Call Transcriber"
pb "Add :CFBundleDisplayName string Call Transcriber"; pb "Set :CFBundleDisplayName Call Transcriber"
pb "Set :CFBundleIdentifier local.calltranscriber"
pb "Set :CFBundleShortVersionString $VERSION"
pb "Set :CFBundleVersion $VERSION"
pb "Add :NSMicrophoneUsageDescription string Discord voice calls use your microphone."
pb "Set :NSMicrophoneUsageDescription Discord voice calls use your microphone."
pb "Add :NSCameraUsageDescription string Discord video calls use your camera."
pb "Set :NSCameraUsageDescription Discord video calls use your camera."

# 3. The app code: a stub that loads app/desktop from this repository.
RES="$APP/Contents/Resources"
rm -f "$RES/default_app.asar"
mkdir -p "$RES/app"
cat > "$RES/app/package.json" <<JSON
{ "name": "call-transcriber", "productName": "Call Transcriber", "version": "$VERSION", "main": "index.js" }
JSON
# (JSON-quoted path, so spaces and quotes in it are safe)
node -e "require('fs').writeFileSync(process.argv[1], 'require(' + JSON.stringify(process.argv[2]) + ');\n')" "$RES/app/index.js" "$DESKTOP/main.js"

codesign --force --deep -s - "$APP" 2>/dev/null
DEST="${CT_INSTALL_DIR:-$HOME/Applications}"     # (tests install elsewhere)
mkdir -p "$DEST"
rm -rf "$DEST/Call Transcriber.app"
cp -R "$APP" "$DEST/"
echo "Installed: $DEST/Call Transcriber.app"
