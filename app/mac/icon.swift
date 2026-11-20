// Renders the app icon (1024px PNG): blue gradient squircle with a waveform.
import AppKit
let size: CGFloat = 1024
let img = NSImage(size: NSSize(width: size, height: size))
img.lockFocus()
let inset: CGFloat = 100, rect = NSRect(x: inset, y: inset, width: size - 2 * inset, height: size - 2 * inset)
let shadow = NSShadow(); shadow.shadowBlurRadius = 30; shadow.shadowOffset = NSSize(width: 0, height: -12)
shadow.shadowColor = NSColor.black.withAlphaComponent(0.28); shadow.set()
let path = NSBezierPath(roundedRect: rect, xRadius: 185, yRadius: 185)
NSGradient(starting: NSColor(srgbRed: 0.35, green: 0.40, blue: 0.95, alpha: 1),
           ending: NSColor(srgbRed: 0.18, green: 0.56, blue: 0.92, alpha: 1))!.draw(in: path, angle: -50)
NSShadow().set()
// waveform bars
let heights: [CGFloat] = [0.20, 0.42, 0.66, 0.40, 0.78, 0.52, 0.30, 0.56, 0.24]
let barW: CGFloat = 46, gap: CGFloat = 30
let total = CGFloat(heights.count) * barW + CGFloat(heights.count - 1) * gap
var x = (size - total) / 2
NSColor.white.setFill()
for h in heights {
    let bh = h * 520
    NSBezierPath(roundedRect: NSRect(x: x, y: (size - bh) / 2, width: barW, height: bh), xRadius: barW / 2, yRadius: barW / 2).fill()
    x += barW + gap
}
img.unlockFocus()
let rep = NSBitmapImageRep(data: img.tiffRepresentation!)!
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
