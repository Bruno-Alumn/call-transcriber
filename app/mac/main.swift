// Call Transcriber.app — owns the local transcription server (starts it on
// launch, stops it on quit) and shows its live dashboard in a native window.
// Closing the window keeps transcribing (click the Dock icon to bring it
// back); Quit stops the server. If the app dies, the server notices
// (--parent-pid) and shuts itself down.
import AppKit
import WebKit

let home = FileManager.default.homeDirectoryForCurrentUser
let project = home.appendingPathComponent("discord-transcriber")
let serverDir = project.appendingPathComponent("server")
let port = Int(ProcessInfo.processInfo.environment["CT_PORT"] ?? "") ?? 8765   // CT_PORT: for testing
let base = URL(string: "http://127.0.0.1:\(port)/")!

/// Run a command and return its stdout.
func sh(_ path: String, _ args: [String]) -> String {
    let p = Process(); p.executableURL = URL(fileURLWithPath: path); p.arguments = args
    let out = Pipe(); p.standardOutput = out; p.standardError = Pipe()
    do { try p.run() } catch { return "" }
    p.waitUntilExit()
    return String(data: out.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
}

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKScriptMessageHandler {
    var window: NSWindow!
    var web: WKWebView!
    var server: Process?
    var stoppedByUser = false
    var activity: NSObjectProtocol?

    func applicationDidFinishLaunching(_ note: Notification) {
        buildMenu()
        let config = WKWebViewConfiguration()
        config.userContentController.add(self, name: "app")     // dashboard -> app: start/stop/restart
        web = WKWebView(frame: .zero, configuration: config)
        web.navigationDelegate = self
        web.setValue(false, forKey: "drawsBackground")
        // A normal title bar (content starts below it, so nothing sits under
        // the window buttons), tinted to match the dashboard background.
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1320, height: 860),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "Call Transcriber"
        window.titlebarAppearsTransparent = true
        window.backgroundColor = NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                ? NSColor(srgbRed: 0x11 / 255, green: 0x11 / 255, blue: 0x10 / 255, alpha: 1)
                : NSColor(srgbRed: 0xf4 / 255, green: 0xf4 / 255, blue: 0xf2 / 255, alpha: 1)
        }
        window.minSize = NSSize(width: 980, height: 620)   // keeps the two-column, no-page-scroll layout
        window.contentView = web
        window.setFrameAutosaveName("main")
        window.isReleasedWhenClosed = false
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        // Keep transcription running at full speed when the app is in the background.
        activity = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .idleSystemSleepDisabled],
                                                         reason: "Transcribing voice calls")
        startServer()
    }

    // MARK: server

    /// Our server(s) already listening on the port — e.g. one started from a
    /// terminal, or left over from a crash. Only processes running server.py.
    func strayServers() -> [Int32] {
        sh("/usr/sbin/lsof", ["-nP", "-t", "-iTCP:\(port)", "-sTCP:LISTEN"])
            .split(separator: "\n").compactMap { Int32($0) }
            .filter { pid in
                pid != server?.processIdentifier &&
                sh("/bin/ps", ["-o", "command=", "-p", String(pid)]).contains("server.py")
            }
    }

    func kill(_ pids: [Int32]) {
        for pid in pids { Darwin.kill(pid, SIGINT) }
        let deadline = Date().addingTimeInterval(4)
        while Date() < deadline && pids.contains(where: { Darwin.kill($0, 0) == 0 }) { usleep(100_000) }
        for pid in pids where Darwin.kill(pid, 0) == 0 { Darwin.kill(pid, SIGKILL) }
    }

    func startServer() {
        stoppedByUser = false
        if let p = server, p.isRunning { loadDashboard(); return }
        showSplash("Starting the transcription server…")
        DispatchQueue.global().async {
            let strays = self.strayServers()
            if !strays.isEmpty { self.kill(strays) }      // the app owns the server
            DispatchQueue.main.async { self.launch() }
        }
    }

    func launch() {
        let python = serverDir.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: python.path) else {
            showSplash("Couldn't find the server in ~/discord-transcriber/server.", error: true); return
        }
        let log = project.appendingPathComponent("server.log")
        FileManager.default.createFile(atPath: log.path, contents: nil)
        let handle = try? FileHandle(forWritingTo: log)
        let p = Process()
        p.executableURL = python
        p.arguments = ["-u", "server.py", "--port", String(port),
                       "--parent-pid", String(ProcessInfo.processInfo.processIdentifier)]
        p.currentDirectoryURL = serverDir
        p.standardOutput = handle
        p.standardError = handle
        p.terminationHandler = { proc in
            DispatchQueue.main.async {
                guard self.server === proc else { return }
                self.server = nil
                if self.stoppedByUser { self.showStopped() }
                else { self.showSplash("The server stopped unexpectedly (exit \(proc.terminationStatus)).", error: true, offerStart: true) }
            }
        }
        do { try p.run(); server = p } catch {
            showSplash("Couldn't start the server: \(error.localizedDescription)", error: true, offerStart: true); return
        }
        waitForServer(tries: 0)
    }

    func stopServer() {
        guard let p = server, p.isRunning else { showStopped(); return }
        stoppedByUser = true
        showSplash("Stopping the server…")
        DispatchQueue.global().async { self.kill([p.processIdentifier]) }
    }

    func restartServer() {
        guard let p = server, p.isRunning else { startServer(); return }
        stoppedByUser = true
        showSplash("Restarting the server…")
        DispatchQueue.global().async {
            self.kill([p.processIdentifier])
            DispatchQueue.main.async { self.server = nil; self.startServer() }
        }
    }

    func ping(_ done: @escaping (Bool) -> Void) {
        var req = URLRequest(url: base.appendingPathComponent("api/state"))
        req.timeoutInterval = 1
        URLSession.shared.dataTask(with: req) { data, resp, _ in
            let ok = (resp as? HTTPURLResponse)?.statusCode == 200
                && (try? JSONSerialization.jsonObject(with: data ?? Data())) != nil
            DispatchQueue.main.async { done(ok) }
        }.resume()
    }

    func waitForServer(tries: Int) {
        guard server != nil else { return }
        ping { up in
            if up { self.loadDashboard() }
            else if tries > 240 { self.showSplash("The server is taking too long to start. See View ▸ Show Server Log.", error: true) }
            else { DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { self.waitForServer(tries: tries + 1) } }
        }
    }

    func loadDashboard() { web.load(URLRequest(url: base)) }

    func showStopped() {
        showSplash("The server is stopped. Nothing is being transcribed.", offerStart: true, spinner: false)
    }

    func showSplash(_ text: String, error: Bool = false, offerStart: Bool = false, spinner: Bool = true) {
        let button = offerStart ? "<button onclick=\"webkit.messageHandlers.app.postMessage({action:'start'})\">Start server</button>" : ""
        let html = """
        <html><head><meta name="color-scheme" content="light dark"><style>
        html,body{margin:0;height:100%;overflow:hidden}
        body{display:grid;place-items:center;font:15px -apple-system,system-ui;background:#f4f4f2;color:#52514e}
        @media (prefers-color-scheme:dark){body{background:#111110;color:#c3c2b7}}
        .s{width:28px;height:28px;border-radius:50%;border:3px solid #3987e544;border-top-color:#3987e5;animation:r 1s linear infinite;margin:0 auto 16px}
        @keyframes r{to{transform:rotate(360deg)}} .e{color:#e66767}
        button{margin-top:16px;font:inherit;font-size:14px;padding:8px 18px;border-radius:10px;border:0;background:#3987e5;color:#fff;cursor:pointer}
        </style></head><body><div style="text-align:center">\((spinner && !error && !offerStart) ? "<div class=s></div>" : "")<div class="\(error ? "e" : "")">\(text)</div>\(button)</div></body></html>
        """
        web.loadHTMLString(html, baseURL: nil)
    }

    // Dashboard buttons (server popover) and the splash's Start button.
    func userContentController(_ c: WKUserContentController, didReceive message: WKScriptMessage) {
        guard let body = message.body as? [String: Any], let action = body["action"] as? String else { return }
        switch action {
        case "start": startServer()
        case "stop": stopServer()
        case "restart": restartServer()
        case "relaunch": relaunchAfterUpdate(rebuild: body["rebuild"] as? Bool ?? false, deps: body["deps"] as? Bool ?? false)
        case "save": save(name: body["name"] as? String ?? "transcript.txt", content: body["content"] as? String ?? "")
        default: break
        }
    }

    // MARK: export + update

    /// Native Save dialog for transcript exports (WKWebView can't do blob downloads).
    func save(name: String, content: String) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = name
        panel.directoryURL = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first
        panel.canCreateDirectories = true
        panel.beginSheetModal(for: window) { response in
            guard response == .OK, let url = panel.url else { return }
            do { try content.write(to: url, atomically: true, encoding: .utf8) }
            catch { NSAlert(error: error).runModal() }
        }
    }

    /// After the server pulled a new release: stop it, refresh dependencies,
    /// rebuild this app if its code changed, then reopen.
    func relaunchAfterUpdate(rebuild: Bool, deps: Bool) {
        stoppedByUser = true
        showSplash("Installing the update…")
        let pid = server?.processIdentifier
        DispatchQueue.global().async {
            if let pid = pid { self.kill([pid]) }
            let zsh = { (cmd: String) -> Int32 in
                let p = Process(); p.executableURL = URL(fileURLWithPath: "/bin/zsh"); p.arguments = ["-lc", cmd]
                p.standardOutput = FileHandle.nullDevice; p.standardError = FileHandle.nullDevice
                try? p.run(); p.waitUntilExit(); return p.terminationStatus
            }
            if deps { _ = zsh("cd \"\(serverDir.path)\" && uv sync --inexact") }
            var ok = true
            if rebuild { ok = zsh("\"\(project.path)/app/mac/build.sh\"") == 0 }
            DispatchQueue.main.async {
                if !ok { self.showSplash("The app rebuild failed. Run app/mac/build.sh in a terminal to see why.", error: true, offerStart: true); return }
                // Reopen (the freshly built copy in ~/Applications, or this one).
                let target = rebuild ? home.appendingPathComponent("Applications/Call Transcriber.app").path : Bundle.main.bundlePath
                let p = Process(); p.executableURL = URL(fileURLWithPath: "/bin/sh")
                p.arguments = ["-c", "sleep 1; open -n \"\(target)\""]
                try? p.run()
                self.server = nil
                NSApp.terminate(nil)
            }
        }
    }

    // The dashboard lost its server mid-load: show state instead of a blank page.
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        if server == nil { stoppedByUser ? showStopped() : showSplash("Server not running.", error: true, offerStart: true) }
        else { DispatchQueue.main.asyncAfter(deadline: .now() + 1) { self.waitForServer(tries: 0) } }
    }

    // MARK: lifecycle

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        window.makeKeyAndOrderFront(nil); return true
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    func applicationWillTerminate(_ note: Notification) {
        guard let p = server, p.isRunning else { return }
        server = nil
        kill([p.processIdentifier])                       // SIGINT first: lets the server flush pending lines
    }

    // MARK: menu

    func buildMenu() {
        let main = NSMenu()
        func item(_ m: NSMenu, _ title: String, _ sel: Selector, _ key: String, _ mods: NSEvent.ModifierFlags = .command) {
            let i = NSMenuItem(title: title, action: sel, keyEquivalent: key); i.keyEquivalentModifierMask = mods; m.addItem(i)
        }
        let appItem = NSMenuItem(); main.addItem(appItem)
        let app = NSMenu()
        item(app, "About Call Transcriber", #selector(NSApplication.orderFrontStandardAboutPanel(_:)), "")
        app.addItem(.separator())
        item(app, "Hide Call Transcriber", #selector(NSApplication.hide(_:)), "h")
        app.addItem(.separator())
        item(app, "Quit Call Transcriber (stops transcribing)", #selector(NSApplication.terminate(_:)), "q")
        appItem.submenu = app

        let srvItem = NSMenuItem(); main.addItem(srvItem)
        let srv = NSMenu(title: "Server")
        item(srv, "Start Server", #selector(menuStart), "")
        item(srv, "Restart Server", #selector(menuRestart), "r", [.command, .shift])
        item(srv, "Stop Server", #selector(menuStop), "")
        srv.addItem(.separator())
        item(srv, "Show Server Log", #selector(openLog), "l")
        srvItem.submenu = srv

        let viewItem = NSMenuItem(); main.addItem(viewItem)
        let view = NSMenu(title: "View")
        item(view, "Reload Dashboard", #selector(reload), "r")
        item(view, "Show/Hide Calls List", #selector(toggleCalls), "\\")
        item(view, "Search All Calls…", #selector(searchAll), "f", [.command, .shift])
        item(view, "Open Transcripts Folder", #selector(openTranscripts), "o")
        item(view, "Open in Browser", #selector(openBrowser), "b")
        viewItem.submenu = view

        let editItem = NSMenuItem(); main.addItem(editItem)
        let edit = NSMenu(title: "Edit")
        item(edit, "Copy", #selector(NSText.copy(_:)), "c")
        item(edit, "Select All", #selector(NSText.selectAll(_:)), "a")
        item(edit, "Find…", #selector(focusSearch), "f")
        editItem.submenu = edit

        let winItem = NSMenuItem(); main.addItem(winItem)
        let win = NSMenu(title: "Window")
        item(win, "Close Window (keeps transcribing)", #selector(NSWindow.performClose(_:)), "w")
        item(win, "Minimize", #selector(NSWindow.miniaturize(_:)), "m")
        winItem.submenu = win
        NSApp.mainMenu = main
    }

    @objc func menuStart() { startServer() }
    @objc func menuStop() { stopServer() }
    @objc func menuRestart() { restartServer() }
    @objc func reload() { if server != nil { loadDashboard() } else { showStopped() } }
    @objc func openTranscripts() { NSWorkspace.shared.open(project.appendingPathComponent("transcripts")) }
    @objc func openLog() { NSWorkspace.shared.open(project.appendingPathComponent("server.log")) }
    @objc func openBrowser() { NSWorkspace.shared.open(base) }
    @objc func focusSearch() { web.evaluateJavaScript("document.getElementById('q')?.focus()") }
    @objc func toggleCalls() { web.evaluateJavaScript("document.getElementById('toggle-side')?.click()") }
    @objc func searchAll() { web.evaluateJavaScript("(()=>{const s=document.getElementById('shell');s?.classList.remove('collapsed');document.getElementById('hq')?.focus()})()") }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
