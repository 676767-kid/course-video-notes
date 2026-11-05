//
//  vision_ocr.swift —— course-video-notes 的 Apple Vision 批量 OCR 小工具
//
//  为什么要有它：
//    macOS 自带的 Vision 框架对中文印刷体（课件、PPT 截图）识别效果很好，
//    但 Python 没有 pip 安装权限时拿不到 pyobjc，只能用 swiftc 现场编译一个
//    小命令行程序来调用。100+ 页幻灯片必须「一次进程批量处理」，否则每页都
//    启动一次 Vision（每次初始化 + 加载语言模型约 0.3~0.5 秒）会很慢。
//
//  调用协议（ocr.py 依赖，改动需同步两边）：
//    标准输入：每行一个图片文件的绝对路径（UTF-8，允许空格 / 中文）。
//    标准输出：每行一个 JSON 对象，与输入行一一对应（顺序一致）：
//
//      {"file": "/abs/a.png",
//       "width": 1200, "height": 800,           // 像素尺寸（附加信息）
//       "lines": [{"text": "第三章 电枢反应",     // 识别文本
//                  "conf": 0.5,                  // 置信度 0..1
//                  "x": 0.05, "y": 0.087,        // 归一化左上角（y 已翻转为自上而下）
//                  "w": 0.29, "h": 0.062}]}      // 归一化宽高
//
//      识别失败（文件损坏/不是图片）时输出：
//      {"file": "/abs/bad.png", "error": "load-failed"}
//      单张失败不会中断整批，也不会影响其它行。
//
//  输入为空行会被跳过；程序读完 stdin 后退出（退出码 0）。
//  单张图片出问题时用 stderr 打一行提示，stdout 永远只放 JSON。
//
//  可选参数（ocr.py 会传）：
//    --languages zh-Hans,en-US   识别语言，逗号分隔，默认中文简体+英文
//    --fast                      用 .fast 档（快但准确率低，应急用）
//    --version                   打印版本后退出
//

import Foundation
import CoreGraphics
import ImageIO
import Vision
import AppKit

// ---------------------------------------------------------------------------
// 参数解析（手写，避免依赖 ArgumentParser）
// ---------------------------------------------------------------------------

var languages: [String] = ["zh-Hans", "en-US"]
// 用 Bool 而不是存枚举变量：新版 SDK 把该枚举挂在 Vision.RecognizeTextRequest
// 上（Swift overlay），直接写 VNRecognitionLevel 是编译不过的，存 Bool 最省事。
var useFastLevel = false
let version = "1.1"

func parseArgs(_ raw: [String]) -> [String] {
    var rest: [String] = []
    var idx = 0
    while idx < raw.count {
        let arg = raw[idx]
        switch arg {
        case "--languages":
            idx += 1
            if idx < raw.count {
                languages = raw[idx]
                    .split(separator: ",")
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                    .filter { !$0.isEmpty }
            }
        case "--fast":
            useFastLevel = true
        case "--accurate":
            useFastLevel = false
        case "--version":
            print("vision_ocr \(version)")
            exit(0)
        case "--help", "-h":
            FileHandle.standardError.write(
                "usage: vision_ocr [--languages zh-Hans,en-US] [--fast] < image_paths.txt\n"
                    .data(using: .utf8)!)
            exit(2)
        default:
            // 允许把图片路径直接写进命令行（调试时方便），但正常走 stdin
            if !arg.hasPrefix("-") { rest.append(arg) }
        }
        idx += 1
    }
    return rest
}

// ---------------------------------------------------------------------------
// 图片加载：先用 ImageIO（支持方向校正 + 超大图缩略），失败再退回 NSImage
// ---------------------------------------------------------------------------

func loadCGImage(_ path: String) -> CGImage? {
    let url = URL(fileURLWithPath: path)
    guard FileManager.default.fileExists(atPath: path) else { return nil }

    // ImageIO：手机拍的图会带 EXIF 方向，用 thumbnail 接口可以顺带校正并限幅
    if let src = CGImageSourceCreateWithURL(url as CFURL, nil) {
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: 4096,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        if let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary) {
            return cg
        }
        if let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) {
            return cg
        }
    }

    // 兜底：NSImage 能读一些 ImageIO 不认的格式
    if let img = NSImage(contentsOf: url),
       let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) {
        return cg
    }
    return nil
}

// ---------------------------------------------------------------------------
// 单张图片识别
// ---------------------------------------------------------------------------

func recognize(_ path: String) -> [String: Any] {
    guard let cg = loadCGImage(path) else {
        return ["file": path, "error": "load-failed"]
    }

    let request = VNRecognizeTextRequest()
    request.recognitionLevel = useFastLevel ? .fast : .accurate
    request.recognitionLanguages = languages
    request.usesLanguageCorrection = true
    // 注意：VNRecognizeTextRequest 上没有 recognitionCandidates 属性，别加。

    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    do {
        try handler.perform([request])
    } catch {
        return ["file": path, "error": "ocr-failed"]
    }

    var lines: [[String: Any]] = []
    for observation in (request.results ?? []) {
        guard let candidate = observation.topCandidates(1).first else { continue }
        let box = observation.boundingBox          // 归一化，原点在左下
        lines.append([
            "text": candidate.string,
            "conf": Double(candidate.confidence),
            "x": Double(box.origin.x),
            // Vision 的 y 轴自下而上，这里翻转成自上而下，下游排序才好写
            "y": Double(1.0 - box.origin.y - box.size.height),
            "w": Double(box.size.width),
            "h": Double(box.size.height),
        ])
    }

    return [
        "file": path,
        "width": cg.width,
        "height": cg.height,
        "lines": lines,
    ]
}

// ---------------------------------------------------------------------------
// 主循环：逐行读 stdin，逐行写 stdout（不攒到最后，便于 Python 端显示进度）
// ---------------------------------------------------------------------------

func emit(_ object: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: object, options: []),
          var text = String(data: data, encoding: .utf8) else { return }
    text += "\n"
    FileHandle.standardOutput.write(text.data(using: .utf8)!)
}

let positional = parseArgs(Array(CommandLine.arguments.dropFirst()))

if !positional.isEmpty {
    for path in positional { emit(recognize(path)) }
    exit(0)
}

while let raw = readLine(strippingNewline: true) {
    let path = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    if path.isEmpty { continue }
    emit(recognize(path))
}
