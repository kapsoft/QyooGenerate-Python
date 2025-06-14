import XCTest
import CoreML
import UIKit
@testable import QyooDemo

class MaskVisualizationTests: XCTestCase {
    
    var detector: QyooDetector!
    
    override func setUpWithError() throws {
        detector = QyooDetector()
    }
    
    func testMaskExtraction() throws {
        // Load test image
        guard let testImage = UIImage(named: "test_image_2.jpg", in: Bundle(for: type(of: self)), compatibleWith: nil) else {
            XCTFail("Could not load test_image_2.jpg")
            return
        }
        
        print("\n🎭 MASK EXTRACTION TEST")
        print("Input image size: \(testImage.size)")
        
        // Run detection
        guard let result = detector.detect(image: testImage) else {
            XCTFail("No detection found")
            return
        }
        
        print("✅ Detection found!")
        print("  Confidence: \(result.confidence) (\(result.confidence * 100)%)")
        print("  Box: \(result.box)")
        print("  Mask size: \(result.mask.size)")
        
        // Save mask for visualization
        saveMaskVisualization(
            original: testImage,
            mask: result.mask,
            box: result.box,
            confidence: result.confidence
        )
        
        // Verify mask properties
        XCTAssertGreaterThan(result.confidence, 0.5, "Confidence should be > 50%")
        XCTAssertEqual(result.mask.size, testImage.size, "Mask should match image size")
    }
    
    func testMaskGeneration() throws {
        // Test the mask generation with known values
        let detector = QyooDetector()
        
        // Create a simple prototype (2x2 for testing)
        let prototypes = try MLMultiArray(shape: [1, 32, 128, 128], dataType: .float32)
        
        // Set first prototype to have a simple pattern
        for y in 40..<88 {
            for x in 40..<88 {
                prototypes[[0, 0, y, x] as [NSNumber]] = 1.0
            }
        }
        
        // Create coefficients (first one active, others zero)
        var coefficients = [Float](repeating: 0, count: 32)
        coefficients[0] = 5.0  // Strong positive activation
        
        // Generate mask
        let maskArray = detector.generateMaskArray(
            coefficients: coefficients,
            prototypes: prototypes
        )
        
        // Check that mask has high values in the center
        let centerIdx = 64 * 128 + 64
        let centerValue = maskArray[centerIdx]
        
        print("\n🧪 Mask generation test:")
        print("  Center value: \(centerValue)")
        print("  Should be close to sigmoid(5.0) = \(1.0 / (1.0 + exp(-5.0)))")
        
        XCTAssertGreaterThan(centerValue, 0.9, "Center should have high activation")
    }
    
    func testEnhancedMaskVisualization() throws {
        guard let testImage = UIImage(named: "test_image_2.jpg", in: Bundle(for: type(of: self)), compatibleWith: nil) else {
            XCTFail("Could not load test_image_2.jpg")
            return
        }
        
        print("\n🔍 ENHANCED MASK VISUALIZATION")
        
        // Run detection
        guard let result = detector.detect(image: testImage) else {
            XCTFail("No detection found")
            return
        }
        
        // Convert mask to grayscale values
        guard let cgImage = result.mask.cgImage,
              let dataProvider = cgImage.dataProvider,
              let data = dataProvider.data else {
            XCTFail("Failed to get mask data")
            return
        }
        
        let width = cgImage.width
        let height = cgImage.height
        let pixelData = CFDataGetBytePtr(data)!
        
        // Analyze mask values
        var minAlpha: UInt8 = 255
        var maxAlpha: UInt8 = 0
        var histogram = [Int](repeating: 0, count: 256)
        
        for y in 0..<height {
            for x in 0..<width {
                let idx = (y * width + x) * 4 + 3  // Alpha channel
                let alpha = pixelData[idx]
                minAlpha = min(minAlpha, alpha)
                maxAlpha = max(maxAlpha, alpha)
                histogram[Int(alpha)] += 1
            }
        }
        
        print("\n📊 MASK STATISTICS:")
        print("  Alpha range: \(minAlpha) - \(maxAlpha)")
        print("  Non-zero pixels: \(histogram[1...255].reduce(0, +))")
        
        // Find significant alpha values
        print("\n📈 HISTOGRAM (showing non-zero bins):")
        for (value, count) in histogram.enumerated() where count > 0 {
            let percentage = Float(count) * 100.0 / Float(width * height)
            if value > 0 {  // Skip fully transparent
                print("  Alpha \(value): \(count) pixels (\(String(format: "%.2f", percentage))%)")
            }
        }
        
        // Create multiple enhanced visualizations
        createEnhancedVisualizations(mask: result.mask, original: testImage, box: result.box)
    }
    
    func testDetailedMaskDebugging() throws {
        // Load test image
        guard let testImage = UIImage(named: "test_image_2.jpg", in: Bundle(for: type(of: self)), compatibleWith: nil) else {
            XCTFail("Could not load test_image_2.jpg")
            return
        }
        
        print("\n🔍 DETAILED MASK DEBUGGING")
        
        // Get model and run prediction manually to access raw outputs
        guard let pixelBuffer = testImage.toCVPixelBuffer(size: CGSize(width: 512, height: 512)) else {
            XCTFail("Failed to create pixel buffer")
            return
        }
        
        // We need to access the model directly - add a getter in QyooDetector or use reflection
        // For now, let's use the detector's methods
        
        print("\n📊 Running detection to analyze outputs...")
        guard let result = detector.detect(image: testImage) else {
            XCTFail("No detection found")
            return
        }
        
        print("\n🎯 DETECTION RESULT:")
        print("  Confidence: \(result.confidence)")
        print("  Box: \(result.box)")
        print("  Mask pixels: \(result.mask.size.width * result.mask.size.height)")
        
        // Analyze the mask in detail
        analyzeMaskPixels(mask: result.mask, box: result.box)
    }
    
    func testMaskScaling() throws {
        print("\n🔬 TESTING MASK SCALING")
        
        // Create a simple test mask (128x128) with a clear pattern
        var testMask = [Float](repeating: 0, count: 128 * 128)
        
        // Create a square in the center
        for y in 32..<96 {
            for x in 32..<96 {
                testMask[y * 128 + x] = 0.9
            }
        }
        
        // Test box that should contain the mask
        let testBox = CGRect(x: 100, y: 200, width: 400, height: 200)
        let imageSize = CGSize(width: 640, height: 640)
        
        // We can't directly test private methods, so let's create a test through the public interface
        print("Test mask created with center square pattern")
        print("  Original non-zero pixels: \(testMask.filter { $0 > 0.1 }.count)")
        print("  Test box: \(testBox)")
        
        // Create a test image and verify scaling works
        let testImage = createTestImage(size: imageSize)
        if let result = detector.detect(image: testImage) {
            print("  Detection result box: \(result.box)")
            analyzeMaskPixels(mask: result.mask, box: result.box)
        }
    }
    
    // MARK: - Helper Methods
    
    private func saveMaskVisualization(original: UIImage, mask: UIImage, box: CGRect, confidence: Float) {
        // Create visualization with original + mask overlay
        let renderer = UIGraphicsImageRenderer(size: original.size)
        
        let visualization = renderer.image { context in
            // Draw original
            original.draw(at: .zero)
            
            // Draw mask with transparency
            mask.draw(at: .zero, blendMode: .normal, alpha: 0.5)
            
            // Draw bounding box
            context.cgContext.setStrokeColor(UIColor.green.cgColor)
            context.cgContext.setLineWidth(3)
            context.cgContext.stroke(box)
            
            // Add confidence label
            let attributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.boldSystemFont(ofSize: 24),
                .foregroundColor: UIColor.green,
                .backgroundColor: UIColor.black.withAlphaComponent(0.7)
            ]
            
            let text = String(format: "%.1f%%", confidence * 100)
            text.draw(at: CGPoint(x: box.minX, y: box.minY - 30), withAttributes: attributes)
        }
        
        // Save to documents directory (visible in Xcode device explorer)
        if let data = visualization.pngData() {
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            let maskPath = documentsPath.appendingPathComponent("mask_visualization.png")
            let maskOnlyPath = documentsPath.appendingPathComponent("mask_only.png")
            
            try? data.write(to: maskPath)
            try? mask.pngData()?.write(to: maskOnlyPath)
            
            print("\n📁 Saved visualizations to:")
            print("  Full: \(maskPath)")
            print("  Mask only: \(maskOnlyPath)")
            
            // Also print to console for CI/debugging
            print("\n🎨 You can view the mask by:")
            print("  1. Going to Xcode > Devices and Simulators")
            print("  2. Select your device/simulator")
            print("  3. Select the QyooDemo app")
            print("  4. Download Container...")
            print("  5. Look in AppData/Documents/")
        }
    }
    
    private func createEnhancedVisualizations(mask: UIImage, original: UIImage, box: CGRect) {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        
        // 1. Threshold visualization - make faint pixels visible
        saveThresholdMask(mask: mask, threshold: 10, filename: "mask_threshold_10.png")
        saveThresholdMask(mask: mask, threshold: 25, filename: "mask_threshold_25.png")
        saveThresholdMask(mask: mask, threshold: 50, filename: "mask_threshold_50.png")
        
        // 2. Heat map visualization
        saveHeatMapMask(mask: mask, filename: "mask_heatmap.png")
        
        // 3. Binary visualization at different thresholds
        saveBinaryMask(mask: mask, threshold: 0.05, filename: "mask_binary_5percent.png")
        saveBinaryMask(mask: mask, threshold: 0.1, filename: "mask_binary_10percent.png")
        
        // 4. Overlay with enhanced contrast
        saveEnhancedOverlay(mask: mask, original: original, box: box, filename: "mask_enhanced_overlay.png")
        
        print("\n📁 Saved enhanced visualizations to Documents folder")
    }
    
    private func saveThresholdMask(mask: UIImage, threshold: UInt8, filename: String) {
        guard let cgImage = mask.cgImage else { return }
        
        let width = cgImage.width
        let height = cgImage.height
        
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height))
        let enhanced = renderer.image { context in
            // Black background
            UIColor.black.setFill()
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            
            // Get pixel data
            guard let dataProvider = cgImage.dataProvider,
                  let data = dataProvider.data else { return }
            
            let pixelData = CFDataGetBytePtr(data)!
            
            for y in 0..<height {
                for x in 0..<width {
                    let idx = (y * width + x) * 4 + 3  // Alpha channel
                    let alpha = pixelData[idx]
                    
                    if alpha > threshold {
                        // Scale up the visibility
                        let normalized = Float(alpha - threshold) / Float(255 - threshold)
                        UIColor(white: CGFloat(normalized), alpha: 1).setFill()
                        context.fill(CGRect(x: x, y: y, width: 1, height: 1))
                    }
                }
            }
        }
        
        if let data = enhanced.pngData() {
            let path = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                .appendingPathComponent(filename)
            try? data.write(to: path)
        }
    }
    
    private func saveHeatMapMask(mask: UIImage, filename: String) {
        guard let cgImage = mask.cgImage else { return }
        
        let width = cgImage.width
        let height = cgImage.height
        
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height))
        let heatmap = renderer.image { context in
            UIColor.black.setFill()
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            
            guard let dataProvider = cgImage.dataProvider,
                  let data = dataProvider.data else { return }
            
            let pixelData = CFDataGetBytePtr(data)!
            
            for y in 0..<height {
                for x in 0..<width {
                    let idx = (y * width + x) * 4 + 3
                    let alpha = Float(pixelData[idx]) / 255.0
                    
                    // Heat map colors
                    let color: UIColor
                    if alpha > 0.7 {
                        color = UIColor.red
                    } else if alpha > 0.5 {
                        color = UIColor.orange
                    } else if alpha > 0.3 {
                        color = UIColor.yellow
                    } else if alpha > 0.1 {
                        color = UIColor.green
                    } else if alpha > 0.05 {
                        color = UIColor.cyan
                    } else if alpha > 0.01 {
                        color = UIColor.blue
                    } else {
                        continue
                    }
                    
                    color.setFill()
                    context.fill(CGRect(x: x, y: y, width: 1, height: 1))
                }
            }
        }
        
        if let data = heatmap.pngData() {
            let path = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                .appendingPathComponent(filename)
            try? data.write(to: path)
        }
    }
    
    private func saveBinaryMask(mask: UIImage, threshold: Float, filename: String) {
        guard let cgImage = mask.cgImage else { return }
        
        let width = cgImage.width
        let height = cgImage.height
        
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height))
        let binary = renderer.image { context in
            UIColor.black.setFill()
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            
            guard let dataProvider = cgImage.dataProvider,
                  let data = dataProvider.data else { return }
            
            let pixelData = CFDataGetBytePtr(data)!
            let thresholdByte = UInt8(threshold * 255)
            
            UIColor.white.setFill()
            for y in 0..<height {
                for x in 0..<width {
                    let idx = (y * width + x) * 4 + 3
                    if pixelData[idx] > thresholdByte {
                        context.fill(CGRect(x: x, y: y, width: 1, height: 1))
                    }
                }
            }
        }
        
        if let data = binary.pngData() {
            let path = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                .appendingPathComponent(filename)
            try? data.write(to: path)
        }
    }
    
    private func saveEnhancedOverlay(mask: UIImage, original: UIImage, box: CGRect, filename: String) {
        let renderer = UIGraphicsImageRenderer(size: original.size)
        
        let enhanced = renderer.image { context in
            // Draw original
            original.draw(at: .zero)
            
            // Draw mask with colored tint
            context.cgContext.setBlendMode(.multiply)
            UIColor.red.withAlphaComponent(0.5).setFill()
            mask.draw(at: .zero)
            
            // Draw box
            context.cgContext.setBlendMode(.normal)
            context.cgContext.setStrokeColor(UIColor.green.cgColor)
            context.cgContext.setLineWidth(3)
            context.cgContext.stroke(box)
        }
        
        if let data = enhanced.pngData() {
            let path = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                .appendingPathComponent(filename)
            try? data.write(to: path)
        }
    }
    
    private func analyzeMaskPixels(mask: UIImage, box: CGRect) {
        guard let cgImage = mask.cgImage,
              let dataProvider = cgImage.dataProvider,
              let data = dataProvider.data else { return }
        
        let width = cgImage.width
        let height = cgImage.height
        let pixelData = CFDataGetBytePtr(data)!
        
        // Count pixels within the box
        var boxPixels = 0
        var nonZeroBoxPixels = 0
        
        let boxMinX = Int(max(0, box.minX))
        let boxMaxX = Int(min(CGFloat(width), box.maxX))
        let boxMinY = Int(max(0, box.minY))
        let boxMaxY = Int(min(CGFloat(height), box.maxY))
        
        for y in boxMinY..<boxMaxY {
            for x in boxMinX..<boxMaxX {
                boxPixels += 1
                let idx = (y * width + x) * 4 + 3  // Alpha channel
                if pixelData[idx] > 0 {
                    nonZeroBoxPixels += 1
                }
            }
        }
        
        print("\n📦 BOX ANALYSIS:")
        print("  Box dimensions: \(box.width) x \(box.height)")
        print("  Box area: \(boxPixels) pixels")
        print("  Non-zero pixels in box: \(nonZeroBoxPixels)")
        print("  Fill percentage: \(Float(nonZeroBoxPixels) * 100.0 / Float(boxPixels))%")
    }
    
    private func createTestImage(size: CGSize) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { context in
            // Gray background
            UIColor.gray.setFill()
            context.fill(CGRect(origin: .zero, size: size))
            
            // Add some features
            UIColor.white.setFill()
            context.fill(CGRect(x: 200, y: 200, width: 100, height: 100))
        }
    }
}

import Foundation

extension MaskVisualizationTests {
    
    // Get or create TestOutput directory in project
    func getTestOutputDirectory() -> URL {
        let fileManager = FileManager.default
        
        // Try to find the project directory by looking for a TestOutput folder we'll create
        // Start with Documents and create TestOutput there
        let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        let testOutputURL = documentsURL.appendingPathComponent("TestOutput")
        
        // Create directory if it doesn't exist
        if !fileManager.fileExists(atPath: testOutputURL.path) {
            try? fileManager.createDirectory(at: testOutputURL, withIntermediateDirectories: true)
            print("📁 Created TestOutput directory")
        }
        
        return testOutputURL
    }
    
    // Save image with simple name (overwrites existing)
    func saveTestImage(_ image: UIImage, name: String) -> URL? {
        let outputDir = getTestOutputDirectory()
        guard let data = image.pngData() else { return nil }
        
        let fileURL = outputDir.appendingPathComponent("\(name).png")
        
        do {
            try data.write(to: fileURL)
            return fileURL
        } catch {
            print("❌ Failed to save \(name): \(error)")
            return nil
        }
    }
    
    // Main visualization test
    func testQyooVisualization() throws {
        guard let testImage = UIImage(named: "test_image_2.jpg", in: Bundle(for: type(of: self)), compatibleWith: nil) else {
            XCTFail("Could not load test_image_2.jpg")
            return
        }
        
        let outputDir = getTestOutputDirectory()
        print("\n📁 OUTPUT DIRECTORY: \(outputDir.path)")
        
        // For easy copying on simulator
        #if targetEnvironment(simulator)
        print("   Copy this path: \(outputDir.path)")
        #endif
        
        // Save original for reference
        _ = saveTestImage(testImage, name: "original")
        
        // Run detection
        let detector = QyooDetector()
        guard let result = detector.detect(image: testImage) else {
            XCTFail("No detection found")
            return
        }
        
        print("\n✅ DETECTION:")
        print("  Confidence: \(String(format: "%.1f%%", result.confidence * 100))")
        print("  Box: \(result.box)")
        
        // Create visualizations
        let renderer = UIGraphicsImageRenderer(size: testImage.size)
        
        // 1. Detection with box
        let boxViz = renderer.image { context in
            testImage.draw(at: .zero)
            
            context.cgContext.setStrokeColor(UIColor.green.cgColor)
            context.cgContext.setLineWidth(3)
            context.cgContext.stroke(result.box)
            
            let label = String(format: "Qyoo: %.1f%%", result.confidence * 100)
            let attrs: [NSAttributedString.Key: Any] = [
                .font: UIFont.boldSystemFont(ofSize: 20),
                .foregroundColor: UIColor.white,
                .backgroundColor: UIColor.black.withAlphaComponent(0.7)
            ]
            label.draw(at: CGPoint(x: result.box.minX, y: result.box.minY - 25), withAttributes: attrs)
        }
        
        // 2. Mask overlay
        let maskViz = renderer.image { context in
            testImage.draw(at: .zero)
            result.mask.draw(at: .zero, blendMode: .normal, alpha: 0.5)
            
            context.cgContext.setStrokeColor(UIColor.yellow.cgColor)
            context.cgContext.setLineWidth(2)
            context.cgContext.stroke(result.box)
        }
        
        // 3. Cropped detection
        let cropRenderer = UIGraphicsImageRenderer(size: result.box.size)
        let cropped = cropRenderer.image { context in
            testImage.draw(at: CGPoint(x: -result.box.origin.x, y: -result.box.origin.y))
        }
        
        // 4. Mask only on white background
        let maskOnWhite = renderer.image { context in
            UIColor.white.setFill()
            context.fill(CGRect(origin: .zero, size: testImage.size))
            result.mask.draw(at: .zero)
        }
        
        // Save all
        let files = [
            "detection_box": boxViz,
            "mask_overlay": maskViz,
            "cropped_detection": cropped,
            "mask_only": result.mask,
            "mask_on_white": maskOnWhite
        ]
        
        print("\n📸 SAVED FILES:")
        for (name, image) in files {
            if let _ = saveTestImage(image, name: name) {
                print("  ✅ \(name).png")
            }
        }
        
        // Create info file
        let info = """
        Qyoo Detection Results
        ======================
        Date: \(Date())
        Image: test_image_2.jpg
        Confidence: \(String(format: "%.1f%%", result.confidence * 100))
        Box: \(result.box)
        Size: \(Int(result.box.width)) x \(Int(result.box.height))
        
        Files:
        - original.png: Test image
        - detection_box.png: Detection with green box
        - mask_overlay.png: Mask overlay (50% opacity)
        - cropped_detection.png: Just the detected area
        - mask_only.png: Raw mask
        - mask_on_white.png: Mask on white background
        """
        
        let infoURL = outputDir.appendingPathComponent("info.txt")
        try? info.write(to: infoURL, atomically: true, encoding: .utf8)
        
        #if targetEnvironment(simulator)
        print("\n💡 TO VIEW: In Finder, press Cmd+Shift+G and paste:")
        print("  \(outputDir.path)")
        
        // Copy path to clipboard for convenience
        UIPasteboard.general.string = outputDir.path
        print("  📋 Path copied to clipboard!")
        #else
        print("\n💡 TO VIEW ON DEVICE:")
        print("  Xcode > Window > Devices and Simulators")
        print("  > Select device > QyooDemo app > Download Container...")
        print("  > Show Package Contents > AppData/Documents/TestOutput/")
        #endif
    }
    
    // Quick test for mask threshold experiments
    func testMaskThresholds() throws {
        guard let testImage = UIImage(named: "test_image_2.jpg", in: Bundle(for: type(of: self)), compatibleWith: nil) else {
            XCTFail("Could not load test_image_2.jpg")
            return
        }
        
        let detector = QyooDetector()
        
        // You can modify QyooDetector temporarily with different thresholds
        // and run this test to see the differences
        
        if let result = detector.detect(image: testImage) {
            _ = saveTestImage(result.mask, name: "mask_current_threshold")
            print("✅ Saved mask with current threshold settings")
        }
    }
    
    // Clean up test output directory
    func testCleanOutput() throws {
        let outputDir = getTestOutputDirectory()
        let fileManager = FileManager.default
        
        if let files = try? fileManager.contentsOfDirectory(at: outputDir, includingPropertiesForKeys: nil) {
            for file in files {
                try? fileManager.removeItem(at: file)
            }
            print("🧹 Cleaned \(files.count) files from TestOutput directory")
        }
    }
}
