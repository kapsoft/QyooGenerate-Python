import XCTest
import CoreML
import UIKit
import CoreVideo
@testable import QyooDemo

extension UIImage {
    // The working CVPixelBuffer creation method that matches Python
    func toCVPixelBufferFixed(canvasSize: Int = 512) -> CVPixelBuffer? {
        // Create a renderer with explicit settings
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        format.opaque = true
        format.preferredRange = .standard  // Use standard range, not extended
        
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: canvasSize, height: canvasSize), format: format)
        
        let resizedImage = renderer.image { context in
            // White background (matching Python PIL default)
            UIColor.white.setFill()
            context.fill(CGRect(x: 0, y: 0, width: canvasSize, height: canvasSize))
            
            // Draw image
            self.draw(in: CGRect(x: 0, y: 0, width: canvasSize, height: canvasSize))
        }
        
        // Get CGImage
        guard let cgImage = resizedImage.cgImage else { return nil }
        
        // Create CVPixelBuffer
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            canvasSize,
            canvasSize,
            kCVPixelFormatType_32BGRA,  // CoreML expects this format
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }
        
        // Draw using CIContext with no color space conversion
        let ciImage = CIImage(cgImage: cgImage)
        let context = CIContext(options: [
            .useSoftwareRenderer: true,
            .outputColorSpace: NSNull(),  // No color space conversion
            .workingColorSpace: NSNull()   // No color space conversion
        ])
        context.render(ciImage, to: buffer)
        
        return buffer
    }
}

class ModelTests: XCTestCase {
    
    var model: MLModel?
    
    override func setUpWithError() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        
        // Load from main app bundle
        guard let modelURL = Bundle.main.url(forResource: "best", withExtension: "mlmodelc") else {
            throw NSError(domain: "ModelTestError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model file not found in main bundle"])
        }
        
        model = try MLModel(contentsOf: modelURL, configuration: config)
        print("✅ Model loaded for testing")
        
        // Debug model requirements
        guard let loadedModel = model else {
            throw NSError(domain: "ModelTestError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Model failed to load"])
        }
        
        let description = loadedModel.modelDescription
        print("🔍 Model input features:")
        for (name, desc) in description.inputDescriptionsByName {
            print("  \(name): \(desc)")
            if let imageConstraint = desc.imageConstraint {
                print("    Expected size: \(imageConstraint.pixelsWide) x \(imageConstraint.pixelsHigh)")
                print("    Pixel format: \(imageConstraint.pixelFormatType)")
            }
        }
        
        print("🔍 Model output features:")
        for (name, desc) in description.outputDescriptionsByName {
            print("  \(name): \(desc)")
            if let multiArrayConstraint = desc.multiArrayConstraint {
                print("    Shape: \(multiArrayConstraint.shape)")
            }
        }
    }
    
    func testModelWithImage1() {
        guard let model = model else {
            XCTFail("Model not loaded")
            return
        }
        
        guard let testImage = UIImage(named: "test_image_1.jpg", in: Bundle(for: type(of: self)), compatibleWith: nil) else {
            XCTFail("Could not load test_image_1.jpg")
            return
        }

        print("\n🧪 Testing with Image 1 - Size: \(testImage.size)")
        testModelWithImage(model: model, image: testImage, imageName: "Image 1", expectedConfidence: nil)
    }
    
    func testModelWithImage2() {
        guard let model = model else {
            XCTFail("Model not loaded")
            return
        }
        
        guard let testImage = UIImage(named: "test_image_2.jpg", in: Bundle(for: type(of: self)), compatibleWith: nil) else {
            XCTFail("Could not load test_image_2.jpg")
            return
        }
        
        print("\n🧪 Testing with Image 2 - Size: \(testImage.size)")
        // Python gets 55.908203% for this image
        testModelWithImage(model: model, image: testImage, imageName: "Image 2", expectedConfidence: 0.559)
    }
    
    private func testModelWithImage(model: MLModel, image: UIImage, imageName: String, expectedConfidence: Float?) {
        print("🔍 \(imageName) - Original: \(image.size)")
        
        // Create pixel buffer using the working method
        guard let pixelBuffer = image.toCVPixelBufferFixed(canvasSize: 512) else {
            XCTFail("Failed to create pixel buffer for \(imageName)")
            return
        }
        
        // Debug pixel values
        debugPixelValues(pixelBuffer: pixelBuffer, imageName: imageName)
        
        // Make prediction
        do {
            let input = try MLDictionaryFeatureProvider(
                dictionary: ["image": MLFeatureValue(pixelBuffer: pixelBuffer)]
            )
            
            let output = try model.prediction(from: input)
            print("✅ \(imageName) - Model prediction completed")
            
            // Check outputs exist
            XCTAssertTrue(output.featureNames.contains("var_1052"), "Should have detections output")
            XCTAssertTrue(output.featureNames.contains("p"), "Should have prototypes output")
            
            // Analyze detections
            if let detections = output.featureValue(for: "var_1052")?.multiArrayValue {
                let shape = detections.shape
                print("  Detections shape: \(shape)")
                
                var maxConf: Float = 0
                let numDetections = shape[2].intValue
                
                // Check ALL detections to find maximum confidence
                for i in 0..<numDetections {
                    let conf = detections[[0, 4, i] as [NSNumber]].floatValue
                    maxConf = max(maxConf, conf)
                }
                
                print("  🎯 Max confidence: \(maxConf) (\(maxConf * 100)%)")
                
                // Verify confidence if expected
                if let expected = expectedConfidence {
                    let difference = abs(maxConf - expected)
                    print("  📊 Expected: ~\(expected * 100)%, Difference: \(difference * 100)%")
                    XCTAssertLessThan(difference, 0.01, "Confidence should be within 1% of expected")
                }
                
                XCTAssertGreaterThanOrEqual(maxConf, 0, "Confidence should be non-negative")
            }
            
            if let prototypes = output.featureValue(for: "p")?.multiArrayValue {
                print("  Prototypes shape: \(prototypes.shape)")
            }
            
        } catch {
            XCTFail("\(imageName) prediction failed: \(error)")
        }
    }
    
    // Debug function to check pixel values
    func debugPixelValues(pixelBuffer: CVPixelBuffer, imageName: String) {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        
        print("\n🔍 \(imageName) - Pixel buffer analysis:")
        print("  Format: 32BGRA")
        print("  Size: \(width)x\(height), Bytes per row: \(bytesPerRow)")
        
        // Python reference values for Image 2
        if imageName == "Image 2" {
            print("\n📍 PYTHON REFERENCE:")
            print("  Top-left: R=28, G=39, B=35")
            print("  Center: R=128, G=118, B=114")
        }
        
        // Top-left pixel (BGRA format)
        print("\n🔍 iOS Top-left pixel:")
        print("  BGRA: [\(buffer[0]), \(buffer[1]), \(buffer[2]), \(buffer[3])]")
        print("  As RGB: R=\(buffer[2]), G=\(buffer[1]), B=\(buffer[0])")
        
        // Center pixel
        let centerRow = height / 2
        let centerCol = width / 2
        let centerIndex = centerRow * bytesPerRow + centerCol * 4
        
        print("\n🔍 iOS Center pixel:")
        print("  BGRA: [\(buffer[centerIndex]), \(buffer[centerIndex+1]), \(buffer[centerIndex+2]), \(buffer[centerIndex+3])]")
        print("  As RGB: R=\(buffer[centerIndex+2]), G=\(buffer[centerIndex+1]), B=\(buffer[centerIndex])")
        
        // For Image 2, calculate difference from Python
        if imageName == "Image 2" {
            let topLeftR = Int(buffer[2])
            let topLeftG = Int(buffer[1])
            let topLeftB = Int(buffer[0])
            
            let centerR = Int(buffer[centerIndex+2])
            let centerG = Int(buffer[centerIndex+1])
            let centerB = Int(buffer[centerIndex])
            
            print("\n📊 Difference from Python:")
            print("  Top-left: ΔR=\(abs(topLeftR - 28)), ΔG=\(abs(topLeftG - 39)), ΔB=\(abs(topLeftB - 35))")
            print("  Center: ΔR=\(abs(centerR - 128)), ΔG=\(abs(centerG - 118)), ΔB=\(abs(centerB - 114))")
            
            let totalDiff = abs(topLeftR - 28) + abs(topLeftG - 39) + abs(topLeftB - 35) +
                           abs(centerR - 128) + abs(centerG - 118) + abs(centerB - 114)
            
            if totalDiff < 30 {
                print("  ✅ Pixel values match Python closely!")
            } else {
                print("  ⚠️ Significant difference from Python (total: \(totalDiff))")
            }
        }
    }
    
    // Debug test to check image loading (useful for future debugging)
    func testDebugImageLoading() {
        guard let imagePath = Bundle(for: type(of: self)).path(forResource: "test_image_2", ofType: "jpg") else {
            XCTFail("Could not find test_image_2.jpg")
            return
        }
        
        print("\n🔍 DEBUG IMAGE LOADING TEST:")
        
        if let dataProvider = CGDataProvider(filename: imagePath),
           let cgImage = CGImage(jpegDataProviderSource: dataProvider,
                                 decode: nil,
                                 shouldInterpolate: false,
                                 intent: .defaultIntent) {
            
            print("CGImage loaded directly:")
            print("  Size: \(cgImage.width)x\(cgImage.height)")
            print("  Color space: \(String(describing: cgImage.colorSpace))")
            
            // Get raw pixel
            var pixel = [UInt8](repeating: 0, count: 4)
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let context = CGContext(
                data: &pixel,
                width: 1,
                height: 1,
                bitsPerComponent: 8,
                bytesPerRow: 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )!
            
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: CGFloat(cgImage.width), height: CGFloat(cgImage.height)))
            print("  Top-left pixel: R=\(pixel[0]), G=\(pixel[1]), B=\(pixel[2])")
            print("  Expected (Python): R=28, G=39, B=35")
            print("  Note: iOS applies sRGB color profile, Python PIL doesn't")
        }
    }
}
