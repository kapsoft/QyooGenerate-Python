// Create: QyooDemoTests/ModelTests.swift
import XCTest
import CoreML
import UIKit
import CoreVideo
// At the top of your ModelTests.swift file, add this extension:
import XCTest
import CoreML
import UIKit
import CoreVideo
@testable import QyooDemo

extension UIImage {
    func toCVPixelBuffer(size: CGSize) -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(size.width),
            Int(size.height),
            kCVPixelFormatType_32ARGB,
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0)) }
        
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        
        guard let context = CGContext(
            data: pixelData,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: rgbColorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            return nil
        }
        
        context.translateBy(x: 0, y: size.height)
        context.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        UIGraphicsPopContext()
        
        return pixelBuffer
    }
}


class ModelTests: XCTestCase {
    
    var model: MLModel?
    
    override func setUpWithError() throws {
        // Load model once for all tests
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        
        // Add this line - you're missing the modelURL definition
//        guard let modelURL = Bundle(for: type(of: self)).url(forResource: "best", withExtension: "mlmodelc") else {
//            throw NSError(domain: "ModelTestError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model file not found in test bundle"])
//        }
        // Load from main app bundle (since we have host dependency)
        guard let modelURL = Bundle.main.url(forResource: "best", withExtension: "mlmodelc") else {
            throw NSError(domain: "ModelTestError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model file not found in main bundle"])
        }
        
        
        model = try MLModel(contentsOf: modelURL, configuration: config)
        Swift.print("✅ Model loaded for testing")
        
        // Debug model requirements - unwrap the optional
        guard let loadedModel = model else {
            throw NSError(domain: "ModelTestError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Model failed to load"])
        }
        
        let description = loadedModel.modelDescription
        Swift.print("🔍 Model input features:")
        for (name, desc) in description.inputDescriptionsByName {
            Swift.print("  \(name): \(desc)")
            if let imageConstraint = desc.imageConstraint {
                Swift.print("    Expected size: \(imageConstraint.pixelsWide) x \(imageConstraint.pixelsHigh)")
                Swift.print("    Pixel format: \(imageConstraint.pixelFormatType)")
            }
            if let multiArrayConstraint = desc.multiArrayConstraint {
                Swift.print("    Shape: \(multiArrayConstraint.shape)")
                Swift.print("    Data type: \(multiArrayConstraint.dataType)")
            }
        }
        
        Swift.print("🔍 Model output features:")
        for (name, desc) in description.outputDescriptionsByName {
            Swift.print("  \(name): \(desc)")
            if let multiArrayConstraint = desc.multiArrayConstraint {
                Swift.print("    Shape: \(multiArrayConstraint.shape)")
            }
        }
    }
    
    func testModelWithImage1() {
        guard let model = model else {
            XCTFail("Model not loaded")
            return
        }
        
        guard let testImage = UIImage(named: "test_image_1.jpg", in: Bundle(for: type(of: self)), compatibleWith: nil) else {
            Swift.print("Could not load test_image_1.jpg")
            XCTFail("Could not load test_image_1.jpg")
            return
        }

        Swift.print("🧪 Testing with Image 1 - Size: \(testImage.size)")
        
        testModelWithImage(model: model, image: testImage, imageName: "Image 1")
    }
    
    func testModelWithImage2() {
        guard let model = model else {
            XCTFail("Model not loaded")
            return
        }
        
        guard let testImage = UIImage(named: "test_image_2.jpg", in: Bundle(for: type(of: self)), compatibleWith: nil) else {
            Swift.print("Could not load test_image_2.jpg")
            XCTFail("Could not load test_image_1.jpg")
            return
        }
        
        Swift.print("🧪 Testing with Image 2 - Size: \(testImage.size)")
        testModelWithImage(model: model, image: testImage, imageName: "Image 2")
    }
    
    private func testModelWithImage(model: MLModel, image: UIImage, imageName: String) {
        // Convert to pixel buffer
        guard let pixelBuffer = image.toCVPixelBuffer(size: CGSize(width: 512, height: 512)) else {
            XCTFail("Failed to create pixel buffer for \(imageName)")
            return
        }
        
        do {
            // Create input
            let input = try MLDictionaryFeatureProvider(
                dictionary: ["image": MLFeatureValue(pixelBuffer: pixelBuffer)]
            )
            
            // Run prediction
            let output = try model.prediction(from: input)
            
            Swift.print("✅ \(imageName) - Model prediction completed")
            
            // Check outputs exist
            XCTAssertTrue(output.featureNames.contains("var_1052"), "Should have detections output")
            XCTAssertTrue(output.featureNames.contains("p"), "Should have prototypes output")
            
            // Analyze detections
            if let detections = output.featureValue(for: "var_1052")?.multiArrayValue {
                let shape = detections.shape
                Swift.print("  Detections shape: \(shape)")
                
                // Find max confidence
                var maxConf: Float = 0
                let numDetections = shape[2].intValue
                
                for i in 0..<min(100, numDetections) {
                    let conf = detections[[0, 4, i] as [NSNumber]].floatValue
                    maxConf = max(maxConf, conf)
                }
                
                Swift.print("  Max confidence: \(maxConf)")
                XCTAssertGreaterThanOrEqual(maxConf, 0, "Confidence should be non-negative")
            }
            
            // Analyze prototypes
            if let prototypes = output.featureValue(for: "p")?.multiArrayValue {
                Swift.print("  Prototypes shape: \(prototypes.shape)")
            }
            
        } catch {
            XCTFail("\(imageName) prediction failed: \(error)")
        }
    }
}
