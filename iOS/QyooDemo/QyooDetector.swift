import CoreML
import Vision
import UIKit
import Accelerate

class QyooDetector {
    private let model: MLModel
    private let imageSize: CGFloat = 512
    
    init() {
        print("Looking for model in bundle...")
        print("Bundle path: \(Bundle.main.bundlePath)")
        print("Bundle resources path: \(Bundle.main.resourcePath ?? "nil")")
        
        // List all files in bundle for debugging
        if let resourcePath = Bundle.main.resourcePath {
            let allFiles = try? FileManager.default.contentsOfDirectory(atPath: resourcePath)
            print("ALL files in bundle: \(allFiles ?? [])")
        }
        
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly  // Force CPU to avoid GPU compatibility issues
            
            if let modelURL = Bundle.main.url(forResource: "best", withExtension: "mlmodelc") {
                print("Found model at: \(modelURL)")
                self.model = try MLModel(contentsOf: modelURL, configuration: config)
                print("Successfully loaded model with CPU-only config")
            } else {
                print("Model not found in bundle")
                
                // Check for both .mlmodelc and .mlpackage files
                let mlmodelcPaths = Bundle.main.paths(forResourcesOfType: "mlmodelc", inDirectory: nil)
                let mlpackagePaths = Bundle.main.paths(forResourcesOfType: "mlpackage", inDirectory: nil)
                
                print("Number of mlmodelc files found: \(mlmodelcPaths.count)")
                for path in mlmodelcPaths {
                    print("- \(path)")
                }
                
                print("Number of mlpackage files found: \(mlpackagePaths.count)")
                for path in mlpackagePaths {
                    print("- \(path)")
                }
                
                fatalError("Failed to find model in bundle")
            }
        } catch {
            print("Error loading model: \(error)")
            print("Error details: \(error.localizedDescription)")
            fatalError("Failed to load model: \(error)")
        }
    }
    
    func detect(image: UIImage) -> (mask: UIImage, confidence: Float)? {
        guard let pixelBuffer = image.toCVPixelBuffer(size: CGSize(width: imageSize, height: imageSize)) else {
            return nil
        }
        
        do {
            // Run prediction
            let input = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: pixelBuffer)])
            let output = try model.prediction(from: input)
            
            // Get outputs
            guard let detections = output.featureValue(for: "var_1052")?.multiArrayValue,
                  let prototypes = output.featureValue(for: "p")?.multiArrayValue else {
                return nil
            }
            
            // Process detections
            let (mask, confidence) = processYOLOv8Output(detections: detections, prototypes: prototypes, originalImage: image)
            guard let mask = mask else { return nil }
            return (mask, confidence)
            
        } catch {
            print("Prediction error: \(error)")
            return nil
        }
    }
    
    private func processYOLOv8Output(detections: MLMultiArray, prototypes: MLMultiArray, originalImage: UIImage) -> (UIImage?, Float) {
        // YOLOv8 output format: [1, 37, 5376]
        // 37 = 4 (box) + 1 (class) + 32 (mask coefficients)
        let numDetections = detections.shape[2].intValue // 5376
        
        var bestConfidence: Float = 0.3 // threshold
        var bestBox: CGRect?
        var bestMaskCoeffs: [Float] = []
        
        // Find best detection
        for i in 0..<numDetections {
            // Get confidence (class probability)
            let confidence = detections[[0, 4, i] as [NSNumber]].floatValue
            
            if confidence > bestConfidence {
                bestConfidence = confidence
                
                // Extract box coordinates (x_center, y_center, width, height)
                let xc = detections[[0, 0, i] as [NSNumber]].floatValue
                let yc = detections[[0, 1, i] as [NSNumber]].floatValue
                let w = detections[[0, 2, i] as [NSNumber]].floatValue
                let h = detections[[0, 3, i] as [NSNumber]].floatValue
                
                // Convert to CGRect (YOLO format to pixel coordinates)
                let x = CGFloat(xc - w/2) * originalImage.size.width / imageSize
                let y = CGFloat(yc - h/2) * originalImage.size.height / imageSize
                let width = CGFloat(w) * originalImage.size.width / imageSize
                let height = CGFloat(h) * originalImage.size.height / imageSize
                
                bestBox = CGRect(x: x, y: y, width: width, height: height)
                
                // Extract 32 mask coefficients
                bestMaskCoeffs = []
                for j in 5..<37 {
                    bestMaskCoeffs.append(detections[[0, j, i] as [NSNumber]].floatValue)
                }
            }
        }
        
        // If no detection found
        guard let box = bestBox, !bestMaskCoeffs.isEmpty else {
            return (nil, 0.0)
        }
        
        // Generate mask from coefficients and prototypes
        let mask = generateMask(coefficients: bestMaskCoeffs, prototypes: prototypes, box: box, imageSize: originalImage.size)
        
        return (mask, bestConfidence)
    }
    
    private func generateMask(coefficients: [Float], prototypes: MLMultiArray, box: CGRect, imageSize: CGSize) -> UIImage? {
        // Prototypes shape: [1, 32, 128, 128]
        let protoH = 128
        let protoW = 128
        
        // Create mask by combining prototypes with coefficients
        var mask = [Float](repeating: 0, count: protoH * protoW)
        
        // Linear combination of prototypes
        for p in 0..<32 {
            let coeff = coefficients[p]
            for y in 0..<protoH {
                for x in 0..<protoW {
                    let idx = y * protoW + x
                    let protoValue = prototypes[[0, p, y, x] as [NSNumber]].floatValue
                    mask[idx] += coeff * protoValue
                }
            }
        }
        
        // Apply sigmoid to get probabilities
        for i in 0..<mask.count {
            mask[i] = 1.0 / (1.0 + exp(-mask[i]))
        }
        
        // Crop mask to box region and resize to original image
        return createMaskImage(mask: mask, protoSize: CGSize(width: protoW, height: protoH), 
                              box: box, imageSize: imageSize)
    }
    
    private func createMaskImage(mask: [Float], protoSize: CGSize, box: CGRect, imageSize: CGSize) -> UIImage? {
        // Create binary mask image
        let width = Int(imageSize.width)
        let height = Int(imageSize.height)
        
        // Scale prototype mask to box size
        let scaledMask = scaleMaskToBox(mask: mask, protoSize: protoSize, box: box, imageSize: imageSize)
        
        // Convert to UIImage
        return maskToUIImage(mask: scaledMask, size: imageSize)
    }
    
    private func scaleMaskToBox(mask: [Float], protoSize: CGSize, box: CGRect, imageSize: CGSize) -> [Float] {
        let width = Int(imageSize.width)
        let height = Int(imageSize.height)
        var scaledMask = [Float](repeating: 0, count: width * height)
        
        // Calculate scaling factors
        let scaleX = Float(box.width / protoSize.width)
        let scaleY = Float(box.height / protoSize.height)
        
        // Scale and translate mask
        for y in 0..<Int(protoSize.height) {
            for x in 0..<Int(protoSize.width) {
                let srcIdx = y * Int(protoSize.width) + x
                let srcX = Float(x) * scaleX + Float(box.minX)
                let srcY = Float(y) * scaleY + Float(box.minY)
                
                // Bilinear interpolation
                let x0 = Int(srcX)
                let y0 = Int(srcY)
                let x1 = min(x0 + 1, width - 1)
                let y1 = min(y0 + 1, height - 1)
                
                let wx = srcX - Float(x0)
                let wy = srcY - Float(y0)
                
                let idx00 = y0 * width + x0
                let idx01 = y0 * width + x1
                let idx10 = y1 * width + x0
                let idx11 = y1 * width + x1
                
                let val = mask[srcIdx]
                scaledMask[idx00] += val * (1 - wx) * (1 - wy)
                scaledMask[idx01] += val * wx * (1 - wy)
                scaledMask[idx10] += val * (1 - wx) * wy
                scaledMask[idx11] += val * wx * wy
            }
        }
        
        return scaledMask
    }
    
    private func maskToUIImage(mask: [Float], size: CGSize) -> UIImage? {
        let width = Int(size.width)
        let height = Int(size.height)
        
        // Create bitmap context
        let bitsPerComponent = 8
        let bytesPerRow = width * 4
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let context = CGContext(data: nil,
                                    width: width,
                                    height: height,
                                    bitsPerComponent: bitsPerComponent,
                                    bytesPerRow: bytesPerRow,
                                    space: colorSpace,
                                    bitmapInfo: bitmapInfo.rawValue) else {
            return nil
        }
        
        // Convert mask to RGBA
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                let rgbaIdx = idx * 4
                let val = UInt8(mask[idx] * 255)
                rgba[rgbaIdx] = 0     // R
                rgba[rgbaIdx + 1] = 0 // G
                rgba[rgbaIdx + 2] = 0 // B
                rgba[rgbaIdx + 3] = val // A
            }
        }
        
        // Create image from context
        context.data?.copyMemory(from: rgba, byteCount: rgba.count)
        guard let cgImage = context.makeImage() else {
            return nil
        }
        
        return UIImage(cgImage: cgImage)
    }
}

// Extension for UIImage to CVPixelBuffer conversion
extension UIImage {
    func toCVPixelBuffer(size: CGSize) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                       Int(size.width),
                                       Int(size.height),
                                       kCVPixelFormatType_32ARGB,
                                       nil,
                                       &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                              width: Int(size.width),
                              height: Int(size.height),
                              bitsPerComponent: 8,
                              bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        context?.draw(self.cgImage!, in: CGRect(origin: .zero, size: size))
        
        return buffer
    }
} 
