import CoreML
import Vision
import UIKit
import Accelerate

class QyooDetector {
    private let model: MLModel
    private let imageSize: CGFloat = 512
    
    // Make these accessible for testing
    struct Detection {
        let box: CGRect
        let confidence: Float
        let maskCoefficients: [Float]
    }
    
    struct MaskResult {
        let mask: UIImage
        let confidence: Float
        let box: CGRect
    }
    
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
                fatalError("Failed to find model in bundle")
            }
        } catch {
            print("Error loading model: \(error)")
            print("Error details: \(error.localizedDescription)")
            fatalError("Failed to load model: \(error)")
        }
    }
    
    // MARK: - Public Methods
    
    func detect(image: UIImage) -> MaskResult? {
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
            return processMask(detections: detections, prototypes: prototypes, originalImage: image)
            
        } catch {
            print("Prediction error: \(error)")
            return nil
        }
    }
    
    // MARK: - Testable Methods (internal for testing)
    
    internal func findBestDetection(in detections: MLMultiArray, threshold: Float = 0.3) -> Detection? {
        let numDetections = detections.shape[2].intValue // 5376
        
        var bestConfidence: Float = threshold
        var bestDetection: Detection?
        
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
                
                // Convert to CGRect (normalized coordinates 0-512)
                let box = CGRect(
                    x: CGFloat(xc - w/2),
                    y: CGFloat(yc - h/2),
                    width: CGFloat(w),
                    height: CGFloat(h)
                )
                
                // Extract 32 mask coefficients
                var maskCoeffs: [Float] = []
                for j in 5..<37 {
                    maskCoeffs.append(detections[[0, j, i] as [NSNumber]].floatValue)
                }
                
                bestDetection = Detection(
                    box: box,
                    confidence: confidence,
                    maskCoefficients: maskCoeffs
                )
            }
        }
        
        return bestDetection
    }
    
    internal func generateMaskArray(coefficients: [Float], prototypes: MLMultiArray) -> [Float] {
        // Prototypes shape: [1, 32, 128, 128]
        let protoH = 128
        let protoW = 128
        
        // Create mask by combining prototypes with coefficients
        var mask = [Float](repeating: 0, count: protoH * protoW)
        
        // Linear combination of prototypes
        for p in 0..<32 {
            let coeff = coefficients[p]
            
            // ADD THIS CHECK
            if coeff < 0 {
                continue  // Skip negative coefficients
            }
            
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
        
        return mask
    }
    
    // MARK: - Private Methods
    
    private func processMask(detections: MLMultiArray, prototypes: MLMultiArray, originalImage: UIImage) -> MaskResult? {
        // Find best detection
        guard let detection = findBestDetection(in: detections) else {
            return nil
        }
        
        // Generate mask array
        let maskArray = generateMaskArray(
            coefficients: detection.maskCoefficients,
            prototypes: prototypes
        )
        
        // Scale box to original image size
        let scaledBox = CGRect(
            x: detection.box.origin.x * originalImage.size.width / imageSize,
            y: detection.box.origin.y * originalImage.size.height / imageSize,
            width: detection.box.width * originalImage.size.width / imageSize,
            height: detection.box.height * originalImage.size.height / imageSize
        )
        
        // Create mask image
        guard let maskImage = createMaskImage(
            mask: maskArray,
            protoSize: CGSize(width: 128, height: 128),
            box: scaledBox,
            imageSize: originalImage.size
        ) else {
            return nil
        }
        
        return MaskResult(
            mask: maskImage,
            confidence: detection.confidence,
            box: scaledBox
        )
    }
    
    private func createMaskImage(mask: [Float], protoSize: CGSize, box: CGRect, imageSize: CGSize) -> UIImage? {
        // Scale prototype mask to box size
        let scaledMask = scaleMaskToBox(mask: mask, protoSize: protoSize, box: box, imageSize: imageSize)
        
        // Convert to UIImage
        return maskToUIImage(mask: scaledMask, size: imageSize)
    }
    
    // Replace the scaleMaskToBox method in QyooDetector.swift with this fixed version:

    private func scaleMaskToBox(mask: [Float], protoSize: CGSize, box: CGRect, imageSize: CGSize) -> [Float] {
        let width = Int(imageSize.width)
        let height = Int(imageSize.height)
        var scaledMask = [Float](repeating: 0, count: width * height)
        
        let protoW = Int(protoSize.width)   // 128
        let protoH = Int(protoSize.height)  // 128
        
        // For each pixel in the output mask within the box
        let boxMinX = Int(max(0, box.minX))
        let boxMaxX = Int(min(Float(width), Float(box.maxX)))
        let boxMinY = Int(max(0, box.minY))
        let boxMaxY = Int(min(Float(height), Float(box.maxY)))
        
        for y in boxMinY..<boxMaxY {
            for x in boxMinX..<boxMaxX {
                // Map back to prototype coordinates
                let protoX = Float(x - boxMinX) * Float(protoW) / Float(box.width)
                let protoY = Float(y - boxMinY) * Float(protoH) / Float(box.height)
                
                // Bilinear interpolation
                let x0 = Int(protoX)
                let y0 = Int(protoY)
                let x1 = min(x0 + 1, protoW - 1)
                let y1 = min(y0 + 1, protoH - 1)
                
                let wx = protoX - Float(x0)
                let wy = protoY - Float(y0)
                
                // Bounds check
                if x0 >= 0 && x1 < protoW && y0 >= 0 && y1 < protoH {
                    // Get the four surrounding values
                    let v00 = mask[y0 * protoW + x0]
                    let v01 = mask[y0 * protoW + x1]
                    let v10 = mask[y1 * protoW + x0]
                    let v11 = mask[y1 * protoW + x1]
                    
                    // Interpolate
                    let value = v00 * (1 - wx) * (1 - wy) +
                               v01 * wx * (1 - wy) +
                               v10 * (1 - wx) * wy +
                               v11 * wx * wy
                    
                    let dstIdx = y * width + x
                    scaledMask[dstIdx] = value
                }
            }
        }
        
        return scaledMask
    }
    
    // Also update maskToUIImage to better visualize low values:
    private func maskToUIImage(mask: [Float], size: CGSize) -> UIImage? {
        let width = Int(size.width)
        let height = Int(size.height)
        
        // Find the actual range of values for better visualization
        let nonZeroMask = mask.filter { $0 > 0.01 }
        let minVal = nonZeroMask.min() ?? 0
        let maxVal = nonZeroMask.max() ?? 1
        
        print("Mask statistics: min=\(minVal), max=\(maxVal), non-zero pixels=\(nonZeroMask.count)")
        
        // Create bitmap context
        let bitsPerComponent = 8
        let bytesPerRow = width * 4
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            return nil
        }
        
        // Convert mask to RGBA with better normalization
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                let rgbaIdx = idx * 4
                
                let maskValue = mask[idx]
                
                // Apply threshold and normalize
                if maskValue > 0.1 {  // Adjust threshold as needed
                    // Enhance the visualization
                    let normalizedValue = (maskValue - 0.3) / (1.0 - 0.3)
                    let val = UInt8(min(max(normalizedValue, 0.0), 1.0) * 255)
                    
                    rgba[rgbaIdx] = 255     // R
                    rgba[rgbaIdx + 1] = 255 // G
                    rgba[rgbaIdx + 2] = 255 // B
                    rgba[rgbaIdx + 3] = val // A
                } else {
                    // Fully transparent
                    rgba[rgbaIdx] = 0
                    rgba[rgbaIdx + 1] = 0
                    rgba[rgbaIdx + 2] = 0
                    rgba[rgbaIdx + 3] = 0
                }
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
