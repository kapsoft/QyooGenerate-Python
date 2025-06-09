import UIKit
import CoreVideo
import CoreImage

extension UIImage {
    /// The ONE working method for converting UIImage to CVPixelBuffer
    /// Used by both the app and tests
    func toCVPixelBuffer(size: CGSize) -> CVPixelBuffer? {
        let canvasSize = Int(size.width)  // Assuming square for now
        
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
