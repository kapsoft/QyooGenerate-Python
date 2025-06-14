//
//  SegmentationCoordinator.swift
//  QyooDemo
//
//  Created by Jeffrey Berthiaume on 5/16/25.
//


import Vision
import AVFoundation
import SwiftUI

final class SegmentationCoordinator: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let shared: SharedState
    private let session = AVCaptureSession()
    private let visionQueue = DispatchQueue(label: "vision.q")
    private let ciContext = CIContext()
    private let detector: QyooDetector
    
    /// Hooked up in makeUIView
    weak var previewLayer: AVCaptureVideoPreviewLayer?
    
    init(shared: SharedState) {
        self.shared = shared
        self.detector = QyooDetector()
        super.init()
    }
    
    func startSession() {
        
        // Skip camera setup during testing
           if ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil {
               print("🧪 Test environment detected - skipping camera session")
               return
           }
        
        guard previewLayer != nil else {
            fatalError("❌ previewLayer must be set before calling startSession()")
        }
        
        session.sessionPreset = .hd1280x720
        
        // Safe camera detection with fallbacks
        var camera: AVCaptureDevice?
        
        // Try back camera first
        camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
        
        if camera == nil {
            print("⚠️ Back camera not available, trying front camera...")
            camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
        }
        
        if camera == nil {
            print("⚠️ Wide angle camera not available, trying default camera...")
            camera = AVCaptureDevice.default(for: .video)
        }
        
        guard let cam = camera else {
            #if targetEnvironment(simulator)
                print("📱 Running in simulator - camera not available")
                fatalError("Camera not available in iOS Simulator. Please test on a physical device.")
            #else
                fatalError("❌ No camera available on this device")
            #endif
        }
        
        print("✅ Using camera: \(cam.localizedName) at position: \(cam.position)")
        
        do {
            let input = try AVCaptureDeviceInput(device: cam)
            
            if session.canAddInput(input) {
                session.addInput(input)
            } else {
                fatalError("❌ Cannot add camera input to session")
            }
            
            let out = AVCaptureVideoDataOutput()
            out.setSampleBufferDelegate(self, queue: visionQueue)
            
            if session.canAddOutput(out) {
                session.addOutput(out)
            } else {
                fatalError("❌ Cannot add video output to session")
            }
            
            previewLayer!.session = session
            previewLayer!.videoGravity = .resizeAspectFill
            
            DispatchQueue.global(qos: .background).async {
                self.session.startRunning()
                print("📹 Camera session started")
            }
            
        } catch {
            fatalError("❌ Error setting up camera input: \(error.localizedDescription)")
        }
    }
    
    // MARK: – AVCaptureVideoDataOutputSampleBufferDelegate
    func captureOutput(_ _: AVCaptureOutput,
                       didOutput sample: CMSampleBuffer,
                       from _: AVCaptureConnection)
    {
        guard let pb = CMSampleBufferGetImageBuffer(sample) else { return }
        
        // Convert CMSampleBuffer to UIImage
        let ciImage = CIImage(cvPixelBuffer: pb)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        let image = UIImage(cgImage: cgImage)
        
        // Process image with our detector
        if let result = detector.detect(image: image) {
            handleDetection(result: result, imageSize: image.size)
        }
    }
    
    // MARK: – Detection handling
    // Change the method signature:
    private func handleDetection(result: QyooDetector.MaskResult, imageSize: CGSize) {
        guard let pl = previewLayer else { return }
        
        // Convert mask to CGImage
        guard let cgMask = result.mask.cgImage else { return }
        
        // Use result.box directly instead of calculating from mask
        // This is actually better because we already have the box!
        
        // Convert box coords to view coords
        let W = pl.bounds.width, H = pl.bounds.height
        let scaleX = W / imageSize.width
        let scaleY = H / imageSize.height
        
        let viewRect = CGRect(
            x: result.box.origin.x * scaleX,
            y: (imageSize.height - result.box.maxY) * scaleY,  // flip Y
            width: result.box.width * scaleX,
            height: result.box.height * scaleY
        )
        
        let det = Detection(
            rect: viewRect,
            confidence: result.confidence,
            mask: cgMask
        )
        
        DispatchQueue.main.async {
            self.shared.detections = [det]
        }
    }
}
