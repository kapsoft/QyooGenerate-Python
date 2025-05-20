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
    
    // The VNCoreMLRequest dedicated to your pure-segmentation model
    private lazy var visionRequest: VNCoreMLRequest = {
        let mlmodel = try! best(configuration: .init()).model
        
        print("=== model outputs ===")
        for (name, desc) in mlmodel.modelDescription.outputDescriptionsByName {
            print("•", name,
                  "multiArray:",   desc.multiArrayConstraint != nil,
                  "image:",        desc.imageConstraint      != nil,
                  desc.multiArrayConstraint?.shape ?? [] ,
                  desc.imageConstraint?.pixelsWide ?? 0,
                  desc.imageConstraint?.pixelsHigh ?? 0)
        }
        
        let vnModel  = try! VNCoreMLModel(for: mlmodel)
        let req = VNCoreMLRequest(model: vnModel, completionHandler: handleSegmentation)
        req.imageCropAndScaleOption = .scaleFill
        return req
    }()
    
    /// Hooked up in makeUIView
    weak var previewLayer: AVCaptureVideoPreviewLayer?
    
    init(shared: SharedState) {
        self.shared = shared
        super.init()
    }
    
    func startSession() {
        guard previewLayer != nil else {
            fatalError("❌ previewLayer must be set before calling startSession()")
        }
        
        session.sessionPreset = .hd1280x720
        let cam = AVCaptureDevice.default(.builtInWideAngleCamera,
                                          for: .video, position: .back)!
        session.addInput( try! AVCaptureDeviceInput(device: cam) )
        
        let out = AVCaptureVideoDataOutput()
        out.setSampleBufferDelegate(self, queue: visionQueue)
        session.addOutput(out)
        
        previewLayer!.session = session
        previewLayer!.videoGravity = .resizeAspectFill
        
        DispatchQueue.global(qos: .background).async {
            self.session.startRunning()
        }
    }
    
    // MARK: – AVCaptureVideoDataOutputSampleBufferDelegate
    func captureOutput(_ _: AVCaptureOutput,
                       didOutput sample: CMSampleBuffer,
                       from _: AVCaptureConnection)
    {
        guard let pb = CMSampleBufferGetImageBuffer(sample) else { return }
        let handler = VNImageRequestHandler(cvPixelBuffer: pb,
                                            orientation: .right,
                                            options: [:])
        try? handler.perform([visionRequest])
    }
    
    // MARK: – Vision completion
    private func handleSegmentation(request: VNRequest, error: Error?) {
        guard
            let pixObs = request.results?.first as? VNPixelBufferObservation,
            let pl     = previewLayer
        else { return }
        
        let maskPB = pixObs.pixelBuffer
        let ciMask = CIImage(cvPixelBuffer: maskPB)
        guard let cgMask = ciContext.createCGImage(ciMask,
                                                   from: ciMask.extent)
        else { return }
        
        // find the minimal bounding rect in the mask
        let w = cgMask.width, h = cgMask.height
        guard let data = cgMask.dataProvider?.data,
              let ptr  = CFDataGetBytePtr(data)
        else { return }
        
        var minX = w, minY = h, maxX = 0, maxY = 0
        for y in 0..<h {
            for x in 0..<w where ptr[y*w + x] > 128 {
                minX = min(minX, x); minY = min(minY, y)
                maxX = max(maxX, x); maxY = max(maxY, y)
            }
        }
        
        // convert mask coords → view coords
        let W = pl.bounds.width, H = pl.bounds.height
        let scaleX = W / CGFloat(w), scaleY = H / CGFloat(h)
        let viewRect = CGRect(
            x: CGFloat(minX) * scaleX,
            y: CGFloat(h - maxY) * scaleY,      // flip Y
            width:  CGFloat(maxX - minX) * scaleX,
            height: CGFloat(maxY - minY) * scaleY
        )
        
        let det = Detection(rect: viewRect,
                            confidence: pixObs.confidence,
                            mask: cgMask)
        
        DispatchQueue.main.async {
            self.shared.detections = [det]
        }
        
        DispatchQueue.main.async {
            if self.shared.wantDump {
                self.shared.wantDump = false
                print("rect:", det.rect, "conf:", det.confidence)
            }
            //draw(det)
        }
    }
}
