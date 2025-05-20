//
//  CameraPreview.swift
//  QyooDemo
//
//  Core ML direct inference for YOLOv8‑seg without NMS.
//  Shows red bbox + translucent green mask.
//
//  Created by Jeffrey Berthiaume on 5/14/25.
//

import SwiftUI

struct CameraPreview: UIViewRepresentable {
    @ObservedObject var shared: SharedState
    
    func makeCoordinator() -> SegmentationCoordinator {
        SegmentationCoordinator(shared: shared)
    }
    
    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.configureForBackground()
        
        // **Use the existing coordinator** rather than making a new one:
        let coord = context.coordinator
        coord.previewLayer = view.videoPreviewLayer
        coord.startSession()
        
        return view
    }
    
    func updateUIView(_ uiView: PreviewView, context: Context) {
        // nothing
    }
}
