//
//  PreviewView.swift
//  QyooDemo
//
//  Created by Jeffrey Berthiaume on 5/16/25.
//

import UIKit
import AVFoundation

class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    func configureForBackground() {
        isUserInteractionEnabled = false
        layer.zPosition = -1
    }
}
