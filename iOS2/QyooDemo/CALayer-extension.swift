//
//  CALayer-extension.swift
//  QyooDemo
//
//  Created by Jeffrey Berthiaume on 5/7/25.
//

import UIKit

extension CALayer {
    /// Snapshot of *visible* pixels in `rect` (preview layer space)
    func snapshot(rect: CGRect, scale: CGFloat = 1) -> CGImage? {
        let fmt = UIGraphicsImageRendererFormat()
        fmt.scale = scale
        return UIGraphicsImageRenderer(size: rect.size, format: fmt)
            .image { ctx in
                ctx.cgContext.translateBy(x: -rect.origin.x, y: -rect.origin.y)
                render(in: ctx.cgContext)
            }.cgImage
    }
}

extension UIColor {
    static func random() -> UIColor {
        return UIColor(
           red:   .random(),
           green: .random(),
           blue:  .random(),
           alpha: 1.0
        )
    }
}

extension CGFloat {
    static func random() -> CGFloat {
        return CGFloat(arc4random()) / CGFloat(UInt32.max)
    }
}
