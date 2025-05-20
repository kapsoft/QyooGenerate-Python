//
//  Detection.swift
//  QyooDemo
//
//  Created by Jeffrey Berthiaume on 5/16/25.
//

import UIKit

struct Detection: Identifiable {
    let id = UUID()
    let rect: CGRect
    let confidence: Float
    let mask: CGImage
}
