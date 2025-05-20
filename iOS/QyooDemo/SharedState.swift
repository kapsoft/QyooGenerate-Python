//
//  SharedState.swift
//  QyooDemo
//
//  Created by Jeffrey Berthiaume on 5/16/25.
//

import SwiftUI

class SharedState: ObservableObject {
    @Published var thumb: UIImage? = nil
    @Published var detections: [Detection] = []
    @Published var wantDump = false
}
