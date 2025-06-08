//
//  Overlay.swift
//  QyooDemo
//
//  Created by Jeffrey Berthiaume on 5/6/25.
//

import SwiftUI
import AudioToolbox

struct Overlay: View {
    @ObservedObject var shared: SharedState

    var body: some View {
        ZStack {
            // Display detections and masks
            ForEach(shared.detections) { detection in
                // Draw mask
                Image(detection.mask, scale: 1.0, label: Text("Mask"))
                    .resizable()
                    .interpolation(.none)
                    .opacity(0.5)
                
                // Draw bounding box
                Rectangle()
                    .stroke(Color.green, lineWidth: 2)
                    .frame(width: detection.rect.width,
                           height: detection.rect.height)
                    .position(x: detection.rect.midX,
                             y: detection.rect.midY)
            }
            
            // Confidence display
            if let best = shared.detections.max(by: { $0.confidence < $1.confidence }) {
                Text(String(format: "Confidence: %.2f", best.confidence))
                    .font(.title3.bold())
                    .padding(8)
                    .background(.ultraThinMaterial,
                                in: RoundedRectangle(cornerRadius: 12))
                    .position(x: UIScreen.main.bounds.width - 100,
                             y: 50)
            }
            
            // Debug button
            VStack {
                HStack {
                    Spacer()
                    Button {
                        print("Debug info:")
                        for det in shared.detections {
                            print("Rect:", det.rect)
                            print("Confidence:", det.confidence)
                        }
                        shared.wantDump = true
                        AudioServicesPlaySystemSound(kSystemSoundID_Vibrate)
                    } label: {
                        Text("Debug")
                    }
                    .padding(80)
                    .font(.title)
                    .buttonStyle(.borderedProminent)
                }
                
                Spacer()
            }
        }
        // 1) stretch to fill the entire screen
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        // 2) give it a transparent hit-testable background
        .background(Color.clear)
        // 3) make sure it goes full-screen under any safe areas
        .ignoresSafeArea()
    }
}

#Preview {
    Overlay(shared: SharedState())
}
