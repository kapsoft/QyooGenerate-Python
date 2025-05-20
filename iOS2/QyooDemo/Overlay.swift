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
            /*
            GeometryReader { _ in            // fills the parent
                VStack(spacing: 8) {
                    // ---------- thumbnail ----------
                    if let img = shared.thumb {
                        Image(uiImage: img)
                            .resizable()
                            .interpolation(.none)
                            .frame(width: 128, height: 128)
                            .border(.white.opacity(0.8), width: 1)
                    }
                    // ---------- confidence ----------
                    if let best = shared.detections.max(by: { $0.confidence < $1.confidence }) {
                        Text(String(format: "conf %.2f", best.confidence))
                            .font(.title3.bold())
                            .padding(8)
                            .background(.ultraThinMaterial,
                                        in: RoundedRectangle(cornerRadius: 12))
                    }
                }
                .frame(maxWidth: .infinity,
                       maxHeight: .infinity,
                       alignment: .bottom)
                .padding(.bottom, 40)
            }
            .allowsHitTesting(false)
            */
            
            
            VStack {
                
                HStack {
                    Spacer()
                    Button {
                        print("twoop!")
                        shared.wantDump = true
                        AudioServicesPlaySystemSound(kSystemSoundID_Vibrate)
                    } label: {
                        Text("clicky")
                    }
                    .padding(80)
                    .font(.title)
                    .buttonStyle(.borderedProminent)
                }
                
                Spacer()
            }
            
//            Color.clear
//                .ignoresSafeArea()          // ← ensure it really fills the screen
//                .contentShape(Rectangle())
//                .allowsHitTesting(true)     // ← explicit
//                .onTapGesture {
//                    print("boop!")          // should appear every tap
//                    shared.wantDump = true
//                }
//                .zIndex(9999)
            
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
