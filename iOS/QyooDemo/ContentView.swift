//
//  ContentView.swift
//  QyooDemo
//
//  Created by Jeffrey Berthiaume on 5/6/25.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var shared = SharedState()
    @State private var dets: [Detection] = []

    var body: some View {
        ZStack {
            CameraPreview(shared: shared)
                .frame(height: 600)
                .allowsHitTesting(false)
            
            Overlay(shared: shared)
                .zIndex(1)
                .ignoresSafeArea()
        }
    }
}
