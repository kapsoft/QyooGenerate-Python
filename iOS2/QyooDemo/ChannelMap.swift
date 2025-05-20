//
//  ChannelMap.swift
//  QyooDemo
//
//  Created by Jeffrey Berthiaume on 5/14/25.
//

import CoreML

// Detects channel roles the first time it sees a prediction tensor.
// Caches the result so it runs only once.
struct ChannelMap {
    var idxCx  = -1, idxCy = -1, idxW = -1, idxH = -1
    var idxObj = -1, idxCls = -1, idxCoeff0 = -1

    mutating func analyse(pred: MLMultiArray) {
        guard idxCx == -1 else { return }          // already done

        let c = pred.shape[1].intValue             // 37
        let n = pred.shape[2].intValue             // 5376
        let stride = pred.strides[1].intValue
        let p = pred.dataPointer.bindMemory(to: Float32.self,
                                            capacity: pred.count)

        // sample first 500 anchors
        var mins = [Float](repeating: .greatestFiniteMagnitude, count: c)
        var maxs = [Float](repeating: -.greatestFiniteMagnitude, count: c)

        for i in 0..<min(500,n) {
            for ch in 0..<c {
                let v = p[ch*stride + i]
                mins[ch] = min(mins[ch], v)
                maxs[ch] = max(maxs[ch], v)
            }
        }

        // heuristics -------------------------------------------------------
        // coeffs: roughly [-5 … +5] but can be any sign, never > imgSide
        // bbox  : in   [0 … imgSide]       and max >  30
        // obj   : in   [0 … 1]
        // cls   : idem
        for ch in 0..<c {
            switch maxs[ch] {
            case 300...:                      // definitely bbox (≥ imgSide/2)
                if idxW == -1      { idxW  = ch; continue }
                if idxH == -1      { idxH  = ch; continue }
                if idxCx == -1     { idxCx = ch; continue }
                if idxCy == -1     { idxCy = ch; continue }

            case 1.05...5:                   // still could be bbox on tiny anchors
                if idxCx == -1     { idxCx = ch; continue }
                if idxCy == -1     { idxCy = ch; continue }
                if idxW  == -1     { idxW  = ch; continue }
                if idxH  == -1     { idxH  = ch; continue }

            case 0.9...1.05:                 // objectness / class
                if idxObj == -1   { idxObj = ch; continue }
                if idxCls == -1   { idxCls = ch; continue }

            default:
                if idxCoeff0 == -1 { idxCoeff0 = ch }   // first coeff channel
            }
        }

        print("📌 Channel map:", [idxCoeff0,idxCx,idxCy,idxW,idxH,idxObj,idxCls])
    }
}
