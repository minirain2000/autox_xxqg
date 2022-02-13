package com.baidu.paddle.lite.demo.ocr

import android.graphics.Rect
import com.baidu.paddle.lite.demo.ocr.OcrResult.RectLocation
import com.baidu.paddle.lite.demo.ocr.OcrResult

class OcrResult : Comparable<Any?> {
    var confidence = 0f
    var preprocessTime = 0f
    var inferenceTime = 0f
    var words: String? = null
    var bounds: Rect? = null
    var location: RectLocation? = null
    override fun compareTo(o: Any?): Int {
        val s = o as OcrResult?
        val deviation = 9
        return if (Math.abs(bounds!!.bottom - s!!.bounds!!.bottom) <= deviation) {
            bounds!!.left - s.bounds!!.left
        } else {
            bounds!!.bottom - s.bounds!!.bottom
        }
    }

    class RectLocation(var left: Int, var top: Int, var width: Int, var height: Int)
}