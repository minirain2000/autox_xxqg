package com.baidu.paddle.lite.demo.ocr

import android.graphics.Point
import java.util.ArrayList

class OcrResultModel {
    private val points: MutableList<Point>
    private val wordIndex: MutableList<Int>
    var label: String? = null
    var confidence = 0f
    fun addPoints(x: Int, y: Int) {
        val point = Point(x, y)
        points.add(point)
    }

    fun addWordIndex(index: Int) {
        wordIndex.add(index)
    }

    fun getPoints(): List<Point> {
        return points
    }

    fun getWordIndex(): List<Int> {
        return wordIndex
    }

    init {
        points = ArrayList()
        wordIndex = ArrayList()
    }
}