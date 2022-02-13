package com.baidu.paddle.lite.demo.ocr

import android.graphics.Bitmap
import android.util.Log
import com.baidu.paddle.lite.demo.ocr.OcrResultModel
import kotlin.Throws
import com.baidu.paddle.lite.demo.ocr.OCRPredictorNative
import java.lang.RuntimeException
import java.util.ArrayList
import java.util.concurrent.atomic.AtomicBoolean

class OCRPredictorNative(private val config: Config) {
    private var nativePointer: Long = 0
    fun runImage(
        inputData: FloatArray,
        width: Int,
        height: Int,
        channels: Int,
        originalImage: Bitmap?
    ): ArrayList<OcrResultModel> {
        Log.i(
            "OCRPredictorNative",
            "begin to run image " + inputData.size + " " + width + " " + height
        )
        val dims =
            floatArrayOf(1f, channels.toFloat(), height.toFloat(), width.toFloat())
        val rawResults =
            forward(nativePointer, inputData, dims, originalImage)
        return postprocess(rawResults)
    }

    class Config {
        var cpuThreadNum = 0
        var cpuPower: String? = null
        var detModelFilename: String? = null
        var recModelFilename: String? = null
        var clsModelFilename: String? = null
    }

    fun destory() {
        if (nativePointer != 0L) {
            release(nativePointer)
            nativePointer = 0
        }
    }

    protected external fun init(
        detModelPath: String?,
        recModelPath: String?,
        clsModelPath: String?,
        threadNum: Int,
        cpuMode: String?
    ): Long

    protected external fun forward(
        pointer: Long,
        buf: FloatArray?,
        ddims: FloatArray?,
        originalImage: Bitmap?
    ): FloatArray

    protected external fun release(pointer: Long)
    private fun postprocess(raw: FloatArray): ArrayList<OcrResultModel> {
        val results = ArrayList<OcrResultModel>()
        var begin = 0
        while (begin < raw.size) {
            val point_num = Math.round(raw[begin])
            val word_num = Math.round(raw[begin + 1])
            val model = parse(raw, begin + 2, point_num, word_num)
            begin += 2 + 1 + point_num * 2 + word_num
            results.add(model)
        }
        return results
    }

    private fun parse(raw: FloatArray, begin: Int, pointNum: Int, wordNum: Int): OcrResultModel {
        var current = begin
        val model = OcrResultModel()
        model.confidence = raw[current]
        current++
        for (i in 0 until pointNum) {
            model.addPoints(Math.round(raw[current + i * 2]), Math.round(raw[current + i * 2 + 1]))
        }
        current += pointNum * 2
        for (i in 0 until wordNum) {
            val index = Math.round(raw[current + i])
            model.addWordIndex(index)
        }
        Log.i("OCRPredictorNative", "word finished $wordNum")
        return model
    }

    companion object {
        private val isSOLoaded = AtomicBoolean()
        @Throws(RuntimeException::class)
        fun loadLibrary() {
            if (!isSOLoaded.get() && isSOLoaded.compareAndSet(false, true)) {
                try {
                    System.loadLibrary("Native")
                } catch (e: Throwable) {
                    val exception = RuntimeException(
                        "Load libNative.so failed, please check it exists in apk file.", e
                    )
                    throw exception
                }
            }
        }
    }

    init {
        loadLibrary()
        nativePointer = init(
            config.detModelFilename, config.recModelFilename, config.clsModelFilename,
            config.cpuThreadNum, config.cpuPower
        )
        Log.i("OCRPredictorNative", "load success $nativePointer")
    }
}