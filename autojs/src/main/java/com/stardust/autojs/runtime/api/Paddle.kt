package com.stardust.autojs.runtime.api

import android.os.Looper
import android.util.Log
import com.baidu.paddle.lite.demo.ocr.OcrResult
import com.baidu.paddle.lite.demo.ocr.Predictor
import com.stardust.app.GlobalAppContext
import com.stardust.autojs.core.image.ImageWrapper

class Paddle {
    private val mPredictor = Predictor()

    fun initOcr(useSlim: Boolean): Boolean {
        if (!mPredictor.modelLoaded) {
            mPredictor.init(GlobalAppContext.get(), useSlim)
        }
        return mPredictor.modelLoaded
    }

    fun releaseOcr() {
        mPredictor.release()
    }

    @JvmOverloads
    fun ocr(image: ImageWrapper?, cpuThreadNum: Int = 4, useSlim: Boolean = true): List<OcrResult> {
        if (image == null) {
            return emptyList()
        }
        val bitmap = image.bitmap
        if (bitmap == null || bitmap.isRecycled) {
            return emptyList()
        }
        initOcr(useSlim)
        return mPredictor.ocr(bitmap, cpuThreadNum)
    }

    @JvmOverloads
    fun ocrText(
        image: ImageWrapper?,
        cpuThreadNum: Int = 4,
        useSlim: Boolean = true
    ): Array<String?> {
        val words_result = ocr(image, cpuThreadNum, useSlim)
        val outputResult = arrayOfNulls<String>(words_result.size)
        for (i in words_result.indices) {
            outputResult[i] = words_result[i].words
            Log.i("outputResult", outputResult[i].toString()) // show LOG in Logcat panel
        }
        return outputResult
    }
}
