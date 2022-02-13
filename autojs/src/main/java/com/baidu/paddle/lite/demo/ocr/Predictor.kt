package com.baidu.paddle.lite.demo.ocr

import android.content.Context
import android.graphics.*
import android.util.Log
import kotlin.jvm.Volatile
import com.baidu.paddle.lite.demo.ocr.OcrResult.RectLocation
import java.io.File
import java.lang.Exception
import java.lang.StringBuilder
import java.util.*

class Predictor {
    var modelLoaded = false
    var useSlim = true
    protected var mPaddlePredictorNative: OCRPredictorNative? = null
    var warmupIterNum = 1
    var inferIterNum = 1
    var cpuThreadNum = 4
    var cpuPowerMode = "LITE_POWER_HIGH"
    var modelPath = ""
    var modelName = ""
    protected var paddlePredictor: OCRPredictorNative? = null
    protected var inferenceTime = 0f

    // Only for object detection
    protected var wordLabels = Vector<String>()
    protected var inputColorFormat = "BGR"
    protected var inputShape = longArrayOf(1, 3, 960)
    protected var inputMean = floatArrayOf(0.485f, 0.456f, 0.406f)
    protected var inputStd = floatArrayOf(0.229f, 0.224f, 0.225f)
    protected var scoreThreshold = 0.1f
    protected var inputImage0: Bitmap? = null
    protected var outputImage: Bitmap? = null

    @Volatile
    protected var outputResult = ""
    protected var preprocessTime = 0f
    protected var postprocessTime = 0f
    fun init(appCtx: Context, useSlim: Boolean): Boolean {
        if (!modelLoaded || this.useSlim != useSlim) {
            loadLabel(appCtx, "labels/ppocr_keys_v1.txt")
            if (useSlim) {
                loadModel(appCtx, "models/ocr_v2_for_cpu(slim)", 4, "LITE_POWER_HIGH")
            } else {
                loadModel(appCtx, "models/ocr_v2_for_cpu", 4, "LITE_POWER_HIGH")
            }
        }
        modelLoaded = true
        this.useSlim = useSlim
        Log.i(TAG, "modelLoaded: " + modelLoaded)
        return modelLoaded
    }

    fun init(appCtx: Context, modelPath: String, labelPath: String?): Boolean {
        if (!modelLoaded) {
            loadModel(appCtx, modelPath, cpuThreadNum, cpuPowerMode)
            loadLabel(appCtx, labelPath)
            modelLoaded = true
        }
        Log.i(TAG, "PaddleOCR modelLoaded: $modelLoaded")
        return modelLoaded
    }

    fun init(
        appCtx: Context,
        modelPath: String,
        labelPath: String?,
        cpuThreadNum: Int,
        cpuPowerMode: String?,
        inputColorFormat: String,
        inputShape: LongArray,
        inputMean: FloatArray,
        inputStd: FloatArray,
        scoreThreshold: Float
    ): Boolean {
        if (inputShape.size != 3) {
            Log.e(TAG, "Size of input shape should be: 3")
            return false
        }
        if (inputMean.size.toLong() != inputShape[1]) {
            Log.e(
                TAG, "Size of input mean should be: " + java.lang.Long.toString(
                    inputShape[1]
                )
            )
            return false
        }
        if (inputStd.size.toLong() != inputShape[1]) {
            Log.e(
                TAG, "Size of input std should be: " + java.lang.Long.toString(
                    inputShape[1]
                )
            )
            return false
        }
        if (inputShape[0] != 1.toLong()) {
            Log.e(
                TAG,
                "Only one batch is supported in the image classification demo, you can use any batch size in " +
                        "your Apps!"
            )
            return false
        }
        if (inputShape[1] != 1.toLong() && inputShape[1] != 3.toLong()) {
            Log.e(
                TAG,
                "Only one/three channels are supported in the image classification demo, you can use any " +
                        "channel size in your Apps!"
            )
            return false
        }
        if (!inputColorFormat.equals("BGR", ignoreCase = true)) {
            Log.e(TAG, "Only  BGR color format is supported.")
            return false
        }
        val modelLoaded = init(appCtx, modelPath, labelPath)
        if (!modelLoaded) {
            return false
        }
        this.inputColorFormat = inputColorFormat
        this.inputShape = inputShape
        this.inputMean = inputMean
        this.inputStd = inputStd
        this.scoreThreshold = scoreThreshold
        return true
    }

    protected fun loadModel(
        appCtx: Context,
        modelPath: String,
        cpuThreadNum: Int,
        cpuPowerMode: String
    ): Boolean {
        // Release model if exists
        releaseModel()

        // Load model
        if (modelPath.isEmpty()) {
            Log.i(TAG, "modelPath.isEmpty() ")
            return false
        }
        var realPath = modelPath
        if (modelPath.substring(0, 1) != "/") {
            // Read model files from custom path if the first character of mode path is '/'
            // otherwise copy model to cache from assets
            realPath = appCtx.cacheDir.toString() + "/" + modelPath
            Log.i(TAG, "realPath.isEmpty() $realPath")
            Utils.copyDirectoryFromAssets(appCtx, modelPath, realPath)
        }
        if (realPath.isEmpty()) {
            Log.i(TAG, "realPath.isEmpty() ")
            return false
        }
        val config = OCRPredictorNative.Config()
        config.cpuThreadNum = cpuThreadNum
        config.detModelFilename = realPath + File.separator + "ch_ppocr_mobile_v2.0_det_opt.nb"
        config.recModelFilename = realPath + File.separator + "ch_ppocr_mobile_v2.0_rec_opt.nb"
        config.clsModelFilename = realPath + File.separator + "ch_ppocr_mobile_v2.0_cls_opt.nb"
        Log.i(
            "Predictor",
            "model path" + config.detModelFilename + " ; " + config.recModelFilename + ";" + config.clsModelFilename
        )
        config.cpuPower = cpuPowerMode
        paddlePredictor = OCRPredictorNative(config)
        this.cpuThreadNum = cpuThreadNum
        this.cpuPowerMode = cpuPowerMode
        this.modelPath = realPath
        modelName = realPath.substring(realPath.lastIndexOf("/") + 1)
        mPaddlePredictorNative = OCRPredictorNative(config)
        Log.i(TAG, "realPath $realPath")
        return true
    }

    fun release() {
        if (mPaddlePredictorNative != null) {
            mPaddlePredictorNative!!.destory()
            mPaddlePredictorNative = null
        }
        modelLoaded = false
    }

    fun releaseModel() {
        if (paddlePredictor != null) {
            paddlePredictor!!.destory()
            paddlePredictor = null
        }
        modelLoaded = false
        cpuThreadNum = 4
        cpuPowerMode = "LITE_POWER_HIGH"
        modelPath = ""
        modelName = ""
    }

    protected fun loadLabel(appCtx: Context, labelPath: String?): Boolean {
        wordLabels.clear()
        wordLabels.add("black")
        // Load word labels from file
        try {
            val assetsInputStream = appCtx.assets.open(labelPath!!)
            val available = assetsInputStream.available()
            val lines = ByteArray(available)
            assetsInputStream.read(lines)
            assetsInputStream.close()
            val words = String(lines)
            val contents = words.split("\n".toRegex()).toTypedArray()
            for (content in contents) {
                wordLabels.add(content)
            }
            Log.i(TAG, "Word label size: " + wordLabels.size)
        } catch (e: Exception) {
            Log.e(TAG, e.message!!)
            return false
        }
        return true
    }

    fun modelLoaded(): Boolean {
        return mPaddlePredictorNative != null && modelLoaded
    }

    fun modelPath(): String {
        return modelPath
    }

    fun modelName(): String {
        return modelName
    }

    fun cpuThreadNum(): Int {
        return cpuThreadNum
    }

    fun cpuPowerMode(): String {
        return cpuPowerMode
    }

    fun inferenceTime(): Float {
        return inferenceTime
    }

    fun inputImage0(): Bitmap? {
        return inputImage0
    }

    fun outputImage(): Bitmap? {
        return outputImage
    }

    fun outputResult(): String {
        return outputResult
    }

    fun preprocessTime(): Float {
        return preprocessTime
    }

    fun postprocessTime(): Float {
        return postprocessTime
    }

    fun setinputImage0(image: Bitmap?) {
        if (image == null) {
            return
        }
        inputImage0 = image.copy(Bitmap.Config.ARGB_8888, true)
    }

    private fun postprocess(results: ArrayList<OcrResultModel>): ArrayList<OcrResultModel> {
        for (r in results) {
            val word = StringBuffer()
            for (index in r.getWordIndex()) {
                if (index >= 0 && index < wordLabels.size) {
                    word.append(wordLabels[index])
                } else {
                    Log.e(TAG, "Word index is not in label list:$index")
                    word.append("×")
                }
            }
            r.label = word.toString()
        }
        return results
    }

    fun runModel(): Boolean {
        if (inputImage0 == null || !modelLoaded()) {
            return false
        }

        // Pre-process image, and feed input tensor with pre-processed data
        val scaleImage = Utils.resizeWithStep(
            inputImage0!!, java.lang.Long.valueOf(
                inputShape[2]
            ).toInt(), 32
        )
        var start = Date()
        val channels = inputShape[1].toInt()
        val width = scaleImage.width
        val height = scaleImage.height
        val inputData = FloatArray(channels * width * height)
        if (channels == 3) {
            var channelIdx: IntArray? = null
            channelIdx = if (inputColorFormat.equals("RGB", ignoreCase = true)) {
                intArrayOf(0, 1, 2)
            } else if (inputColorFormat.equals("BGR", ignoreCase = true)) {
                intArrayOf(2, 1, 0)
            } else {
                Log.i(
                    TAG,
                    "Unknown color format " + inputColorFormat + ", only RGB and BGR color format is " +
                            "supported!"
                )
                return false
            }
            val channelStride = intArrayOf(width * height, width * height * 2)
            val pixels = IntArray(width * height)
            scaleImage.getPixels(
                pixels,
                0,
                scaleImage.width,
                0,
                0,
                scaleImage.width,
                scaleImage.height
            )
            for (i in pixels.indices) {
                val color = pixels[i]
                val rgb = floatArrayOf(
                    Color.red(color).toFloat() / 255.0f, Color.green(color)
                        .toFloat() / 255.0f,
                    Color.blue(color).toFloat() / 255.0f
                )
                inputData[i] = (rgb[channelIdx[0]] - inputMean[0]) / inputStd[0]
                inputData[i + channelStride[0]] = (rgb[channelIdx[1]] - inputMean[1]) / inputStd[1]
                inputData[i + channelStride[1]] = (rgb[channelIdx[2]] - inputMean[2]) / inputStd[2]
            }
        } else if (channels == 1) {
            val pixels = IntArray(width * height)
            scaleImage.getPixels(
                pixels,
                0,
                scaleImage.width,
                0,
                0,
                scaleImage.width,
                scaleImage.height
            )
            for (i in pixels.indices) {
                val color = pixels[i]
                val gray = (Color.red(color) + Color.green(color) + Color.blue(
                    color
                )).toFloat() / 3.0f / 255.0f
                inputData[i] = (gray - inputMean[0]) / inputStd[0]
            }
        } else {
            Log.i(
                TAG,
                "Unsupported channel size " + Integer.toString(channels) + ",  only channel 1 and 3 is " +
                        "supported!"
            )
            return false
        }
        Log.i(
            TAG,
            "pixels " + inputData[0] + " " + inputData[1] + " " + inputData[2] + " " + inputData[3]
                    + " " + inputData[inputData.size / 2] + " " + inputData[inputData.size / 2 + 1] + " " + inputData[inputData.size - 2] + " " + inputData[inputData.size - 1]
        )
        var end = Date()
        preprocessTime = (end.time - start.time).toFloat()

        // Warm up
        for (i in 0 until warmupIterNum) {
            paddlePredictor!!.runImage(inputData, width, height, channels, inputImage0)
        }
        warmupIterNum = 0 // do not need warm
        // Run inference
        start = Date()
        var results = paddlePredictor!!.runImage(inputData, width, height, channels, inputImage0)
        end = Date()
        inferenceTime = (end.time - start.time) / inferIterNum.toFloat()
        results = postprocess(results)
        Log.i(
            TAG, "[stat] Preprocess Time: " + preprocessTime
                    + " ; Inference Time: " + inferenceTime + " ;Box Size " + results.size
        )
        drawResults(results)
        return true
    }

    private fun drawResults(results: ArrayList<OcrResultModel>) {
        val outputResultSb = StringBuffer("")
        for (i in results.indices) {
            val result = results[i]
            val sb = StringBuilder("")
            sb.append(result.label)
            sb.append(" ").append(result.confidence)
            sb.append("; Points: ")
            for (p in result.getPoints()) {
                sb.append("(").append(p.x).append(",").append(p.y).append(") ")
            }
            Log.i(TAG, sb.toString()) // show LOG in Logcat panel
            outputResultSb.append(i + 1).append(": ").append(result.label).append("\n")
        }
        outputResult = outputResultSb.toString()
        outputImage = inputImage0
        val canvas = Canvas(outputImage!!)
        val paintFillAlpha = Paint()
        paintFillAlpha.style = Paint.Style.FILL
        paintFillAlpha.color = Color.parseColor("#3B85F5")
        paintFillAlpha.alpha = 50
        val paint = Paint()
        paint.color = Color.parseColor("#3B85F5")
        paint.strokeWidth = 5f
        paint.style = Paint.Style.STROKE
        for (result in results) {
            val path = Path()
            val points = result.getPoints()
            path.moveTo(points[0].x.toFloat(), points[0].y.toFloat())
            for (i in points.indices.reversed()) {
                val p = points[i]
                path.lineTo(p.x.toFloat(), p.y.toFloat())
            }
            canvas.drawPath(path, paint)
            canvas.drawPath(path, paintFillAlpha)
        }
    }

    fun transformData(OcrResultModelList: List<OcrResultModel>?): List<OcrResult> {
        if (OcrResultModelList == null) {
            return emptyList()
        }
        val words_result: MutableList<OcrResult> = ArrayList()
        for (model in OcrResultModelList) {
            val pointList = model.getPoints()
            if (pointList.isEmpty()) {
                continue
            }
            val firstPoint = pointList[0]
            var left = firstPoint.x
            var top = firstPoint.y
            var right = firstPoint.x
            var bottom = firstPoint.y
            for (p in pointList) {
                if (p.x < left) {
                    left = p.x
                }
                if (p.x > right) {
                    right = p.x
                }
                if (p.y < top) {
                    top = p.y
                }
                if (p.y > bottom) {
                    bottom = p.y
                }
            }
            val ocrResult = OcrResult()
            ocrResult.preprocessTime = preprocessTime
            ocrResult.inferenceTime = inferenceTime
            ocrResult.confidence = model.confidence
            ocrResult.words = model.label!!.trim { it <= ' ' }.replace("\r", "")
            ocrResult.location =
                RectLocation(left, top, Math.abs(right - left), Math.abs(bottom - top))
            ocrResult.bounds = Rect(left, top, right, bottom)
            words_result.add(ocrResult)
        }
        Collections.sort(words_result)
        return words_result
    }

    fun ocr(inputImage0: Bitmap?, cpuThreadNum: Int): List<OcrResult> {
        this.cpuThreadNum = cpuThreadNum
        if (inputImage0 == null) {
            return emptyList()
        }
        // Pre-process image, and feed input tensor with pre-processed data
        val scaleImage = Utils.resizeWithStep(
            inputImage0, java.lang.Long.valueOf(
                inputShape[2]
            ).toInt(), 32
        )
        var start = Date()
        val channels = inputShape[1].toInt()
        val width = scaleImage.width
        val height = scaleImage.height
        val inputData = FloatArray(channels * width * height)
        if (channels == 3) {
            var channelIdx: IntArray? = null
            channelIdx = if (inputColorFormat.equals("RGB", ignoreCase = true)) {
                intArrayOf(0, 1, 2)
            } else if (inputColorFormat.equals("BGR", ignoreCase = true)) {
                intArrayOf(2, 1, 0)
            } else {
                Log.i(
                    TAG,
                    "Unknown color format " + inputColorFormat + ", only RGB and BGR color format is " +
                            "supported!"
                )
                return emptyList()
            }
            val channelStride = intArrayOf(width * height, width * height * 2)
            val pixels = IntArray(width * height)
            scaleImage.getPixels(
                pixels,
                0,
                scaleImage.width,
                0,
                0,
                scaleImage.width,
                scaleImage.height
            )
            for (i in pixels.indices) {
                val color = pixels[i]
                val rgb = floatArrayOf(
                    Color.red(color).toFloat() / 255.0f, Color.green(color)
                        .toFloat() / 255.0f,
                    Color.blue(color).toFloat() / 255.0f
                )
                inputData[i] = (rgb[channelIdx[0]] - inputMean[0]) / inputStd[0]
                inputData[i + channelStride[0]] = (rgb[channelIdx[1]] - inputMean[1]) / inputStd[1]
                inputData[i + channelStride[1]] = (rgb[channelIdx[2]] - inputMean[2]) / inputStd[2]
            }
        } else if (channels == 1) {
            val pixels = IntArray(width * height)
            scaleImage.getPixels(
                pixels,
                0,
                scaleImage.width,
                0,
                0,
                scaleImage.width,
                scaleImage.height
            )
            for (i in pixels.indices) {
                val color = pixels[i]
                val gray = (Color.red(color) + Color.green(color) + Color.blue(
                    color
                )).toFloat() / 3.0f / 255.0f
                inputData[i] = (gray - inputMean[0]) / inputStd[0]
            }
        } else {
            Log.i(
                TAG,
                "Unsupported channel size " + Integer.toString(channels) + ",  only channel 1 and 3 is " +
                        "supported!"
            )
            return emptyList()
        }
        Log.i(
            TAG,
            "pixels " + inputData[0] + " " + inputData[1] + " " + inputData[2] + " " + inputData[3]
                    + " " + inputData[inputData.size / 2] + " " + inputData[inputData.size / 2 + 1] + " " + inputData[inputData.size - 2] + " " + inputData[inputData.size - 1]
        )
        var end = Date()
        preprocessTime = (end.time - start.time).toFloat()

        // Warm up
        for (i in 0 until warmupIterNum) {
            paddlePredictor!!.runImage(inputData, width, height, channels, inputImage0)
        }
        warmupIterNum = 0 // do not need warm
        // Run inference
        start = Date()
        var results = paddlePredictor!!.runImage(inputData, width, height, channels, inputImage0)
        end = Date()
        inferenceTime = (end.time - start.time).toFloat()
        results = postprocess(results)
        return transformData(results)
    }

    companion object {
        private val TAG = Predictor::class.java.simpleName
    }
}