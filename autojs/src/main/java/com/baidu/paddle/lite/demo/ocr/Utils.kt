package com.baidu.paddle.lite.demo.ocr

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.ExifInterface
import android.os.Environment
import java.io.*
import java.lang.Exception

object Utils {
    private val TAG = Utils::class.java.simpleName
    fun copyFileFromAssets(appCtx: Context, srcPath: String, dstPath: String) {
        if (srcPath.isEmpty() || dstPath.isEmpty()) {
            return
        }
        var `is`: InputStream? = null
        var os: OutputStream? = null
        try {
            `is` = BufferedInputStream(appCtx.assets.open(srcPath))
            os = BufferedOutputStream(FileOutputStream(File(dstPath)))
            val buffer = ByteArray(1024)
            var length = 0
            while (`is`.read(buffer).also { length = it } != -1) {
                os.write(buffer, 0, length)
            }
        } catch (e: FileNotFoundException) {
            e.printStackTrace()
        } catch (e: IOException) {
            e.printStackTrace()
        } finally {
            try {
                os!!.close()
                `is`!!.close()
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }
    }

    fun copyDirectoryFromAssets(appCtx: Context, srcDir: String, dstDir: String) {
        if (srcDir.isEmpty() || dstDir.isEmpty()) {
            return
        }
        try {
            if (!File(dstDir).exists()) {
                File(dstDir).mkdirs()
            }
            for (fileName in appCtx.assets.list(srcDir)!!) {
                val srcSubPath = srcDir + File.separator + fileName
                val dstSubPath = dstDir + File.separator + fileName
                if (File(srcSubPath).isDirectory) {
                    copyDirectoryFromAssets(appCtx, srcSubPath, dstSubPath)
                } else {
                    copyFileFromAssets(appCtx, srcSubPath, dstSubPath)
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun parseFloatsFromString(string: String, delimiter: String): FloatArray {
        val pieces =
            string.trim { it <= ' ' }.toLowerCase().split(delimiter.toRegex()).toTypedArray()
        val floats = FloatArray(pieces.size)
        for (i in pieces.indices) {
            floats[i] = pieces[i].trim { it <= ' ' }.toFloat()
        }
        return floats
    }

    fun parseLongsFromString(string: String, delimiter: String): LongArray {
        val pieces =
            string.trim { it <= ' ' }.toLowerCase().split(delimiter.toRegex()).toTypedArray()
        val longs = LongArray(pieces.size)
        for (i in pieces.indices) {
            longs[i] = pieces[i].trim { it <= ' ' }.toLong()
        }
        return longs
    }

    val sDCardDirectory: String
        get() = Environment.getExternalStorageDirectory().absolutePath

    // String hardware = android.os.Build.HARDWARE;
    // return hardware.equalsIgnoreCase("kirin810") || hardware.equalsIgnoreCase("kirin990");
    val isSupportedNPU: Boolean
        get() = false

    // String hardware = android.os.Build.HARDWARE;
    // return hardware.equalsIgnoreCase("kirin810") || hardware.equalsIgnoreCase("kirin990");
    fun resizeWithStep(bitmap: Bitmap, maxLength: Int, step: Int): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val maxWH = Math.max(width, height)
        var ratio = 1f
        var newWidth = width
        var newHeight = height
        if (maxWH > maxLength) {
            ratio = maxLength * 1.0f / maxWH
            newWidth = Math.floor((ratio * width).toDouble()).toInt()
            newHeight = Math.floor((ratio * height).toDouble()).toInt()
        }
        newWidth = newWidth - newWidth % step
        if (newWidth == 0) {
            newWidth = step
        }
        newHeight = newHeight - newHeight % step
        if (newHeight == 0) {
            newHeight = step
        }
        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
    }

    fun rotateBitmap(bitmap: Bitmap, orientation: Int): Bitmap? {
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_NORMAL -> return bitmap
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.setScale(-1f, 1f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.setRotate(180f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> {
                matrix.setRotate(180f)
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.setRotate(90f)
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.setRotate(90f)
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.setRotate(-90f)
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.setRotate(-90f)
            else -> return bitmap
        }
        return try {
            val bmRotated =
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            bitmap.recycle()
            bmRotated
        } catch (e: OutOfMemoryError) {
            e.printStackTrace()
            null
        }
    }
}