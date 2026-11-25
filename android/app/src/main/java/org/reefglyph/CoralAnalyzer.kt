package org.reefglyph

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.SystemClock
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect as CvRect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.IOException
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class LabColor(val l: Double, val a: Double, val b: Double)

data class AnalysisResult(
    val bleachIndex: Double,
    val flagged: Boolean,
    val healthScore: Int,
    val textureBin: Int,
    val dominantColors: List<LabColor>,
    val elapsedMs: Double
)

object CoralAnalyzer {
    private const val MAX_SIDE = 320
    private const val EROSION_SIZE = 7

    private data class Parameters(
        val lightnessThreshold: Double,
        val chromaThreshold: Double,
        val flagThreshold: Double,
        val healthEdges: DoubleArray,
        val textureEdges: DoubleArray,
        val normalize: Boolean
    )

    @Volatile
    private var parameters: Parameters? = null

    @Synchronized
    @Throws(IOException::class)
    fun initialize(context: Context) {
        if (parameters != null) return

        val json = context.assets.open("params.json").bufferedReader().use {
            JSONObject(it.readText())
        }
        val healthEdges = json.getDoubleArray("health_edges")
        val textureEdges = json.getDoubleArray("texture_edges")
        require(healthEdges.size == 5) { "params.json must contain five health_edges" }
        require(textureEdges.size == 15) { "params.json must contain fifteen texture_edges" }
        require(isFiniteAscending(healthEdges)) {
            "health_edges must be finite and ascending"
        }
        require(isFiniteAscending(textureEdges)) {
            "texture_edges must be finite and ascending"
        }
        val loadedParameters = Parameters(
            lightnessThreshold = json.getDouble("L_t"),
            chromaThreshold = json.getDouble("C_t"),
            flagThreshold = json.getDouble("tau"),
            healthEdges = healthEdges,
            textureEdges = textureEdges,
            normalize = json.optInt("norm", 0) != 0
        )
        check(OpenCVLoader.initLocal()) { "OpenCV native library initialization failed" }
        parameters = loadedParameters
    }

    fun analyzeL0(bitmap: Bitmap, rect: Rect): AnalysisResult {
        val startNs = SystemClock.elapsedRealtimeNanos()
        val config = parameters ?: throw IllegalStateException("Call CoralAnalyzer.initialize(context) first")
        require(bitmap.width > 0 && bitmap.height > 0) { "Bitmap must not be empty" }
        require(rect.left >= 0 && rect.top >= 0 && rect.right <= bitmap.width && rect.bottom <= bitmap.height) {
            "Analysis rectangle must be inside the bitmap"
        }
        require(rect.width() > 0 && rect.height() > 0) { "Analysis rectangle must have positive dimensions" }

        val source = Mat()
        val sourceBgr = Mat()
        val working = Mat()
        val grabCutMask = Mat()
        val foregroundMask = Mat()
        val erodedMask = Mat()
        var kernel = Mat()
        val lab = Mat()
        val chroma = Mat()
        val gradientX = Mat()
        val gradientY = Mat()
        val magnitude = Mat()
        try {
            Utils.bitmapToMat(bitmap, source)
            Imgproc.cvtColor(source, sourceBgr, Imgproc.COLOR_RGBA2BGR)

            val scale = min(1.0, MAX_SIDE.toDouble() / max(sourceBgr.cols(), sourceBgr.rows()))
            val targetWidth = max(1, (sourceBgr.cols() * scale).roundToInt())
            val targetHeight = max(1, (sourceBgr.rows() * scale).roundToInt())
            if (scale < 1.0) {
                Imgproc.resize(sourceBgr, working, Size(targetWidth.toDouble(), targetHeight.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            } else {
                sourceBgr.copyTo(working)
            }

            val scaledRect = CvRect(
                (rect.left * scale).toInt(),
                (rect.top * scale).toInt(),
                (rect.width() * scale).toInt(),
                (rect.height() * scale).toInt()
            )
            require(scaledRect.width > 0 && scaledRect.height > 0) { "Scaled analysis rectangle is empty" }
            grabCutMask.create(working.rows(), working.cols(), CvType.CV_8UC1)
            grabCutMask.setTo(Scalar.all(Imgproc.GC_BGD.toDouble()))
            val backgroundModel = Mat.zeros(1, 65, CvType.CV_64FC1)
            val foregroundModel = Mat.zeros(1, 65, CvType.CV_64FC1)
            try {
                Imgproc.grabCut(
                    working,
                    grabCutMask,
                    scaledRect,
                    backgroundModel,
                    foregroundModel,
                    4,
                    Imgproc.GC_INIT_WITH_RECT
                )
            } finally {
                backgroundModel.release()
                foregroundModel.release()
            }

            val grabLabels = ByteArray(working.rows() * working.cols())
            grabCutMask.get(0, 0, grabLabels)
            val maskBytes = ByteArray(grabLabels.size) { index ->
                val label = grabLabels[index].toInt() and 0xff
                if (label == Imgproc.GC_FGD || label == Imgproc.GC_PR_FGD) 0xff.toByte() else 0
            }
            foregroundMask.create(working.rows(), working.cols(), CvType.CV_8UC1)
            foregroundMask.put(0, 0, maskBytes)

            kernel.release()
            kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT,
                Size(EROSION_SIZE.toDouble(), EROSION_SIZE.toDouble())
            )
            Imgproc.erode(foregroundMask, erodedMask, kernel)
            val erodedCount = Core.countNonZero(erodedMask)
            if (erodedCount < 20) foregroundMask.copyTo(erodedMask)
            if (Core.countNonZero(erodedMask) == 0) erodedMask.setTo(Scalar.all(255.0))

            if (config.normalize) grayWorld(working)
            Imgproc.cvtColor(working, lab, Imgproc.COLOR_BGR2Lab)

            val rows = working.rows()
            val cols = working.cols()
            val pixelCount = rows * cols
            val labBytes = ByteArray(pixelCount * 3)
            val bgrBytes = ByteArray(pixelCount * 3)
            val selectedBytes = ByteArray(pixelCount)
            lab.get(0, 0, labBytes)
            working.get(0, 0, bgrBytes)
            erodedMask.get(0, 0, selectedBytes)

            val lightness = DoubleArray(Core.countNonZero(erodedMask))
            val chromaValues = DoubleArray(lightness.size)
            var sampleCount = 0
            for (index in 0 until pixelCount) {
                if ((selectedBytes[index].toInt() and 0xff) == 0) continue
                val labIndex = index * 3
                val l = (labBytes[labIndex].toInt() and 0xff) * (100.0 / 255.0)
                val a = (labBytes[labIndex + 1].toInt() and 0xff) - 128.0
                val b = (labBytes[labIndex + 2].toInt() and 0xff) - 128.0
                lightness[sampleCount] = l
                chromaValues[sampleCount] = hypot(a, b)
                sampleCount++
            }
            check(sampleCount > 0) { "GrabCut produced an empty analysis mask" }

            val sortedLightness = lightness.copyOf(sampleCount).apply { sort() }
            val percentileIndex = 0.99 * (sampleCount - 1)
            val lowerIndex = percentileIndex.toInt()
            val upperIndex = min(sampleCount - 1, lowerIndex + 1)
            val percentile99 = sortedLightness[lowerIndex] +
                (sortedLightness[upperIndex] - sortedLightness[lowerIndex]) * (percentileIndex - lowerIndex)

            var keptCount = 0
            var lightnessSum = 0.0
            var chromaSum = 0.0
            var bleachedCount = 0
            for (index in 0 until sampleCount) {
                val l = lightness[index]
                if (l > percentile99) continue
                val c = chromaValues[index]
                lightnessSum += l
                chromaSum += c
                if (l > config.lightnessThreshold && c < config.chromaThreshold) bleachedCount++
                keptCount++
            }
            check(keptCount > 0) { "No pixels remained after specular exclusion" }

            val bleachIndex = bleachedCount.toDouble() / keptCount
            val healthValue = lightnessSum / keptCount - chromaSum / keptCount
            val healthScore = lowerBoundIndex(config.healthEdges, healthValue) + 1
            val flagged = bleachIndex > config.flagThreshold

            val chromaPlane = FloatArray(pixelCount)
            for (index in 0 until pixelCount) {
                val labIndex = index * 3
                val a = (labBytes[labIndex + 1].toInt() and 0xff) - 128.0
                val b = (labBytes[labIndex + 2].toInt() and 0xff) - 128.0
                chromaPlane[index] = hypot(a, b).toFloat()
            }
            chroma.create(rows, cols, CvType.CV_32FC1)
            chroma.put(0, 0, chromaPlane)
            Imgproc.Scharr(chroma, gradientX, CvType.CV_32F, 1, 0)
            Imgproc.Scharr(chroma, gradientY, CvType.CV_32F, 0, 1)
            Core.magnitude(gradientX, gradientY, magnitude)
            val gradientValues = FloatArray(pixelCount)
            magnitude.get(0, 0, gradientValues)
            var gradientSum = 0.0
            var gradientCount = 0
            for (index in 0 until pixelCount) {
                if ((selectedBytes[index].toInt() and 0xff) != 0) {
                    gradientSum += gradientValues[index]
                    gradientCount++
                }
            }
            val textureMagnitude = if (gradientCount == 0) 0.0 else gradientSum / gradientCount
            val textureBin = lowerBoundIndex(config.textureEdges, textureMagnitude).coerceIn(0, 15)

            val pixels = IntArray(sampleCount)
            var pixelIndex = 0
            for (index in 0 until pixelCount) {
                if ((selectedBytes[index].toInt() and 0xff) == 0) continue
                val offset = index * 3
                pixels[pixelIndex++] =
                    ((bgrBytes[offset].toInt() and 0xff) shl 16) or
                    ((bgrBytes[offset + 1].toInt() and 0xff) shl 8) or
                    (bgrBytes[offset + 2].toInt() and 0xff)
            }
            val colors = medianCutLab(pixels)

            return AnalysisResult(
                bleachIndex = bleachIndex,
                flagged = flagged,
                healthScore = healthScore.coerceIn(1, 6),
                textureBin = textureBin,
                dominantColors = colors,
                elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000.0
            )
        } finally {
            magnitude.release()
            gradientY.release()
            gradientX.release()
            chroma.release()
            lab.release()
            kernel.release()
            erodedMask.release()
            foregroundMask.release()
            grabCutMask.release()
            working.release()
            sourceBgr.release()
            source.release()
        }
    }

    private fun grayWorld(bgr: Mat) {
        val values = ByteArray(bgr.rows() * bgr.cols() * 3)
        bgr.get(0, 0, values)
        val means = DoubleArray(3)
        val pixelCount = bgr.rows() * bgr.cols()
        for (index in 0 until pixelCount) {
            val offset = index * 3
            for (channel in 0..2) means[channel] += values[offset + channel].toInt() and 0xff
        }
        for (channel in 0..2) means[channel] /= pixelCount
        val targetMean = means.average()
        for (index in 0 until pixelCount) {
            val offset = index * 3
            for (channel in 0..2) {
                val original = values[offset + channel].toInt() and 0xff
                val normalized = if (means[channel] == 0.0) original else original * (targetMean / means[channel])
                values[offset + channel] = normalized.toInt().coerceIn(0, 255).toByte()
            }
        }
        bgr.put(0, 0, values)
    }

    private fun medianCutLab(pixels: IntArray): List<LabColor> {
        check(pixels.isNotEmpty()) { "Cannot compute a palette from an empty mask" }
        var channel = 0
        var largestRange = -1
        for (candidate in 0..2) {
            var minimum = 255
            var maximum = 0
            for (pixel in pixels) {
                val value = channelValue(pixel, candidate)
                minimum = min(minimum, value)
                maximum = max(maximum, value)
            }
            if (maximum - minimum > largestRange) {
                largestRange = maximum - minimum
                channel = candidate
            }
        }
        sortPixels(pixels, channel)
        val midpoint = max(1, pixels.size / 2)
        val first = averageColor(pixels, 0, midpoint)
        val second = if (pixels.size > 1) averageColor(pixels, midpoint, pixels.size) else first
        return listOf(first, second).map { bgr ->
            val color = Mat(1, 1, CvType.CV_8UC3, Scalar(bgr[0].toDouble(), bgr[1].toDouble(), bgr[2].toDouble()))
            val lab = Mat()
            try {
                Imgproc.cvtColor(color, lab, Imgproc.COLOR_BGR2Lab)
                val bytes = ByteArray(3)
                lab.get(0, 0, bytes)
                LabColor(
                    l = (bytes[0].toInt() and 0xff) * (100.0 / 255.0),
                    a = (bytes[1].toInt() and 0xff) - 128.0,
                    b = (bytes[2].toInt() and 0xff) - 128.0
                )
            } finally {
                lab.release()
                color.release()
            }
        }
    }

    private fun averageColor(pixels: IntArray, start: Int, end: Int): IntArray {
        val count = end - start
        val sums = LongArray(3)
        for (index in start until end) {
            val pixel = pixels[index]
            sums[0] += (pixel ushr 16) and 0xff
            sums[1] += (pixel ushr 8) and 0xff
            sums[2] += pixel and 0xff
        }
        return IntArray(3) { (sums[it] / count).toInt() }
    }

    private fun sortPixels(pixels: IntArray, channel: Int) {
        fun quickSort(low: Int, high: Int) {
            var left = low
            var right = high
            val pivot = channelValue(pixels[(low + high) ushr 1], channel)
            while (left <= right) {
                while (channelValue(pixels[left], channel) < pivot) left++
                while (channelValue(pixels[right], channel) > pivot) right--
                if (left <= right) {
                    val value = pixels[left]
                    pixels[left] = pixels[right]
                    pixels[right] = value
                    left++
                    right--
                }
            }
            if (low < right) quickSort(low, right)
            if (left < high) quickSort(left, high)
        }
        if (pixels.size > 1) quickSort(0, pixels.lastIndex)
    }

    private fun channelValue(pixel: Int, channel: Int): Int = when (channel) {
        0 -> (pixel ushr 16) and 0xff
        1 -> (pixel ushr 8) and 0xff
        else -> pixel and 0xff
    }

    private fun lowerBoundIndex(edges: DoubleArray, value: Double): Int {
        var low = 0
        var high = edges.size
        while (low < high) {
            val middle = (low + high) ushr 1
            if (edges[middle] < value) low = middle + 1 else high = middle
        }
        return low
    }

    private fun isFiniteAscending(values: DoubleArray): Boolean {
        for (index in values.indices) {
            if (!values[index].isFinite()) return false
            if (index > 0 && values[index] < values[index - 1]) return false
        }
        return true
    }

    private fun JSONObject.getDoubleArray(key: String): DoubleArray {
        val array = getJSONArray(key)
        return DoubleArray(array.length()) { index -> array.getDouble(index) }
    }
}
