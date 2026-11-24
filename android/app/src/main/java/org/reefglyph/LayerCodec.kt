package org.reefglyph

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.log2
import kotlin.math.pow
import kotlin.math.sin

data class LayerRecord(
    var efd: Array<DoubleArray>? = null,
    var locus: Pair<Double, Double> = 0.5 to 0.5,
    var aspect: Double = 1.0,
    var thumb: IntArray? = null,
    var palette: Array<IntArray>? = null
)

data class AirtimeItem(
    val coralId: Int,
    val bleachIndex: Double,
    val flagged: Boolean
)

enum class AirtimeBurst { SHORT, LONG_UPGRADE }

data class AirtimeQueueEntry(
    val coralId: Int,
    val burst: AirtimeBurst,
    val airtimeSec: Double
)

object LayerCodec {
    const val L1_BITS = 175
    const val L2_BITS = 576
    const val SHORT_AIRTIME_SEC = 2.1333333333333333
    const val LONG_AIRTIME_SEC = 14.08
    const val LONG_UPGRADE_AIRTIME_SEC = LONG_AIRTIME_SEC - SHORT_AIRTIME_SEC

    private const val EFD_RANGE = 0.5

    fun packL1(record: LayerRecord): IntArray {
        val efd = requireNotNull(record.efd) { "L1 requires EFD coefficients" }
        require(efd.size == 8 && efd.all { it.size == 4 }) { "EFD must have shape 8x4" }
        val bits = IntArray(L1_BITS)
        var offset = 0
        for (row in efd) {
            for (value in row) {
                write(bits, offset, quantize(value, -EFD_RANGE, EFD_RANGE, 5), 5)
                offset += 5
            }
        }
        write(bits, offset, quantize(record.locus.first, 0.0, 1.0, 5), 5)
        offset += 5
        write(bits, offset, quantize(record.locus.second, 0.0, 1.0, 5), 5)
        offset += 5
        val clampedAspect = record.aspect.coerceIn(0.25, 4.0)
        write(bits, offset, quantize(log2(clampedAspect), -2.0, 2.0, 5), 5)
        return bits
    }

    fun unpackL1(bits: IntArray, record: LayerRecord = LayerRecord()): LayerRecord {
        require(bits.size == L1_BITS) { "L1 must contain $L1_BITS bits" }
        requireBits(bits)
        record.efd = Array(8) { row ->
            DoubleArray(4) { column ->
                dequantize(read(bits, (row * 4 + column) * 5, 5), -EFD_RANGE, EFD_RANGE, 5)
            }
        }
        record.locus = dequantize(read(bits, 160, 5), 0.0, 1.0, 5) to
            dequantize(read(bits, 165, 5), 0.0, 1.0, 5)
        record.aspect = 2.0.pow(dequantize(read(bits, 170, 5), -2.0, 2.0, 5))
        return record
    }

    fun packL2(record: LayerRecord): IntArray {
        val thumb = requireNotNull(record.thumb) { "L2 requires thumbnail indices" }
        val palette = requireNotNull(record.palette) { "L2 requires a palette" }
        require(thumb.size == 256) { "Thumbnail must contain 256 indices" }
        require(palette.size == 4 && palette.all { it.size == 3 }) { "Palette must have shape 4x3" }
        require(thumb.all { it in 0..3 }) { "Thumbnail indices must be in 0..3" }

        val bits = IntArray(L2_BITS)
        var offset = 0
        for (index in thumb) {
            write(bits, offset, index, 2)
            offset += 2
        }
        for (color in palette) {
            val red = color[0].coerceIn(0, 255)
            val green = color[1].coerceIn(0, 255)
            val blue = color[2].coerceIn(0, 255)
            val rgb565 = ((red shr 3) shl 11) or ((green shr 2) shl 5) or (blue shr 3)
            write(bits, offset, rgb565, 16)
            offset += 16
        }
        return bits
    }

    fun unpackL2(bits: IntArray, record: LayerRecord = LayerRecord()): LayerRecord {
        require(bits.size == L2_BITS) { "L2 must contain $L2_BITS bits" }
        requireBits(bits)
        record.thumb = IntArray(256) { read(bits, it * 2, 2) }
        record.palette = Array(4) { index ->
            val packed = read(bits, 512 + index * 16, 16)
            intArrayOf(
                ((packed ushr 11) and 31) shl 3,
                ((packed ushr 5) and 63) shl 2,
                (packed and 31) shl 3
            )
        }
        return record
    }

    fun reconstructContour(record: LayerRecord, nPoints: Int = 64): Array<DoubleArray> {
        require(nPoints > 0) { "nPoints must be positive" }
        val efd = requireNotNull(record.efd) { "Contour reconstruction requires EFD coefficients" }
        require(efd.size == 8 && efd.all { it.size == 4 }) { "EFD must have shape 8x4" }
        return Array(nPoints) { pointIndex ->
            val t = if (nPoints == 1) 0.0 else pointIndex.toDouble() / (nPoints - 1)
            var x = record.locus.first
            var y = record.locus.second
            for (harmonic in efd.indices) {
                val angle = 2.0 * PI * (harmonic + 1) * t
                x += efd[harmonic][0] * cos(angle) + efd[harmonic][1] * sin(angle)
                y += efd[harmonic][2] * cos(angle) + efd[harmonic][3] * sin(angle)
            }
            doubleArrayOf(x, y)
        }
    }

    /** Converts glyph.py's uint8-encoded CIE Lab input to RGB channel values. */
    fun labToRgb(lab: Triple<Double, Double, Double>): IntArray {
        val encodedL = (lab.first * 2.55).coerceIn(0.0, 255.0).toInt()
        val encodedA = (lab.second + 128.0).coerceIn(0.0, 255.0).toInt()
        val encodedB = (lab.third + 128.0).coerceIn(0.0, 255.0).toInt()
        val lightness = encodedL * (100.0 / 255.0)
        val a = encodedA - 128.0
        val b = encodedB - 128.0

        val fy = (lightness + 16.0) / 116.0
        val fx = fy + a / 500.0
        val fz = fy - b / 200.0
        val x = 0.950456 * labPivot(fx)
        val y = labPivot(fy)
        val z = 1.088754 * labPivot(fz)
        val red = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
        val green = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
        val blue = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
        return intArrayOf(toSrgb8(red), toSrgb8(green), toSrgb8(blue))
    }

    fun buildAirtimeQueue(items: List<AirtimeItem>, budgetSec: Double): List<AirtimeQueueEntry> {
        require(budgetSec >= 0.0 && budgetSec.isFinite()) { "Budget must be finite and non-negative" }
        require(items.all { it.bleachIndex.isFinite() }) { "Bleach indices must be finite" }
        val queue = mutableListOf<AirtimeQueueEntry>()
        var remaining = budgetSec
        val scheduledShorts = mutableListOf<AirtimeItem>()

        for (item in items) {
            if (remaining + 1e-12 < SHORT_AIRTIME_SEC) break
            queue += AirtimeQueueEntry(item.coralId, AirtimeBurst.SHORT, SHORT_AIRTIME_SEC)
            scheduledShorts += item
            remaining -= SHORT_AIRTIME_SEC
        }
        for (item in scheduledShorts.filter { it.flagged }.sortedByDescending { it.bleachIndex }) {
            if (remaining + 1e-12 < LONG_UPGRADE_AIRTIME_SEC) break
            queue += AirtimeQueueEntry(item.coralId, AirtimeBurst.LONG_UPGRADE, LONG_UPGRADE_AIRTIME_SEC)
            remaining -= LONG_UPGRADE_AIRTIME_SEC
        }
        return queue
    }

    private fun quantize(value: Double, low: Double, high: Double, bits: Int): Int {
        require(value.isFinite()) { "Quantized values must be finite" }
        val maximum = (1 shl bits) - 1
        return Math.rint((value - low) / (high - low) * maximum)
            .toInt()
            .coerceIn(0, maximum)
    }

    private fun dequantize(value: Int, low: Double, high: Double, bits: Int): Double =
        low + value.toDouble() / ((1 shl bits) - 1) * (high - low)

    private fun write(target: IntArray, offset: Int, value: Int, width: Int) {
        for (index in 0 until width) {
            target[offset + index] = (value ushr (width - index - 1)) and 1
        }
    }

    private fun read(bits: IntArray, offset: Int, width: Int): Int {
        var value = 0
        for (index in 0 until width) value = (value shl 1) or bits[offset + index]
        return value
    }

    private fun requireBits(bits: IntArray) {
        require(bits.all { it == 0 || it == 1 }) { "Bits must contain only 0 or 1" }
    }

    private fun labPivot(value: Double): Double {
        val delta = 6.0 / 29.0
        return if (value > delta) value * value * value else 3.0 * delta * delta * (value - 4.0 / 29.0)
    }

    private fun toSrgb8(linear: Double): Int {
        val clipped = linear.coerceIn(0.0, 1.0)
        val encoded = if (clipped <= 0.0031308) {
            12.92 * clipped
        } else {
            1.055 * clipped.pow(1.0 / 2.4) - 0.055
        }
        return Math.rint(encoded * 255.0).toInt().coerceIn(0, 255)
    }
}
