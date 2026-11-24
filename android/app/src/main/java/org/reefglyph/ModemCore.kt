package org.reefglyph

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.round
import kotlin.math.sin
import kotlin.math.sqrt

data class CoralRecord(
    var transect: Int = 1,
    var coral: Int = 42,
    var depthM: Double = 7.5,
    var timeS: Double = 300.0,
    var bleach: Double = 0.62,
    var health: Int = 4,
    var morph: Int = 0,
    var colors: List<Triple<Double, Double, Double>> = listOf(
        Triple(80.0, 5.0, 10.0),
        Triple(60.0, -4.0, 12.0)
    ),
    var patch: Int = 0,
    var evidence: Int = 1,
    var texture: Int = 7
)

/** Platform-independent modem, framing, FEC, and receive-gating primitives. */
object ModemCore {
    const val SAMPLE_RATE = 48_000
    const val BANDWIDTH = 3_000
    const val CENTER_FREQUENCY = 2_500
    const val DEFAULT_SF = 7
    const val OVERSAMPLE = SAMPLE_RATE / BANDWIDTH
    const val PREAMBLE_UP = 8
    const val PREAMBLE_DOWN = 2
    const val HEADER_BITS = 16
    const val PAYLOAD_BITS = 108
    const val FRAME_BITS = 140
    const val ACK_BITS = 28
    const val MAX_ATTEMPTS = 3

    private const val CRC16_POLY = 0x1021
    private const val CRC8_POLY = 0x07

    data class Frame(
        val layer: Int,
        val idx: Int,
        val n: Int,
        val burst: Int,
        val payload: IntArray,
        val ok: Boolean = true
    ) {
        val framesInBurst: Int get() = n
        val burstId: Int get() = burst
    }

    data class Ack(
        val ok: Boolean,
        val burstId: Int,
        val bitmap: BooleanArray,
        val snr4: Int,
        val pad: Int
    )

    data class SyncInfo(
        val start: Int,
        val cfo: Double,
        val tau: Double,
        val snrDb: Double,
        val snr4: Int
    )

    data class DemodResult(val frames: List<Frame>, val info: SyncInfo?)

    private data class Peak(val bin: Double, val ratio: Double)
    private data class ComplexArray(val re: DoubleArray, val im: DoubleArray)

    private val layerBits = intArrayOf(80, 175, 576)
    private val layerFrames = intArrayOf(1, 2, 6)
    private val codewords = arrayOf(encodeCodewords(7), encodeCodewords(8))
    private val decodeTables = arrayOf(decodeTable(7), decodeTable(8))
    private val chirpCache = HashMap<Int, Pair<DoubleArray, DoubleArray>>()

    fun frameSymbols(sf: Int = DEFAULT_SF, first: Boolean = false): Int {
        require(sf in 1..15)
        return ceil(35.0 / sf).toInt() * if (first) 8 else 7
    }

    fun ackSymbols(sf: Int = DEFAULT_SF): Int {
        require(sf in 1..15)
        return ceil(7.0 / sf).toInt() * 8
    }

    fun bitsToInt(bits: IntArray): Int {
        var value = 0
        for (bit in bits) {
            require(bit == 0 || bit == 1) { "Bits must contain only 0 or 1" }
            value = (value shl 1) or bit
        }
        return value
    }

    fun intToBits(value: Int, count: Int): IntArray {
        require(count >= 0 && count <= 31)
        return IntArray(count) { i -> (value ushr (count - 1 - i)) and 1 }
    }

    fun whiten(bits: IntArray): IntArray {
        var state = 0xff
        return IntArray(bits.size) { i ->
            require(bits[i] == 0 || bits[i] == 1) { "Bits must contain only 0 or 1" }
            val out = bits[i] xor (state and 1)
            val feedback = ((state ushr 7) xor (state ushr 5) xor (state ushr 4) xor (state ushr 3)) and 1
            state = ((state shl 1) or feedback) and 0xff
            out
        }
    }

    fun grayEncode(value: Int): Int = value xor (value ushr 1)

    fun grayDecode(gray: Int): Int {
        var value = 0
        var remaining = gray
        while (remaining != 0) {
            value = value xor remaining
            remaining = remaining ushr 1
        }
        return value
    }

    private fun encodeCodewords(n: Int): Array<IntArray> = Array(16) { nibble ->
        val d0 = (nibble ushr 3) and 1
        val d1 = (nibble ushr 2) and 1
        val d2 = (nibble ushr 1) and 1
        val d3 = nibble and 1
        val word = intArrayOf(
            d0, d1, d2, d3,
            d0 xor d1 xor d3,
            d0 xor d2 xor d3,
            d1 xor d2 xor d3
        )
        if (n == 8) word + (word.sum() and 1) else word
    }

    private fun decodeTable(n: Int): IntArray {
        val words = encodeCodewords(n)
        return IntArray(1 shl n) { value ->
            val received = intToBits(value, n)
            var bestNibble = 0
            var bestDistance = n + 1
            for (nibble in words.indices) {
                var distance = 0
                for (j in 0 until n) if (received[j] != words[nibble][j]) distance++
                if (distance < bestDistance) {
                    bestDistance = distance
                    bestNibble = nibble
                }
            }
            bestNibble
        }
    }

    /** Encode one frame's bits to Gray-mapped CSS symbol bins. */
    fun encodeBits(bits: IntArray, sf: Int = DEFAULT_SF, hammingBits: Int = 7): IntArray {
        require(sf in 1..15 && hammingBits in intArrayOf(7, 8))
        require(bits.size % 4 == 0) { "FEC input length must be a multiple of four bits" }
        val whitened = whiten(bits)
        val words = codewords[if (hammingBits == 7) 0 else 1]
        val encoded = Array(whitened.size / 4) { i ->
            val nibble = bitsToInt(whitened.copyOfRange(i * 4, i * 4 + 4))
            words[nibble]
        }
        val padding = (sf - encoded.size % sf) % sf
        val codewordCount = encoded.size + padding
        val result = IntArray((codewordCount / sf) * hammingBits)
        var out = 0
        for (blockStart in 0 until codewordCount step sf) {
            for (bitIndex in 0 until hammingBits) {
                var value = 0
                for (wordIndex in 0 until sf) {
                    val cw = blockStart + wordIndex
                    value = (value shl 1) or
                        (if (cw < encoded.size) encoded[cw][bitIndex] else 0)
                }
                result[out++] = grayDecode(value)
            }
        }
        return result
    }

    /** Decode Gray-mapped symbol bins back to a whitened bit sequence. */
    fun decodeSymbols(
        symbols: IntArray,
        sf: Int = DEFAULT_SF,
        hammingBits: Int = 7,
        bitCount: Int = FRAME_BITS
    ): IntArray {
        require(sf in 1..15 && hammingBits in intArrayOf(7, 8))
        require(bitCount >= 0 && bitCount % 4 == 0)
        require(symbols.size % hammingBits == 0) { "Symbol count must contain complete codeword blocks" }
        val cwCount = symbols.size / hammingBits * sf
        val wantedCodewords = bitCount / 4
        require(cwCount >= wantedCodewords) { "Not enough symbols to decode requested bits" }
        val decoded = IntArray(wantedCodewords)
        val table = decodeTables[if (hammingBits == 7) 0 else 1]
        for (block in 0 until symbols.size / hammingBits) {
            val values = IntArray(hammingBits) { j ->
                require(symbols[block * hammingBits + j] in 0 until (1 shl sf))
                grayEncode(symbols[block * hammingBits + j])
            }
            for (wordIndex in 0 until sf) {
                val cwIndex = block * sf + wordIndex
                if (cwIndex >= wantedCodewords) continue
                var packed = 0
                for (bitIndex in 0 until hammingBits) {
                    packed = (packed shl 1) or ((values[bitIndex] ushr (sf - 1 - wordIndex)) and 1)
                }
                decoded[cwIndex] = table[packed]
            }
        }
        val bits = IntArray(bitCount)
        for (i in decoded.indices) {
            val nibble = intToBits(decoded[i], 4)
            nibble.copyInto(bits, i * 4)
        }
        return whiten(bits)
    }

    fun crc16(bits: IntArray): Int {
        var crc = 0xffff
        for (bit in bits) {
            require(bit == 0 || bit == 1)
            val top = ((crc ushr 15) and 1) xor bit
            crc = (crc shl 1) and 0xffff
            if (top != 0) crc = crc xor CRC16_POLY
        }
        return crc
    }

    fun crc8(bits: IntArray): Int {
        var crc = 0
        for (bit in bits) {
            require(bit == 0 || bit == 1)
            val top = ((crc ushr 7) and 1) xor bit
            crc = (crc shl 1) and 0xff
            if (top != 0) crc = crc xor CRC8_POLY
        }
        return crc
    }

    fun packFrame(frame: Frame): IntArray {
        require(frame.layer in 0..3 && frame.idx in 0..15 && frame.n in 0..15 && frame.burst in 0..63)
        require(frame.payload.size <= PAYLOAD_BITS)
        val header = intToBits(frame.layer, 2) + intToBits(frame.idx, 4) +
            intToBits(frame.n, 4) + intToBits(frame.burst, 6)
        val payload = IntArray(PAYLOAD_BITS)
        frame.payload.copyInto(payload)
        val body = header + payload
        return body + intToBits(crc16(body), 16)
    }

    fun unpackFrame(bits: IntArray): Frame {
        require(bits.size >= FRAME_BITS) { "A frame must contain $FRAME_BITS bits" }
        val body = bits.copyOfRange(0, 124)
        val crc = bitsToInt(bits.copyOfRange(124, 140))
        return Frame(
            layer = bitsToInt(bits.copyOfRange(0, 2)),
            idx = bitsToInt(bits.copyOfRange(2, 6)),
            n = bitsToInt(bits.copyOfRange(6, 10)),
            burst = bitsToInt(bits.copyOfRange(10, 16)),
            payload = bits.copyOfRange(16, 124),
            ok = crc16(body) == crc
        )
    }

    fun unpackL0(bits: IntArray): CoralRecord {
        require(bits.size >= layerBits[0]) { "L0 must contain ${layerBits[0]} bits" }
        var offset = 0
        fun read(count: Int): Int {
            val value = bitsToInt(bits.copyOfRange(offset, offset + count))
            offset += count
            return value
        }
        fun dequantize(value: Int, maximum: Double, bitCount: Int): Double =
            value.toDouble() / ((1 shl bitCount) - 1) * maximum

        return CoralRecord(
            transect = read(6),
            coral = read(8),
            depthM = dequantize(read(7), 63.5, 7),
            timeS = dequantize(read(10), 5115.0, 10),
            bleach = dequantize(read(6), 1.0, 6),
            health = read(3) + 1,
            morph = read(2),
            colors = List(2) {
                val lightness = dequantize(read(6), 100.0, 6)
                val a = (read(5) - 16) * 4.0
                val b = (read(5) - 16) * 4.0
                Triple(lightness, a, b)
            },
            patch = read(1),
            evidence = read(1),
            texture = read(4)
        )
    }

    fun packetize(layers: Map<Int, IntArray>, burstId: Int, only: Set<Int>? = null): List<IntArray> {
        require(burstId in 0..63)
        val chunks = ArrayList<Pair<Int, IntArray>>()
        for (layer in layers.keys.sorted()) {
            require(layer in 0..2) { "Unknown layer $layer" }
            val source = layers.getValue(layer)
            for (index in 0 until layerFrames[layer]) {
                val payload = IntArray(PAYLOAD_BITS)
                val start = index * PAYLOAD_BITS
                val count = min(PAYLOAD_BITS, max(0, source.size - start))
                if (count > 0) source.copyInto(payload, 0, start, start + count)
                chunks.add(layer to payload)
            }
        }
        val frameCount = chunks.size
        return chunks.mapIndexedNotNull { index, (layer, payload) ->
            if (only != null && index !in only) null
            else packFrame(Frame(layer, index, frameCount, burstId, payload))
        }
    }

    fun buildSelectiveResendBurst(
        originalFrames: List<IntArray>,
        receivedBitmap: BooleanArray,
        attempt: Int
    ): List<IntArray> {
        require(attempt in 1..MAX_ATTEMPTS) { "Selective resend attempts are limited to $MAX_ATTEMPTS" }
        require(receivedBitmap.size >= originalFrames.size)
        return originalFrames.filterIndexed { index, _ -> !receivedBitmap[index] }
    }

    fun packAck(burstId: Int, bitmap: BooleanArray, snr4: Int): IntArray {
        require(burstId in 0..63 && snr4 in 0..15)
        val body = intToBits(burstId, 6) + IntArray(9) { if (it < bitmap.size && bitmap[it]) 1 else 0 } +
            intToBits(snr4, 4)
        val crcInput = body + IntArray(5)
        return body + intToBits(crc8(crcInput), 8) + intArrayOf(0)
    }

    fun packAck(burstId: Int, bitmap: IntArray, snr4: Int): IntArray =
        packAck(burstId, BooleanArray(bitmap.size) { bitmap[it] != 0 }, snr4)

    fun unpackAck(bits: IntArray): Ack {
        require(bits.size >= ACK_BITS) { "An ACK must contain $ACK_BITS bits" }
        val body = bits.copyOfRange(0, 19)
        val crcInput = body + IntArray(5)
        return Ack(
            ok = crc8(crcInput) == bitsToInt(bits.copyOfRange(19, 27)),
            burstId = bitsToInt(bits.copyOfRange(0, 6)),
            bitmap = BooleanArray(9) { bits[6 + it] != 0 },
            snr4 = bitsToInt(bits.copyOfRange(15, 19)),
            pad = bits[27]
        )
    }

    /** Stores successful frames separately per burst, so attempts cannot contaminate one another. */
    class Reassembler {
        private data class BurstState(var n: Int = 9, val got: MutableMap<Int, Frame> = linkedMapOf())
        private val bursts = linkedMapOf<Int, BurstState>()
        var burstId: Int? = null
            private set
        var n: Int = 9
            private set

        fun add(frames: List<Frame>) {
            for (frame in frames) {
                if (!frame.ok || frame.idx !in 0..8 || frame.n !in 1..9) continue
                val state = bursts.getOrPut(frame.burst) { BurstState(frame.n) }
                state.n = frame.n
                state.got[frame.idx] = frame
                burstId = frame.burst
                n = frame.n
            }
        }

        fun bitmap(forBurstId: Int? = burstId): BooleanArray {
            if (forBurstId == null) return BooleanArray(9)
            val state = bursts[forBurstId]
            return BooleanArray(9) { state?.got?.containsKey(it) == true }
        }

        fun missing(forBurstId: Int? = burstId): List<Int> {
            if (forBurstId == null) return emptyList()
            val state = bursts[forBurstId] ?: return emptyList()
            return (0 until state.n).filterNot { state.got.containsKey(it) }
        }

        fun layers(forBurstId: Int? = burstId): Map<Int, IntArray> {
            if (forBurstId == null) return emptyMap()
            val state = bursts[forBurstId] ?: return emptyMap()
            val result = linkedMapOf<Int, IntArray>()
            var base = 0
            for (layer in 0..2) {
                val end = base + layerFrames[layer]
                if (end > state.n) break
                if ((base until end).all { state.got.containsKey(it) }) {
                    val all = IntArray(layerFrames[layer] * PAYLOAD_BITS)
                    for (i in base until end) state.got.getValue(i).payload.copyInto(
                        all, (i - base) * PAYLOAD_BITS
                    )
                    result[layer] = all.copyOf(layerBits[layer])
                }
                base = end
            }
            return result
        }
    }

    fun baseChirp(sf: Int = DEFAULT_SF): Pair<DoubleArray, DoubleArray> {
        require(sf in 1..15)
        val chirp = chirp(sf)
        return chirp.first.copyOf() to chirp.second.copyOf()
    }

    private fun chirp(sf: Int): Pair<DoubleArray, DoubleArray> {
        return synchronized(chirpCache) {
            chirpCache[sf] ?: run {
                val size = 1 shl sf
                val re = DoubleArray(size)
                val im = DoubleArray(size)
                for (i in 0 until size) {
                    val phase = PI * i.toDouble() * i / size - PI * i
                    re[i] = cos(phase)
                    im[i] = sin(phase)
                }
                (re to im).also { chirpCache[sf] = it }
            }
        }
    }

    private fun symbolWave(bin: Int, sf: Int, down: Boolean = false): ComplexArray {
        val (baseRe, baseIm) = chirp(sf)
        val n = baseRe.size
        val re = DoubleArray(n)
        val im = DoubleArray(n)
        for (i in 0 until n) {
            val source = (i + bin) % n
            re[i] = baseRe[source]
            im[i] = if (down) -baseIm[source] else baseIm[source]
        }
        return ComplexArray(re, im)
    }

    fun modulate(frameBits: List<IntArray>, sf: Int = DEFAULT_SF, ack: Boolean = false): FloatArray {
        require(sf in 1..15)
        require(frameBits.isNotEmpty())
        val n = 1 shl sf
        val parts = ArrayList<ComplexArray>()
        repeat(PREAMBLE_UP) { parts.add(symbolWave(0, sf)) }
        repeat(PREAMBLE_DOWN) { parts.add(symbolWave(0, sf, down = true)) }
        for ((position, bits) in frameBits.withIndex()) {
            val codingBits = if (ack) 8 else if (position == 0) 8 else 7
            val symbols = encodeBits(bits, sf, codingBits)
            for (symbol in symbols) parts.add(symbolWave(symbol, sf))
        }
        val complexSize = parts.size * n + 2 * n
        val bb = ComplexArray(DoubleArray(complexSize), DoubleArray(complexSize))
        var offset = 0
        for (part in parts) {
            part.re.copyInto(bb.re, offset)
            part.im.copyInto(bb.im, offset)
            offset += n
        }
        val upsampled = resamplePoly(bb, OVERSAMPLE, 1)
        val audio = FloatArray(upsampled.re.size)
        var peak = 0.0
        for (i in audio.indices) {
            val t = i.toDouble() / SAMPLE_RATE
            val phase = 2.0 * PI * CENTER_FREQUENCY * t
            val sample = upsampled.re[i] * cos(phase) - upsampled.im[i] * sin(phase)
            audio[i] = sample.toFloat()
            peak = max(peak, abs(sample))
        }
        val scale = 0.8 / (peak + 1e-12)
        for (i in audio.indices) audio[i] = (audio[i] * scale).toFloat()
        return audio
    }

    fun demodulate(audio: FloatArray, sf: Int = DEFAULT_SF, maxFrames: Int = 9): DemodResult {
        require(sf in 1..15)
        require(maxFrames >= 0)
        val bb = toBaseband(audio)
        val sync = synchronize(bb, sf) ?: return DemodResult(emptyList(), null)
        val n = 1 shl sf
        for (i in bb.re.indices) {
            val phase = -2.0 * PI * sync.cfo * i / n
            val re = bb.re[i]
            val im = bb.im[i]
            bb.re[i] = re * cos(phase) - im * sin(phase)
            bb.im[i] = re * sin(phase) + im * cos(phase)
        }
        val preambleData = sync.start
        var symbolIndex = 0
        val hs = ArrayList<Int>()
        val hd = ArrayList<Double>()
        val decoded = ArrayList<Frame>()
        var framesLimit = min(maxFrames, 9)
        var physicalFrame = 0
        while (physicalFrame < framesLimit) {
            val codingBits = if (physicalFrame == 0) 8 else 7
            val symbolCount = frameSymbols(sf, physicalFrame == 0)
            val bins = IntArray(symbolCount)
            var validWindow = true
            for (j in 0 until symbolCount) {
                val si = symbolIndex + j
                val predicted = trackTiming(hs, hd, -(sync.tau - roundEven(sync.tau)), si)
                val applied = roundEven(predicted)
                val start = preambleData + si * n + applied
                if (start < 0 || start + n > bb.re.size) {
                    validWindow = false
                    break
                }
                val peak = peak(bb.re, bb.im, start, n, sf, conjugateChirp = true)
                val corrected = peak.bin - applied + predicted
                bins[j] = ((roundEven(corrected) % n) + n) % n
                hs.add(si)
                hd.add(predicted - wrap(corrected, 1.0))
            }
            if (!validWindow) break
            val bits = decodeSymbols(bins, sf, codingBits, FRAME_BITS)
            val frame = unpackFrame(bits)
            decoded.add(frame)
            if (physicalFrame == 0 && frame.ok) framesLimit = min(framesLimit, frame.n)
            symbolIndex += symbolCount
            physicalFrame++
        }
        return DemodResult(decoded, sync)
    }

    fun demodulateAck(audio: FloatArray, sf: Int = DEFAULT_SF): Pair<IntArray?, SyncInfo?> {
        require(sf in 1..15)
        val bb = toBaseband(audio)
        val sync = synchronize(bb, sf) ?: return null to null
        val n = 1 shl sf
        for (i in bb.re.indices) {
            val phase = -2.0 * PI * sync.cfo * i / n
            val re = bb.re[i]
            val im = bb.im[i]
            bb.re[i] = re * cos(phase) - im * sin(phase)
            bb.im[i] = re * sin(phase) + im * cos(phase)
        }
        val hs = ArrayList<Int>()
        val hd = ArrayList<Double>()
        val bins = IntArray(ackSymbols(sf))
        for (j in bins.indices) {
            val predicted = trackTiming(hs, hd, -(sync.tau - roundEven(sync.tau)), j)
            val applied = roundEven(predicted)
            val start = sync.start + j * n + applied
            if (start < 0 || start + n > bb.re.size) return null to sync
            val peak = peak(bb.re, bb.im, start, n, sf, conjugateChirp = true)
            val corrected = peak.bin - applied + predicted
            bins[j] = ((roundEven(corrected) % n) + n) % n
            hs.add(j)
            hd.add(predicted - wrap(corrected, 1.0))
        }
        return decodeSymbols(bins, sf, 8, ACK_BITS) to sync
    }

    private fun synchronize(bb: ComplexArray, sf: Int): SyncInfo? {
        val n = 1 shl sf
        if (bb.re.size < n * 12) return null
        val peaks = ArrayList<Peak>()
        for (g in 0 until bb.re.size / n) {
            peaks.add(peak(bb.re, bb.im, g * n, n, sf, conjugateChirp = true))
        }
        var first: Int? = null
        for (g in 0 until peaks.size - 4) {
            val window = peaks.subList(g, g + 4)
            if (window.all { it.ratio > 7.0 } &&
                window.all { abs(wrap(it.bin - window[0].bin, n.toDouble())) < 2.5 }
            ) {
                first = g
                break
            }
        }
        val g0 = first ?: return null
        var anchor = g0 * n - roundEven(wrap(peaks[g0].bin, n.toDouble()))
        while (anchor < 0) anchor += n
        var downIndex: Int? = null
        for (i in 0 until 16) {
            val start = anchor + i * n
            if (start + n > bb.re.size) return null
            val up = peak(bb.re, bb.im, start, n, sf, conjugateChirp = true).ratio
            val down = peak(bb.re, bb.im, start, n, sf, conjugateChirp = false).ratio
            if (down > up && down > 7.0) {
                downIndex = i
                break
            }
        }
        val iDown = downIndex ?: return null
        if (iDown < 1) return null
        val upBin = wrap(
            peak(bb.re, bb.im, anchor + (iDown - 1) * n, n, sf, true).bin,
            n.toDouble()
        )
        val downBin = wrap(
            peak(bb.re, bb.im, anchor + iDown * n, n, sf, false).bin,
            n.toDouble()
        )
        val cfo = wrap((upBin + downBin) / 2.0, n / 2.0)
        val tau = (upBin - downBin) / 2.0
        val start = anchor + (iDown + 2) * n - roundEven(tau)

        val ratios = ArrayList<Double>(PREAMBLE_UP)
        for (k in 0 until PREAMBLE_UP) {
            val segmentStart = anchor + k * n
            if (segmentStart >= 0 && segmentStart + n <= bb.re.size) {
                ratios.add(peak(bb.re, bb.im, segmentStart, n, sf, true).ratio)
            }
        }
        if (ratios.isEmpty()) return null
        val meanRatio = ratios.average()
        val snrDb = 20.0 * log10(meanRatio)
        val snr4 = roundEven(snrDb / 2.0).coerceIn(0, 15)
        return SyncInfo(start, cfo, tau, snrDb, snr4)
    }

    private fun peak(
        re: DoubleArray,
        im: DoubleArray,
        start: Int,
        size: Int,
        sf: Int,
        conjugateChirp: Boolean
    ): Peak {
        val (chirpRe, chirpIm) = chirp(sf)
        val mixedRe = DoubleArray(size)
        val mixedIm = DoubleArray(size)
        for (i in 0 until size) {
            val j = if (conjugateChirp) -chirpIm[i] else chirpIm[i]
            mixedRe[i] = re[start + i] * chirpRe[i] - im[start + i] * j
            mixedIm[i] = re[start + i] * j + im[start + i] * chirpRe[i]
        }
        fft(mixedRe, mixedIm)
        val magnitudes = DoubleArray(size) { i -> sqrt(mixedRe[i] * mixedRe[i] + mixedIm[i] * mixedIm[i]) }
        var p = 0
        for (i in 1 until size) if (magnitudes[i] > magnitudes[p]) p = i
        val sorted = magnitudes.copyOf()
        sorted.sort()
        val median = if (size % 2 == 0) {
            (sorted[size / 2 - 1] + sorted[size / 2]) / 2.0
        } else sorted[size / 2]
        val left = magnitudes[(p - 1 + size) % size]
        val right = magnitudes[(p + 1) % size]
        val denominator = left - 2.0 * magnitudes[p] + right
        val delta = if (denominator != 0.0) 0.5 * (left - right) / denominator else 0.0
        return Peak(p + delta, magnitudes[p] / (median + 1e-12))
    }

    private fun trackTiming(hs: List<Int>, hd: List<Double>, initial: Double, symbol: Int): Double {
        if (hs.size < 12) {
            if (hs.size < 3) return initial
            val sorted = hd.sorted()
            return if (sorted.size % 2 == 0) {
                (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2.0
            } else sorted[sorted.size / 2]
        }
        val from = max(0, hs.size - 48)
        val count = hs.size - from
        var meanX = 0.0
        var meanY = 0.0
        for (i in from until hs.size) {
            meanX += hs[i]
            meanY += hd[i]
        }
        meanX /= count
        meanY /= count
        var covariance = 0.0
        var variance = 0.0
        for (i in from until hs.size) {
            val dx = hs[i] - meanX
            covariance += dx * (hd[i] - meanY)
            variance += dx * dx
        }
        val slope = if (variance == 0.0) 0.0 else covariance / variance
        return meanY - slope * meanX + slope * symbol
    }

    private fun wrap(value: Double, period: Double): Double {
        val remainder = (value + period / 2.0) % period
        return (if (remainder < 0.0) remainder + period else remainder) - period / 2.0
    }

    private fun roundEven(value: Double): Int = round(value).toInt()

    private fun fft(re: DoubleArray, im: DoubleArray) {
        val n = re.size
        require(n > 0 && (n and (n - 1)) == 0) { "FFT length must be a power of two" }
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                val r = re[i]
                re[i] = re[j]
                re[j] = r
                val v = im[i]
                im[i] = im[j]
                im[j] = v
            }
        }
        var length = 2
        while (length <= n) {
            val angle = -2.0 * PI / length
            val stepRe = cos(angle)
            val stepIm = sin(angle)
            for (block in 0 until n step length) {
                var wRe = 1.0
                var wIm = 0.0
                for (k in 0 until length / 2) {
                    val even = block + k
                    val odd = even + length / 2
                    val oddRe = re[odd] * wRe - im[odd] * wIm
                    val oddIm = re[odd] * wIm + im[odd] * wRe
                    re[odd] = re[even] - oddRe
                    im[odd] = im[even] - oddIm
                    re[even] += oddRe
                    im[even] += oddIm
                    val nextRe = wRe * stepRe - wIm * stepIm
                    wIm = wRe * stepIm + wIm * stepRe
                    wRe = nextRe
                }
            }
            length = length shl 1
        }
    }

    private fun toBaseband(audio: FloatArray): ComplexArray {
        val re = DoubleArray(audio.size)
        val im = DoubleArray(audio.size)
        for (i in audio.indices) {
            val phase = -2.0 * PI * CENTER_FREQUENCY * i / SAMPLE_RATE
            re[i] = 2.0 * audio[i] * cos(phase)
            im[i] = 2.0 * audio[i] * sin(phase)
        }
        return resamplePoly(ComplexArray(re, im), 1, OVERSAMPLE)
    }

    private fun resamplePoly(input: ComplexArray, up: Int, down: Int): ComplexArray {
        val maxRate = max(up, down)
        val halfLength = 10 * maxRate
        val filterLength = 2 * halfLength + 1
        val cutoff = 1.0 / maxRate
        val beta = 5.0
        val windowDenominator = besselI0(beta)
        val filter = DoubleArray(filterLength) { i ->
            val x = i - halfLength
            val ideal = if (x == 0) cutoff else sin(PI * cutoff * x) / (PI * x)
            val ratio = x.toDouble() / halfLength
            val window = besselI0(beta * sqrt(max(0.0, 1.0 - ratio * ratio))) / windowDenominator
            ideal * window
        }
        val normalization = filter.sum()
        for (i in filter.indices) filter[i] = filter[i] * up / normalization
        val outputSize = ceil(input.re.size.toDouble() * up / down).toInt()
        val outRe = DoubleArray(outputSize)
        val outIm = DoubleArray(outputSize)
        for (out in 0 until outputSize) {
            val time = out * down
            val first = max(0, ceil((time - halfLength).toDouble() / up).toInt())
            val last = min(input.re.lastIndex, floor((time + halfLength).toDouble() / up).toInt())
            for (source in first..last) {
                val filterIndex = time - source * up + halfLength
                if (filterIndex in filter.indices) {
                    val coefficient = filter[filterIndex]
                    outRe[out] += input.re[source] * coefficient
                    outIm[out] += input.im[source] * coefficient
                }
            }
        }
        return ComplexArray(outRe, outIm)
    }

    private fun besselI0(x: Double): Double {
        var sum = 1.0
        var term = 1.0
        val quarter = x * x / 4.0
        for (k in 1..40) {
            term *= quarter / (k * k)
            sum += term
            if (term < sum * 1e-16) break
        }
        return sum
    }

    /** Mean energy in a Hann-windowed FFT band, in linear power units. */
    fun bandEnergy(
        samples: FloatArray,
        sampleRate: Int = SAMPLE_RATE,
        centerFrequency: Double = CENTER_FREQUENCY.toDouble(),
        bandwidth: Double = BANDWIDTH.toDouble()
    ): Double {
        require(samples.size >= 2 && sampleRate > 0 && bandwidth > 0.0)
        val size = Integer.highestOneBit(min(1024, samples.size)).coerceAtLeast(2)
        val re = DoubleArray(size)
        val im = DoubleArray(size)
        val start = (samples.size - size) / 2
        for (i in 0 until size) {
            val window = 0.5 - 0.5 * cos(2.0 * PI * i / (size - 1))
            re[i] = samples[start + i] * window
        }
        fft(re, im)
        var power = 0.0
        var count = 0
        for (bin in 0..size / 2) {
            val frequency = bin.toDouble() * sampleRate / size
            if (abs(frequency - centerFrequency) <= bandwidth / 2.0) {
                power += re[bin] * re[bin] + im[bin] * im[bin]
                count++
            }
        }
        return if (count == 0) 0.0 else power / (count * size * size)
    }

    fun stage1BandEnergyGate(bandPower: Double, runningNoiseFloor: Double, marginDb: Double = 6.0): Boolean {
        require(bandPower >= 0.0 && runningNoiseFloor >= 0.0)
        return bandPower > runningNoiseFloor * 10.0.pow(marginDb / 10.0)
    }

    /** Peak-to-median ratio for a single N-sample up-chirp dechirp FFT. */
    fun upChirpPeakMedianRatio(samples: FloatArray, sf: Int = DEFAULT_SF): Double? {
        return upChirpPeakMedianRatio(
            DoubleArray(samples.size) { samples[it].toDouble() },
            DoubleArray(samples.size),
            sf
        )
    }

    fun upChirpPeakMedianRatio(
        real: DoubleArray,
        imaginary: DoubleArray,
        sf: Int = DEFAULT_SF
    ): Double? {
        require(sf in 1..15)
        val n = 1 shl sf
        if (real.size != n || imaginary.size != n) return null
        val (chirpRe, chirpIm) = chirp(sf)
        val re = DoubleArray(n)
        val im = DoubleArray(n)
        for (i in 0 until n) {
            re[i] = real[i] * chirpRe[i] + imaginary[i] * chirpIm[i]
            im[i] = imaginary[i] * chirpRe[i] - real[i] * chirpIm[i]
        }
        fft(re, im)
        val mags = DoubleArray(n) { sqrt(re[it] * re[it] + im[it] * im[it]) }
        val sorted = mags.copyOf()
        sorted.sort()
        val median = (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        return mags.maxOrNullCompat() / (median + 1e-12)
    }

    fun stage2UpChirpGate(samples: FloatArray, sf: Int = DEFAULT_SF, threshold: Double = 7.0): Boolean =
        (upChirpPeakMedianRatio(samples, sf) ?: 0.0) > threshold

    fun stage2UpChirpGate(
        real: DoubleArray,
        imaginary: DoubleArray,
        sf: Int = DEFAULT_SF,
        threshold: Double = 7.0
    ): Boolean = (upChirpPeakMedianRatio(real, imaginary, sf) ?: 0.0) > threshold

    /** Finds an up-chirp in captured RF audio using the same baseband PHY path as demodulation. */
    fun stage2AudioUpChirpGate(
        audio: FloatArray,
        sf: Int = DEFAULT_SF,
        threshold: Double = 7.0
    ): Boolean {
        require(sf in 1..15)
        val symbolSize = 1 shl sf
        if (audio.size < symbolSize * OVERSAMPLE) return false
        val baseband = toBaseband(audio)
        if (baseband.re.size < symbolSize) return false
        val step = max(1, symbolSize / 8)
        for (start in 0..baseband.re.size - symbolSize step step) {
            if (stage2UpChirpGate(
                    baseband.re.copyOfRange(start, start + symbolSize),
                    baseband.im.copyOfRange(start, start + symbolSize),
                    sf,
                    threshold
                )
            ) return true
        }
        return false
    }

    /** Maintains a smoothed noise estimate and only learns from rejected windows. */
    class RunningNoiseFloor(initialPower: Double = 0.0, private val smoothing: Double = 0.05) {
        var power: Double = initialPower
            private set

        init {
            require(initialPower >= 0.0 && smoothing in 0.0..1.0)
        }

        fun accept(bandPower: Double, marginDb: Double = 6.0): Boolean {
            require(bandPower >= 0.0)
            if (power == 0.0) {
                power = bandPower
                return false
            }
            val detected = stage1BandEnergyGate(bandPower, power, marginDb)
            if (!detected) {
                power = (1.0 - smoothing) * power + smoothing * bandPower
            }
            return detected
        }
    }

    private fun DoubleArray.maxOrNullCompat(): Double {
        var maxValue = Double.NEGATIVE_INFINITY
        for (value in this) if (value > maxValue) maxValue = value
        return maxValue
    }
}
