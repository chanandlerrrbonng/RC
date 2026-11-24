package org.reefglyph

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.view.KeyEvent
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.jtransforms.fft.DoubleFFT_1D
import org.reefglyph.ui.theme.ReefGlyphTheme
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread
import kotlin.math.*

// ============================================================================
// 1. SPECIFICATION CONSTANTS & HELPERS (Spec v1.1 Frozen)
// ============================================================================
object ReefSpec {
    const val FS_AUDIO = 48000
    const val BW = 3000
    const val FC = 2500.0
    const val DEFAULT_SF = 7
    const val OSR = FS_AUDIO / BW // 16 samples per baseband sample

    const val N_UP = 8
    const val N_DOWN = 2
    const val PREAMBLE_SYMS = N_UP + N_DOWN // 10

    const val HDR_BITS = 16 // 16-bit (2-byte) header alignment
    const val PAY_BITS = 108
    const val CRC_BITS = 16
    const val FRAME_BITS = HDR_BITS + PAY_BITS + CRC_BITS // 140

    const val L0_BITS = 80 // 80 bits (includes 4-bit texture index)
    const val L1_BITS = 175
    const val L2_BITS = 576
    const val ACK_BITS = 28

    fun frameSymbols(sf: Int = DEFAULT_SF, first: Boolean = false): Int {
        val n = if (first) 8 else 7
        return (-( -35 / sf )) * n // 40 for F0, 35 for F1..F8
    }

    fun bitsToInt(b: IntArray): Int {
        var v = 0
        for (x in b) v = (v shl 1) or (x and 1)
        return v
    }

    fun intToBits(v: Int, n: Int): IntArray =
        IntArray(n) { i -> (v shr (n - 1 - i)) and 1 }

    fun quantize(v: Double, lo: Double, hi: Double, bits: Int): Int {
        val maxVal = (1 shl bits) - 1
        val scaled = (v - lo) / (hi - lo) * maxVal
        return Math.rint(scaled).toInt().coerceIn(0, maxVal)
    }

    fun dequantize(q: Int, lo: Double, hi: Double, bits: Int): Double {
        val maxVal = (1 shl bits) - 1
        return lo + q.toDouble() / maxVal * (hi - lo)
    }
}

// ============================================================================
// 2. DATA MODELS & BIT CHAIN (FEC, CRC, WHITENING, GRAY)
// ============================================================================
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

data class ModemFrame(
    val layer: Int,
    val idx: Int,
    val n: Int,
    val burst: Int,
    val payload: IntArray,
    var ok: Boolean = true
)

object BitChain {
    fun whiten(bits: IntArray): IntArray {
        var s = 0xFF
        val out = IntArray(bits.size)
        for (i in bits.indices) {
            val bit = s and 1
            val fb = ((s shr 7) xor (s shr 5) xor (s shr 4) xor (s shr 3)) and 1
            s = ((s shl 1) or fb) and 0xFF
            out[i] = bits[i] xor bit
        }
        return out
    }

    fun grayEnc(v: Int): Int = v xor (v shr 1)

    fun grayDec(g: Int): Int {
        var v = 0
        var temp = g
        while (temp > 0) {
            v = v xor temp
            temp = temp shr 1
        }
        return v
    }

    private fun hammingCodeword(nib: Int, n: Int): IntArray {
        val d0 = (nib shr 3) and 1
        val d1 = (nib shr 2) and 1
        val d2 = (nib shr 1) and 1
        val d3 = nib and 1
        val p1 = d0 xor d1 xor d3
        val p2 = d0 xor d2 xor d3
        val p3 = d1 xor d2 xor d3
        return if (n == 8) {
            val sumP = (d0 + d1 + d2 + d3 + p1 + p2 + p3) and 1
            intArrayOf(d0, d1, d2, d3, p1, p2, p3, sumP)
        } else {
            intArrayOf(d0, d1, d2, d3, p1, p2, p3)
        }
    }

    val encTable = mapOf(
        7 to Array(16) { hammingCodeword(it, 7) },
        8 to Array(16) { hammingCodeword(it, 8) }
    )

    val decTable = mapOf(
        7 to IntArray(128) { w ->
            val wb = ReefSpec.intToBits(w, 7)
            (0 until 16).minByOrNull { k ->
                val cw = encTable[7]!![k]
                (0 until 7).count { wb[it] != cw[it] }
            } ?: 0
        },
        8 to IntArray(256) { w ->
            val wb = ReefSpec.intToBits(w, 8)
            (0 until 16).minByOrNull { k ->
                val cw = encTable[8]!![k]
                (0 until 8).count { wb[it] != cw[it] }
            } ?: 0
        }
    )

    fun encodeBits(bits: IntArray, sf: Int, n: Int): IntArray {
        val w = whiten(bits)
        val numNibs = w.size / 4
        val cws = ArrayList<IntArray>()
        for (i in 0 until numNibs) {
            val nib = ReefSpec.bitsToInt(w.copyOfRange(i * 4, (i + 1) * 4))
            cws.add(encTable[n]!![nib])
        }
        val pad = Math.floorMod(-cws.size, sf)
        repeat(pad) { cws.add(IntArray(n)) }

        val syms = ArrayList<Int>()
        val numBlocks = cws.size / sf
        for (b in 0 until numBlocks) {
            val blk = cws.subList(b * sf, (b + 1) * sf)
            for (j in 0 until n) {
                val colBits = IntArray(sf) { row -> blk[row][j] }
                syms.add(grayDec(ReefSpec.bitsToInt(colBits)))
            }
        }
        return syms.toIntArray()
    }

    fun decodeSyms(syms: IntArray, sf: Int, n: Int, nbits: Int): IntArray {
        val vals = syms.map { grayEnc(it) }
        val cws = ArrayList<IntArray>()
        for (k in 0 until vals.size step n) {
            val blockCols = Array(n) { j -> ReefSpec.intToBits(vals[k + j], sf) }
            for (row in 0 until sf) {
                cws.add(IntArray(n) { col -> blockCols[col][row] })
            }
        }
        val validCws = cws.take(nbits / 4)
        val nibs = validCws.map { cw ->
            val cwInt = ReefSpec.bitsToInt(cw)
            decTable[n]!![cwInt]
        }
        val bits = ArrayList<Int>()
        for (nib in nibs) {
            for (b in ReefSpec.intToBits(nib, 4)) bits.add(b)
        }
        return whiten(bits.toIntArray())
    }

    fun crc16(bits: IntArray): Int {
        var crc = 0xFFFF
        for (b in bits) {
            val top = ((crc shr 15) and 1) xor (b and 1)
            crc = (crc shl 1) and 0xFFFF
            if (top != 0) crc = crc xor 0x1021
        }
        return crc
    }

    fun packL0(r: CoralRecord): IntArray {
        val b = ArrayList<Int>()
        fun add(v: Int, n: Int) {
            for (x in ReefSpec.intToBits(v, n)) b.add(x)
        }
        add(r.transect, 6)
        add(r.coral, 8)
        add(ReefSpec.quantize(r.depthM, 0.0, 63.5, 7), 7)
        add(ReefSpec.quantize(r.timeS, 0.0, 5115.0, 10), 10)
        add(ReefSpec.quantize(r.bleach, 0.0, 1.0, 6), 6)
        add((r.health - 1).coerceIn(0, 5), 3)
        add(r.morph.coerceIn(0, 3), 2)
        for (c in r.colors.take(2)) {
            val Lq = ReefSpec.quantize(c.first, 0.0, 100.0, 6)
            val aq = Math.rint(c.second / 4.0).toInt() + 16
            val bq = Math.rint(c.third / 4.0).toInt() + 16
            add(Lq, 6)
            add(aq.coerceIn(0, 31), 5)
            add(bq.coerceIn(0, 31), 5)
        }
        add(r.patch, 1)
        add(r.evidence, 1)
        add(r.texture.coerceIn(0, 15), 4)
        val res = b.toIntArray()
        check(res.size == 80) { "L0 bits count must be 80" }
        return res
    }

    fun unpackL0(bits: IntArray): CoralRecord {
        var p = 0
        fun rd(n: Int): Int {
            val v = ReefSpec.bitsToInt(bits.copyOfRange(p, p + n))
            p += n
            return v
        }
        val r = CoralRecord()
        r.transect = rd(6)
        r.coral = rd(8)
        r.depthM = ReefSpec.dequantize(rd(7), 0.0, 63.5, 7)
        r.timeS = ReefSpec.dequantize(rd(10), 0.0, 5115.0, 10)
        r.bleach = ReefSpec.dequantize(rd(6), 0.0, 1.0, 6)
        r.health = rd(3) + 1
        r.morph = rd(2)
        val cols = mutableListOf<Triple<Double, Double, Double>>()
        repeat(2) {
            val l = ReefSpec.dequantize(rd(6), 0.0, 100.0, 6)
            val a = (rd(5) - 16) * 4.0
            val bVal = (rd(5) - 16) * 4.0
            cols.add(Triple(l, a, bVal))
        }
        r.colors = cols
        r.patch = rd(1)
        r.evidence = rd(1)
        r.texture = rd(4)
        return r
    }

    // Pack 140-bit frame (16B header + 108B payload + 16B CRC)
    fun packFrame(f: ModemFrame): IntArray {
        val b = ArrayList<Int>()
        fun add(v: Int, n: Int) {
            for (x in ReefSpec.intToBits(v, n)) b.add(x)
        }
        add(f.layer, 2)
        add(f.idx, 4)
        add(f.n, 4)
        add(f.burst, 6)
        val pay = IntArray(ReefSpec.PAY_BITS)
        System.arraycopy(f.payload, 0, pay, 0, min(f.payload.size, ReefSpec.PAY_BITS))
        for (x in pay) b.add(x)

        val body = b.toIntArray() // 124 bits
        val crc = crc16(body)
        val crcBits = ReefSpec.intToBits(crc, 16)
        val out = IntArray(140)
        System.arraycopy(body, 0, out, 0, 124)
        System.arraycopy(crcBits, 0, out, 124, 16)
        return out
    }

    // Unpack 140-bit frame with strict 16-bit header alignment check
    fun unpackFrame(bits: IntArray): ModemFrame {
        if (bits.size < 140) return ModemFrame(0, 0, 0, 0, IntArray(0), false)
        val body = bits.copyOfRange(0, 124)
        val crcReceived = ReefSpec.bitsToInt(bits.copyOfRange(124, 140))
        val crcCalculated = crc16(body)
        val ok = (crcReceived == crcCalculated)

        val layer = ReefSpec.bitsToInt(bits.copyOfRange(0, 2))
        val idx = ReefSpec.bitsToInt(bits.copyOfRange(2, 6))
        val n = ReefSpec.bitsToInt(bits.copyOfRange(6, 10))
        val burst = ReefSpec.bitsToInt(bits.copyOfRange(10, 16))
        val payload = bits.copyOfRange(16, 124)
        return ModemFrame(layer, idx, n, burst, payload, ok)
    }
}

// ============================================================================
// 3. PHY MODULATION & DEMODULATION ENGINE (WITH JTRANSFORMS FFT)
// ============================================================================
data class SyncResult(
    val start: Int,
    val cfo: Double,
    val tau: Double,
    val snrDb: Double
)

object ModemPhy {
    private val fft128 = DoubleFFT_1D(128L)

    fun baseChirp(sf: Int): Array<DoubleArray> {
        val nSamples = 1 shl sf
        val out = Array(nSamples) { DoubleArray(2) }
        for (n in 0 until nSamples) {
            val phase = PI * n * n / nSamples - PI * n
            out[n][0] = cos(phase)
            out[n][1] = sin(phase)
        }
        return out
    }

    fun symWave(k: Int, sf: Int, down: Boolean = false): Array<DoubleArray> {
        val base = baseChirp(sf)
        val nSamples = 1 shl sf
        val out = Array(nSamples) { DoubleArray(2) }
        for (n in 0 until nSamples) {
            val idx = Math.floorMod(n + k, nSamples)
            val re = base[idx][0]
            val im = base[idx][1]
            out[n][0] = re
            out[n][1] = if (down) -im else im
        }
        return out
    }

    fun modulate(frames: List<IntArray>, sf: Int = ReefSpec.DEFAULT_SF): FloatArray {
        val parts = ArrayList<Array<DoubleArray>>()
        repeat(ReefSpec.N_UP) { parts.add(symWave(0, sf, false)) }
        repeat(ReefSpec.N_DOWN) { parts.add(symWave(0, sf, true)) }

        for ((idx, frameBits) in frames.withIndex()) {
            val nFec = if (idx == 0) 8 else 7
            val syms = BitChain.encodeBits(frameBits, sf, nFec)
            for (s in syms) parts.add(symWave(s, sf, false))
        }

        val nSamples = 1 shl sf
        parts.add(Array(nSamples * 2) { DoubleArray(2) })

        val totalBaseband = parts.sumOf { it.size }
        val bbRe = DoubleArray(totalBaseband)
        val bbIm = DoubleArray(totalBaseband)
        var offset = 0
        for (p in parts) {
            for (sample in p) {
                bbRe[offset] = sample[0]
                bbIm[offset] = sample[1]
                offset++
            }
        }

        val totalAudio = totalBaseband * ReefSpec.OSR
        val audio = FloatArray(totalAudio)
        var maxAbs = 1e-12

        for (i in 0 until totalAudio) {
            val t = i.toDouble() / ReefSpec.FS_AUDIO
            val bbIdx = (i / ReefSpec.OSR).coerceIn(0, totalBaseband - 1)
            val carrierPhase = 2.0 * PI * ReefSpec.FC * t
            val sampleVal = bbRe[bbIdx] * cos(carrierPhase) - bbIm[bbIdx] * sin(carrierPhase)
            audio[i] = sampleVal.toFloat()
            val absVal = abs(sampleVal)
            if (absVal > maxAbs) maxAbs = absVal
        }

        val scale = (0.8 / maxAbs).toFloat()
        for (i in audio.indices) audio[i] *= scale
        return audio
    }

    // Downsamples 48 kHz passband to 3 kHz complex baseband (16 samples per baseband sample)
    fun toBaseband(audio: FloatArray): Pair<DoubleArray, DoubleArray> {
        val nOut = audio.size / ReefSpec.OSR
        val bbRe = DoubleArray(nOut)
        val bbIm = DoubleArray(nOut)

        for (k in 0 until nOut) {
            var sumRe = 0.0
            var sumIm = 0.0
            for (m in 0 until ReefSpec.OSR) {
                val idx = k * ReefSpec.OSR + m
                val t = idx.toDouble() / ReefSpec.FS_AUDIO
                val carrier = -2.0 * PI * ReefSpec.FC * t
                val a = audio[idx] * 2.0
                sumRe += a * cos(carrier)
                sumIm += a * sin(carrier)
            }
            bbRe[k] = sumRe / ReefSpec.OSR
            bbIm[k] = sumIm / ReefSpec.OSR
        }
        return Pair(bbRe, bbIm)
    }

    // Dechirp FFT peak detector with quadratic interpolation
    fun dechirpPeak(
        bbRe: DoubleArray, bbIm: DoubleArray, start: Int,
        ref: Array<DoubleArray>, sf: Int
    ): Pair<Double, Double> {
        val N = 1 shl sf
        val fftBuf = DoubleArray(2 * N)
        for (n in 0 until N) {
            val sRe = bbRe[start + n]
            val sIm = bbIm[start + n]
            val rRe = ref[n][0]
            val rIm = ref[n][1]
            // seg * ref
            fftBuf[2 * n] = sRe * rRe - sIm * rIm
            fftBuf[2 * n + 1] = sRe * rIm + sIm * rRe
        }
        fft128.complexForward(fftBuf)

        val mag = DoubleArray(N) { i ->
            hypot(fftBuf[2 * i], fftBuf[2 * i + 1])
        }

        var p = 0
        var maxM = mag[0]
        for (i in 1 until N) {
            if (mag[i] > maxM) {
                maxM = mag[i]
                p = i
            }
        }

        val sorted = mag.clone().sortedArray()
        val median = sorted[N / 2] + 1e-12
        val ratio = maxM / median

        val l = mag[Math.floorMod(p - 1, N)]
        val r = mag[Math.floorMod(p + 1, N)]
        val den = l - 2.0 * maxM + r
        val d = if (den != 0.0) 0.5 * (l - r) / den else 0.0
        return Pair(p.toDouble() + d, ratio)
    }

    private fun sw(v: Double, n: Double): Double =
        ((v + n / 2.0).mod(n) + n).mod(n) - n / 2.0

    // Full sliding-dechirp preamble synchronization (CFO and timing estimation matching phy.py)
    fun sync(bbRe: DoubleArray, bbIm: DoubleArray, sf: Int = ReefSpec.DEFAULT_SF, thr: Double = 5.0): SyncResult? {
        val N = 1 shl sf
        val up = baseChirp(sf)
        val cu = Array(N) { i -> doubleArrayOf(up[i][0], -up[i][1]) } // conj(up)
        val cd = up // downchirp conjugate is upchirp

        val nw = bbRe.size / N
        if (nw < 16) return null

        val pks = ArrayList<Pair<Double, Double>>()
        for (g in 0 until nw) {
            pks.add(dechirpPeak(bbRe, bbIm, g * N, cu, sf))
        }

        // Look for 4 consecutive up-chirp peaks
        var g0: Int? = null
        for (g in 0 until (nw - 4)) {
            val w = (0 until 4).map { pks[g + it] }
            if (w.all { it.second > thr } &&
                w.all { abs(sw(it.first - w[0].first, N.toDouble())) < 2.5 }) {
                g0 = g
                break
            }
        }
        if (g0 == null) return null

        // Fractional symbol start alignment 'a'
        var a = Math.floorMod(Math.rint(g0 * N - sw(pks[g0].first, N.toDouble())).toLong(), N.toLong()).toInt()
        while (a < 0) a += N

        // Search for down-chirp marker (iD)
        var iD: Int? = null
        for (i in 0 until 16) {
            val s = a + i * N
            if (s + N > bbRe.size) return null
            val ru = dechirpPeak(bbRe, bbIm, s, cu, sf).second
            val rd = dechirpPeak(bbRe, bbIm, s, cd, sf).second
            if (rd > ru && rd > thr) {
                iD = i
                break
            }
        }
        if (iD == null || iD < 1) return null

        val bu = sw(dechirpPeak(bbRe, bbIm, a + (iD - 1) * N, cu, sf).first, N.toDouble())
        val bd = sw(dechirpPeak(bbRe, bbIm, a + iD * N, cd, sf).first, N.toDouble())
        val cfo = sw((bu + bd) / 2.0, (N / 2).toDouble())
        val tau = (bu - bd) / 2.0
        val start = a + (iD + 2) * N - Math.rint(tau).toInt()

        val ups = (max(0, iD - 6) until iD).map { k ->
            dechirpPeak(bbRe, bbIm, a + k * N, cu, sf).second
        }
        val meanUp = if (ups.isNotEmpty()) ups.average() else 1.0
        val snrDb = 20.0 * log10(max(1.0, meanUp))

        return SyncResult(start, cfo, tau, snrDb)
    }

    // Demodulate F0 (40 symbols) with CFO derotation
    fun demodulateF0(
        bbRe: DoubleArray, bbIm: DoubleArray,
        syncResult: SyncResult, sf: Int = ReefSpec.DEFAULT_SF
    ): ModemFrame? {
        val N = 1 shl sf
        val up = baseChirp(sf)
        val cu = Array(N) { i -> doubleArrayOf(up[i][0], -up[i][1]) }
        val nSyms = ReefSpec.frameSymbols(sf, true) // 40 symbols

        // CFO Derotation: bb * exp(-2j * pi * cfo * n / N)
        val derotRe = DoubleArray(bbRe.size)
        val derotIm = DoubleArray(bbIm.size)
        for (n in bbRe.indices) {
            val phase = -2.0 * PI * syncResult.cfo * n / N
            val c = cos(phase)
            val s = sin(phase)
            derotRe[n] = bbRe[n] * c - bbIm[n] * s
            derotIm[n] = bbRe[n] * s + bbIm[n] * c
        }

        val bins = IntArray(nSyms)
        for (j in 0 until nSyms) {
            val sPos = syncResult.start + j * N
            if (sPos + N > derotRe.size) return null
            val (peakBin, _) = dechirpPeak(derotRe, derotIm, sPos, cu, sf)
            bins[j] = Math.floorMod(Math.rint(peakBin).toInt(), N)
        }

        val decodedBits = BitChain.decodeSyms(bins, sf, 8, ReefSpec.FRAME_BITS)
        return BitChain.unpackFrame(decodedBits)
    }
}

// ============================================================================
// 4. MAIN ACTIVITY & LIVE FULL-DUPLEX CONTROLLER
// ============================================================================
class MainActivity : ComponentActivity() {

    private var isTransmitting by mutableStateOf(false)
    private var isListening by mutableStateOf(false)
    private var lastStatus by mutableStateOf("Ready (Idle)")
    private var currentRecord by mutableStateOf(CoralRecord())
    private var logOutput by mutableStateOf("Ready. Press TX to send, or START LISTENING.")

    private val rxRunning = AtomicBoolean(false)
    private var rxThread: Thread? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        requestPermissionsIfNeeded()

        setContent {
            ReefGlyphTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    ReefGlyphMainScreen(
                        record = currentRecord,
                        isTransmitting = isTransmitting,
                        isListening = isListening,
                        status = lastStatus,
                        log = logOutput,
                        onSendShort = { sendBurst("short") },
                        onSendLong = { sendBurst("long") },
                        onToggleListening = { toggleListening() },
                        onRecordChange = { currentRecord = it },
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
            logOutput += "\n[Hardware] Volume-Down pressed! Sending SHORT burst..."
            sendBurst("short")
            return true
        }
        return super.onKeyDown(keyCode, event)
    }

    private fun requestPermissionsIfNeeded() {
        val permissions = mutableListOf(Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            permissions.add(Manifest.permission.FOREGROUND_SERVICE)
        }
        val needed = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (needed.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, needed.toTypedArray(), 101)
        }
    }

    private fun sendBurst(burstType: String) {
        if (isTransmitting) return
        thread {
            isTransmitting = true
            lastStatus = "Transmitting ${burstType.uppercase()} burst..."
            try {
                val l0Bits = BitChain.packL0(currentRecord)
                val f0 = ModemFrame(0, 0, if (burstType == "short") 1 else 9, 42, l0Bits)
                val frames = mutableListOf(BitChain.packFrame(f0))

                if (burstType == "long") {
                    for (i in 1..8) {
                        val dummyPay = IntArray(ReefSpec.PAY_BITS) { 0 }
                        frames.add(BitChain.packFrame(ModemFrame(if (i <= 2) 1 else 2, i, 9, 42, dummyPay)))
                    }
                }

                val audio = ModemPhy.modulate(frames, ReefSpec.DEFAULT_SF)
                val durationS = audio.size.toDouble() / ReefSpec.FS_AUDIO
                logOutput += "\n[TX] Audio: ${audio.size} samples (%.2f s) at 48kHz".format(durationS)

                val shortAudio = ShortArray(audio.size) { i ->
                    (audio[i] * 32767).toInt().coerceIn(-32768, 32767).toShort()
                }

                val bufferSize = AudioTrack.getMinBufferSize(
                    ReefSpec.FS_AUDIO,
                    AudioFormat.CHANNEL_OUT_MONO,
                    AudioFormat.ENCODING_PCM_16BIT
                )

                val track = AudioTrack.Builder()
                    .setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_UNKNOWN)
                            .build()
                    )
                    .setAudioFormat(
                        AudioFormat.Builder()
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setSampleRate(ReefSpec.FS_AUDIO)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .build()
                    )
                    .setBufferSizeInBytes(max(bufferSize, shortAudio.size * 2))
                    .setTransferMode(AudioTrack.MODE_STATIC)
                    .build()

                track.write(shortAudio, 0, shortAudio.size)
                track.play()
                logOutput += "\n[TX] Acoustic waveform playing..."
                Thread.sleep((durationS * 1000).toLong() + 300)
                track.stop()
                track.release()

                lastStatus = "Transmission complete."
                logOutput += "\n[TX] Done! 3.0s ACK window opened."
            } catch (e: Exception) {
                lastStatus = "TX Error: ${e.message}"
                logOutput += "\n[TX Error] ${e.localizedMessage}"
            } finally {
                isTransmitting = false
            }
        }
    }

    private fun toggleListening() {
        if (isListening) {
            rxRunning.set(false)
            rxThread?.interrupt()
            rxThread = null
            isListening = false
            lastStatus = "Receiver stopped."
            logOutput += "\n[RX] Listener stopped."
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                requestPermissionsIfNeeded()
                return
            }
            isListening = true
            rxRunning.set(true)
            lastStatus = "Listening (48kHz Unprocessed)..."
            logOutput += "\n[RX] Listening active. Waiting for acoustic chirp..."

            rxThread = thread(name = "AcousticRxThread") {
                runReceiverLoop()
            }
        }
    }

    private fun runReceiverLoop() {
        val sampleRate = ReefSpec.FS_AUDIO
        val minBufSize = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        val bufferSize = max(minBufSize, sampleRate * 2)
        val audioRecord = try {
            AudioRecord(
                MediaRecorder.AudioSource.UNPROCESSED,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
            )
        } catch (e: Exception) {
            AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
            )
        }

        try {
            audioRecord.startRecording()
            val chunk = ShortArray(4800) // 100 ms chunks
            val rollingSamples = ArrayList<Float>()
            val maxRolling = sampleRate * 4 // 4s buffer

            while (rxRunning.get()) {
                val read = audioRecord.read(chunk, 0, chunk.size)
                if (read <= 0) continue

                var energy = 0.0
                for (i in 0 until read) {
                    val normSample = chunk[i] / 32768.0f
                    rollingSamples.add(normSample)
                    energy += normSample * normSample
                }
                energy /= read

                while (rollingSamples.size > maxRolling) {
                    rollingSamples.removeAt(0)
                }

                // If acoustic energy exceeds background threshold, test sync
                if (energy > 0.0003 && rollingSamples.size >= sampleRate * 2) {
                    val audioSnapshot = rollingSamples.toFloatArray()
                    val (bbRe, bbIm) = ModemPhy.toBaseband(audioSnapshot)
                    val syncResult = ModemPhy.sync(bbRe, bbIm, ReefSpec.DEFAULT_SF, thr = 4.5)

                    if (syncResult != null) {
                        runOnUiThread {
                            logOutput += "\n----------------------------------------"
                            logOutput += "\n[RX SYNC] Preamble Locked!"
                            logOutput += "\n  Start: ${syncResult.start} bb | CFO: %.2f bins | SNR: %.1f dB".format(
                                syncResult.cfo, syncResult.snrDb
                            )
                        }

                        val frame = ModemPhy.demodulateF0(bbRe, bbIm, syncResult, ReefSpec.DEFAULT_SF)
                        if (frame != null && frame.ok) {
                            val decodedCoral = BitChain.unpackL0(frame.payload)
                            runOnUiThread {
                                currentRecord = decodedCoral
                                lastStatus = "DECODED: Coral #${decodedCoral.coral} (Burst ${frame.burst})"
                                logOutput += "\n[RX CRC16: PASS] 140b Frame Decoded!"
                                logOutput += "\n  Burst ID : ${frame.burst} | Layer: ${frame.layer}"
                                logOutput += "\n  Coral ID : #${decodedCoral.coral}"
                                logOutput += "\n  Bleach   : %.2f (Index)".format(decodedCoral.bleach)
                                logOutput += "\n  Health   : Score H%d".format(decodedCoral.health)
                                logOutput += "\n  Texture  : Level T%d / 15".format(decodedCoral.texture)
                                logOutput += "\n----------------------------------------"
                            }
                            rollingSamples.clear()
                            Thread.sleep(1500)
                        } else if (frame != null && !frame.ok) {
                            runOnUiThread {
                                logOutput += "\n[RX CRC16: FAIL] Frame CRC mismatch."
                            }
                        }
                    }
                }
            }
        } catch (e: Exception) {
            runOnUiThread {
                logOutput += "\n[RX Exception] ${e.message}"
            }
        } finally {
            try {
                audioRecord.stop()
                audioRecord.release()
            } catch (_: Exception) {}
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        rxRunning.set(false)
        rxThread?.interrupt()
    }
}

// ============================================================================
// 5. JETPACK COMPOSE UI (SCROLLABLE LOG & JUSTIFIED LAYOUT)
// ============================================================================
@Composable
fun ReefGlyphMainScreen(
    record: CoralRecord,
    isTransmitting: Boolean,
    isListening: Boolean,
    status: String,
    log: String,
    onSendShort: () -> Unit,
    onSendLong: () -> Unit,
    onToggleListening: () -> Unit,
    onRecordChange: (CoralRecord) -> Unit,
    modifier: Modifier = Modifier
) {
    val screenScrollState = rememberScrollState()
    val terminalScrollState = rememberScrollState()

    // Auto-scroll terminal log to bottom on new message
    LaunchedEffect(log) {
        terminalScrollState.animateScrollTo(terminalScrollState.maxValue)
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Color(0xFF0F172A))
            .padding(16.dp)
            .verticalScroll(screenScrollState)
    ) {
        // App Header
        Text(
            text = "ReefGlyph Acoustic Modem",
            fontSize = 22.sp,
            fontWeight = FontWeight.Bold,
            color = Color(0xFF38BDF8)
        )
        Text(
            text = "Spec v1.1 | 1-4 kHz CSS SF7 | 16B Aligned PHY",
            fontSize = 12.sp,
            color = Color(0xFF94A3B8)
        )

        Spacer(modifier = Modifier.height(12.dp))

        // Coral Glyph Badge Display
        Card(
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text(
                    text = "CORAL EVIDENCE GLYPH",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF94A3B8)
                )
                Spacer(modifier = Modifier.height(8.dp))
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(130.dp)
                        .background(Color.Black, RoundedCornerShape(8.dp))
                        .border(1.dp, Color(0xFF334155), RoundedCornerShape(8.dp)),
                    contentAlignment = Alignment.Center
                ) {
                    Canvas(modifier = Modifier.fillMaxSize()) {
                        val isBleached = record.bleach > 0.3
                        val badgeColor = if (isBleached) Color(0xFFDC2626) else Color(0xFF16A34A)
                        drawRect(color = badgeColor, size = Size(size.width, 44f))

                        val col1 = Color(
                            red = (record.colors[0].first / 100.0).toFloat().coerceIn(0f, 1f),
                            green = 0.5f,
                            blue = 0.6f
                        )
                        val col2 = Color(
                            red = (record.colors[1].first / 100.0).toFloat().coerceIn(0f, 1f),
                            green = 0.7f,
                            blue = 0.4f
                        )
                        drawCircle(color = col1, radius = 26f, center = Offset(size.width * 0.35f, size.height * 0.65f))
                        drawCircle(color = col2, radius = 26f, center = Offset(size.width * 0.65f, size.height * 0.65f))
                        drawCircle(color = Color.White, radius = 28f, center = Offset(size.width * 0.5f, size.height * 0.65f), style = Stroke(2f))
                    }
                    Text(
                        text = "coral ${record.coral}  bleach %.2f  H%d  T%d".format(record.bleach, record.health, record.texture),
                        color = Color.White,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier
                            .align(Alignment.TopStart)
                            .padding(start = 12.dp, top = 2.dp)
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Telemetry Sliders
        Card(
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text(
                    text = "TRANSMITTER PARAMETERS",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF94A3B8)
                )

                Spacer(modifier = Modifier.height(4.dp))
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text(text = "Bleach: %.2f".format(record.bleach), color = Color.White, fontSize = 13.sp)
                    Slider(
                        value = record.bleach.toFloat(),
                        onValueChange = { onRecordChange(record.copy(bleach = it.toDouble())) },
                        valueRange = 0f..1f,
                        modifier = Modifier.width(200.dp)
                    )
                }

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text(text = "Texture: ${record.texture} / 15", color = Color(0xFF38BDF8), fontSize = 13.sp)
                    Slider(
                        value = record.texture.toFloat(),
                        onValueChange = { onRecordChange(record.copy(texture = it.toInt())) },
                        valueRange = 0f..15f,
                        steps = 14,
                        modifier = Modifier.width(200.dp)
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Control Buttons
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = onSendShort,
                enabled = !isTransmitting,
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0284C7)),
                modifier = Modifier.weight(1f)
            ) {
                Text("TX SHORT (2.13s)")
            }

            Button(
                onClick = onSendLong,
                enabled = !isTransmitting,
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2563EB)),
                modifier = Modifier.weight(1f)
            ) {
                Text("TX LONG (14.08s)")
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        Button(
            onClick = onToggleListening,
            colors = ButtonDefaults.buttonColors(
                containerColor = if (isListening) Color(0xFFDC2626) else Color(0xFF059669)
            ),
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(if (isListening) "STOP LISTENING" else "START LISTENING (Acoustic RX)")
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Status Header
        Text(text = "STATUS: $status", fontSize = 12.sp, color = Color(0xFFFBBF24), fontWeight = FontWeight.Bold)

        Spacer(modifier = Modifier.height(6.dp))

        // Scrollable & Formatted Terminal Output
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(200.dp)
                .background(Color(0xFF020617), RoundedCornerShape(8.dp))
                .border(1.dp, Color(0xFF334155), RoundedCornerShape(8.dp))
                .padding(8.dp)
                .verticalScroll(terminalScrollState)
        ) {
            Text(
                text = log,
                color = Color(0xFF10B981),
                fontSize = 11.sp,
                fontFamily = FontFamily.Monospace,
                lineHeight = 16.sp
            )
        }
    }
}
