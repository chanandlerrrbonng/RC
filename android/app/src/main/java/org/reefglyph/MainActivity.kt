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
import android.os.SystemClock
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
import org.reefglyph.ui.theme.ReefGlyphTheme
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread
import kotlin.math.roundToInt

class AudioRingBuffer(val capacity: Int) {
    private val samples = FloatArray(capacity)
    private var writePosition = 0
    private var sampleCount = 0

    @Synchronized
    fun write(data: FloatArray) {
        for (sample in data) {
            samples[writePosition] = sample
            writePosition = (writePosition + 1) % capacity
            if (sampleCount < capacity) sampleCount++
        }
    }

    @Synchronized
    fun snapshot(): FloatArray {
        val result = FloatArray(sampleCount)
        val start = (writePosition - sampleCount + capacity) % capacity
        for (i in 0 until sampleCount) result[i] = samples[(start + i) % capacity]
        return result
    }

    @Synchronized
    fun size(): Int = sampleCount

    @Synchronized
    fun clear() {
        writePosition = 0
        sampleCount = 0
    }
}

class MainActivity : ComponentActivity() {
    private var isTransmitting by mutableStateOf(false)
    private var isListening by mutableStateOf(false)
    private var lastStatus by mutableStateOf("Ready (Idle)")
    private var currentRecord by mutableStateOf(CoralRecord())
    private var logOutput by mutableStateOf("Ready. Press TX to send, or START LISTENING.")

    private val rxRunning = AtomicBoolean(false)
    private val rxCalibrated = AtomicBoolean(false)
    private val txActive = AtomicBoolean(false)
    private val ackWindow = AtomicBoolean(false)
    private val sendInProgress = AtomicBoolean(false)
    private val activeBurstId = AtomicInteger(-1)
    private val receivedAck = AtomicReference<ModemCore.Ack?>(null)
    private val burstCounter = AtomicInteger(0)
    private var rxThread: Thread? = null

    private var gatedListen = true

    private val sampleLayers by lazy {
        mapOf(
            1 to hexToBits("10864298e84a96b635cf84653a52b6be33adf3bd5ce6", 175),
            2 to hexToBits("1b1b1b1bb1b1b1b11b1b1b1bb1b1b1b11b1b1b1bb1b1b1b11b1b1b1bb1b1b1b11b1b1b1bb1b1b1b11b1b1b1bb1b1b1b11b1b1b1bb1b1b1b11b1b1b1bb1b1b1b11b1b1b1bb1b1b1b11ae5c4892537de76", 576)
        )
    }

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
                        onSendShort = { sendBurst(short = true) },
                        onSendLong = { sendBurst(short = false) },
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
            appendLog("[Hardware] Volume-Down pressed! Sending SHORT burst...")
            sendBurst(short = true)
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

    private fun sendBurst(short: Boolean) {
        if (!sendInProgress.compareAndSet(false, true)) return
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissionsIfNeeded()
            appendLog("[TX] Audio permission is required for ACK reception.")
            sendInProgress.set(false)
            return
        }

        val burstId = burstCounter.getAndUpdate { (it + 1) and 0x3F }
        val liveL0 = packLiveL0(currentRecord.copy())
        val layers = if (short) {
            mapOf(0 to liveL0)
        } else {
            sampleLayers + (0 to liveL0)
        }
        val originalFrames = ModemCore.packetize(layers, burstId)
        isTransmitting = true
        lastStatus = "Preparing ${if (short) "SHORT" else "LONG"} burst..."

        thread(name = "AcousticTxThread") {
            try {
                var framesToSend = originalFrames
                var complete = false
                for (attempt in 1..ModemCore.MAX_ATTEMPTS) {
                    activeBurstId.set(burstId)
                    receivedAck.set(null)
                    ackWindow.set(false)
                    txActive.set(true)
                    updateStatus("Transmitting ${if (short) "SHORT" else "LONG"} burst (attempt $attempt/${ModemCore.MAX_ATTEMPTS})...")
                    appendLog("[TX] Burst $burstId: attempt $attempt, ${framesToSend.size} frame(s).")
                    val audio = ModemCore.modulate(framesToSend)
                    appendLog("[TX] Audio: ${audio.size} samples (%.2f s) at 48kHz".format(audio.size.toDouble() / ModemCore.SAMPLE_RATE))
                    playAudio(audio)

                    val modemTailMs = (2L * (1 shl ModemCore.DEFAULT_SF) * 1000L) / ModemCore.BANDWIDTH
                    Thread.sleep((300L - modemTailMs).coerceAtLeast(0L))
                    txActive.set(false)
                    appendLog("[TX] ACK listen window opened for 3.0s.")

                    val mainReceiverAvailable = rxRunning.get()
                    ackWindow.set(mainReceiverAvailable)
                    val ack = if (mainReceiverAvailable) {
                        awaitMainReceiverAck(originalFrames)
                    } else {
                        listenForAck(burstId, ACK_WINDOW_MS)
                    }
                    ackWindow.set(false)
                    if (ack == null || !ack.ok || ack.burstId != burstId) {
                        appendLog("[ARQ] No valid ACK for burst $burstId; stopping after attempt $attempt.")
                        break
                    }

                    val missing = originalFrames.indices.filter { !ack.bitmap[it] }
                    if (missing.isEmpty()) {
                        complete = true
                        appendLog("[ARQ] Burst $burstId acknowledged; all ${originalFrames.size} frame(s) received.")
                        break
                    }
                    if (attempt == ModemCore.MAX_ATTEMPTS) {
                        appendLog("[ARQ] Burst $burstId still missing frame(s) $missing after $attempt attempts.")
                        break
                    }

                    appendLog("[ARQ] Missing frame(s) $missing; selectively resending with burst ID $burstId.")
                    framesToSend = ModemCore.buildSelectiveResendBurst(originalFrames, ack.bitmap, attempt)
                    Thread.sleep(100)
                }
                updateStatus(if (complete) "Transmission acknowledged." else "Transmission finished; ACK incomplete or timed out.")
            } catch (e: InterruptedException) {
                Thread.currentThread().interrupt()
                updateStatus("Transmission interrupted.")
                appendLog("[TX] Transmission interrupted.")
            } catch (e: Exception) {
                updateStatus("TX Error: ${e.message}")
                appendLog("[TX Error] ${e.localizedMessage}")
            } finally {
                txActive.set(false)
                ackWindow.set(false)
                activeBurstId.set(-1)
                sendInProgress.set(false)
                runOnUiThread { isTransmitting = false }
            }
        }
    }

    private fun awaitMainReceiverAck(originalFrames: List<IntArray>): ModemCore.Ack? {
        val deadline = SystemClock.elapsedRealtime() + ACK_WINDOW_MS
        while (SystemClock.elapsedRealtime() < deadline && rxRunning.get()) {
            val ack = receivedAck.get()
            if (ack != null && originalFrames.indices.all { it < ack.bitmap.size && ack.bitmap[it] }) break
            Thread.sleep(25)
        }
        val ack = receivedAck.get()
        if (ack != null && originalFrames.indices.any { it >= ack.bitmap.size || !ack.bitmap[it] }) {
            while (SystemClock.elapsedRealtime() < deadline && rxRunning.get()) Thread.sleep(25)
        }
        return receivedAck.get()
    }

    private fun listenForAck(burstId: Int, durationMs: Long): ModemCore.Ack? {
        val sampleRate = ModemCore.SAMPLE_RATE
        val minBufferSize = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        val bufferSize = maxOf(minBufferSize, sampleRate * 2)
        val audioRecord = try {
            AudioRecord(
                MediaRecorder.AudioSource.UNPROCESSED,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
            )
        } catch (_: IllegalArgumentException) {
            AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
            )
        }

        try {
            check(audioRecord.state == AudioRecord.STATE_INITIALIZED) { "Could not initialize ACK microphone capture" }
            audioRecord.startRecording()
            val chunk = ShortArray(4800)
            val ringBuffer = AudioRingBuffer(sampleRate * RX_RING_SECONDS)
            val deadline = SystemClock.elapsedRealtime() + durationMs
            var decodedAck: ModemCore.Ack? = null
            while (SystemClock.elapsedRealtime() < deadline && !rxRunning.get()) {
                val read = audioRecord.read(chunk, 0, chunk.size)
                check(read > 0) { "Microphone capture failed during the ACK listen window" }
                ringBuffer.write(FloatArray(read) { chunk[it] / 32768.0f })
                if (decodedAck == null && ringBuffer.size() >= sampleRate) {
                    val (ackBits, _) = ModemCore.demodulateAck(ringBuffer.snapshot())
                    if (ackBits != null) {
                        val ack = ModemCore.unpackAck(ackBits)
                        if (ack.ok && ack.burstId == burstId) {
                            decodedAck = ack
                            appendLog("[RX] ACK for burst ${ack.burstId} decoded (SNR4=${ack.snr4}).")
                        }
                    }
                }
            }
            return decodedAck
        } finally {
            try {
                if (audioRecord.recordingState == AudioRecord.RECORDSTATE_RECORDING) audioRecord.stop()
            } finally {
                audioRecord.release()
            }
        }
    }

    private fun toggleListening() {
        if (rxRunning.get()) {
            rxRunning.set(false)
            rxCalibrated.set(false)
            rxThread?.interrupt()
            rxThread = null
            isListening = false
            lastStatus = "Receiver stopped."
            appendLog("[RX] Listener stopped.")
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                requestPermissionsIfNeeded()
                return
            }
            startListening()
        }
    }

    private fun startListening() {
        rxCalibrated.set(false)
        rxRunning.set(true)
        isListening = true
        lastStatus = "Listening (48kHz Unprocessed)..."
        appendLog("[RX] Listening active. Calibrating noise floor for 1s...")
        rxThread = thread(name = "AcousticRxThread") { runReceiverLoop() }
    }

    private fun runReceiverLoop() {
        val sampleRate = ModemCore.SAMPLE_RATE
        val minBufferSize = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        val bufferSize = maxOf(minBufferSize, sampleRate * 2)
        val audioRecord = try {
            AudioRecord(
                MediaRecorder.AudioSource.UNPROCESSED,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
            )
        } catch (_: IllegalArgumentException) {
            AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
            )
        }

        try {
            check(audioRecord.state == AudioRecord.STATE_INITIALIZED) { "Could not initialize microphone capture" }
            audioRecord.startRecording()
            val chunk = ShortArray(4800)
            val ringBuffer = AudioRingBuffer(sampleRate * RX_RING_SECONDS)
            val reassembler = ModemCore.Reassembler()
            var noiseFloor: ModemCore.RunningNoiseFloor? = null
            var calibrationPower = 0.0
            var calibrationChunks = 0
            var calibrationSamples = 0
            var previousGateSamples = FloatArray(0)
            var candidate = false
            var lastSignalAt = 0L
            var lastUngatedSearchAt = 0L

            while (rxRunning.get()) {
                val read = audioRecord.read(chunk, 0, chunk.size)
                if (read <= 0) continue
                if (txActive.get()) {
                    ringBuffer.clear()
                    candidate = false
                    previousGateSamples = FloatArray(0)
                    continue
                }

                val audioChunk = FloatArray(read) { chunk[it] / 32768.0f }
                ringBuffer.write(audioChunk)

                if (ackWindow.get()) {
                    if (receivedAck.get() == null && ringBuffer.size() >= sampleRate) {
                        val (ackBits, _) = ModemCore.demodulateAck(ringBuffer.snapshot())
                        if (ackBits != null) {
                            val ack = ModemCore.unpackAck(ackBits)
                            if (ack.ok && ack.burstId == activeBurstId.get()) {
                                receivedAck.compareAndSet(null, ack)
                                appendLog("[RX] ACK for burst ${ack.burstId} decoded (SNR4=${ack.snr4}).")
                            }
                        }
                    }
                    continue
                }

                val bandPower = ModemCore.bandEnergy(audioChunk)
                if (noiseFloor == null) {
                    calibrationPower += bandPower
                    calibrationChunks++
                    calibrationSamples += read
                    if (calibrationSamples >= sampleRate) {
                        noiseFloor = ModemCore.RunningNoiseFloor(calibrationPower / calibrationChunks)
                        rxCalibrated.set(true)
                        appendLog("[RX] Noise-floor calibration complete.")
                    }
                    previousGateSamples = tailSamples(previousGateSamples, audioChunk)
                    continue
                }

                val energyDetected = noiseFloor.accept(bandPower)
                val now = SystemClock.elapsedRealtime()
                if (candidate && energyDetected) lastSignalAt = now

                val gateSamples = concatenate(previousGateSamples, audioChunk)
                if (!candidate && gatedListen && energyDetected && ModemCore.stage2AudioUpChirpGate(gateSamples)) {
                    candidate = true
                    lastSignalAt = now
                    appendLog("[RX] Stage 1 energy and Stage 2 up-chirp gates passed.")
                } else if (!candidate && !gatedListen && ringBuffer.size() >= sampleRate && now - lastUngatedSearchAt >= 500) {
                    lastUngatedSearchAt = now
                    val result = ModemCore.demodulate(ringBuffer.snapshot())
                    reassembler.add(result.frames)
                    if (result.info != null) {
                        candidate = true
                        lastSignalAt = now
                        appendLog("[RX] Demodulator synchronized (gating disabled).")
                    }
                }

                if (candidate && now - lastSignalAt >= BURST_QUIET_MS) {
                    val result = ModemCore.demodulate(ringBuffer.snapshot())
                    reassembler.add(result.frames)
                    val validFrame = result.frames.firstOrNull { it.ok }
                    if (validFrame != null) {
                        val burstId = validFrame.burstId
                        val bitmap = reassembler.bitmap(burstId)
                        val missing = reassembler.missing(burstId)
                        val layer0 = reassembler.layers(burstId)[0]
                        if (layer0 != null) {
                            val decoded = ModemCore.unpackL0(layer0)
                            runOnUiThread {
                                currentRecord = decoded
                                val burstKind = if (validFrame.n == 1) "SHORT" else "LONG"
                                lastStatus = "DECODED: Coral #${decoded.coral} ($burstKind)"
                                logOutput += "\n----------------------------------------"
                                logOutput += "\n[RX CRC16: PASS] ${result.frames.count { it.ok }} frame(s) decoded."
                                logOutput += "\n  Burst Mode : $burstKind (${validFrame.n} frame${if (validFrame.n > 1) "s" else ""})"
                                logOutput += "\n  Coral ID   : #${decoded.coral}"
                                logOutput += "\n  Bleach     : %.2f (Index)".format(decoded.bleach)
                                logOutput += "\n  Health     : Score H%d".format(decoded.health)
                                logOutput += "\n  Texture    : Level T%d / 15".format(decoded.texture)
                                result.info?.let { logOutput += "\n  Signal SNR : %.1f dB".format(it.snrDb) }
                                logOutput += "\n  Missing    : ${if (missing.isEmpty()) "none" else missing}"
                                logOutput += "\n----------------------------------------"
                            }
                        } else {
                            appendLog("[RX] Burst ${validFrame.burstId}: ${result.frames.count { it.ok }} valid frame(s), missing $missing.")
                        }
                        val ackBits = ModemCore.packAck(burstId, bitmap, result.info?.snr4 ?: 0)
                        ringBuffer.clear()
                        candidate = false
                        txActive.set(true)
                        try {
                            val ackAudio = ModemCore.modulate(listOf(ackBits), ack = true)
                            if (audioRecord.recordingState == AudioRecord.RECORDSTATE_RECORDING) audioRecord.stop()
                            playAudio(ackAudio)
                            Thread.sleep(ACK_ECHO_SETTLE_MS)
                            ringBuffer.clear()
                            previousGateSamples = FloatArray(0)
                            if (rxRunning.get()) {
                                audioRecord.startRecording()
                                drainMutedCapture(audioRecord, chunk, ringBuffer, sampleRate / 4L)
                            }
                            appendLog("[TX] Replied to burst $burstId with a 28-bit ACK.")
                        } finally {
                            txActive.set(false)
                        }
                    } else {
                        appendLog("[RX] No CRC-valid frames found in synchronized burst; buffer cleared, continuing to listen.")
                        ringBuffer.clear()
                    }
                    candidate = false
                    previousGateSamples = FloatArray(0)
                }

                previousGateSamples = tailSamples(previousGateSamples, audioChunk)
            }
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
        } catch (e: Exception) {
            runOnUiThread {
                logOutput += "\n[RX Exception] ${e.message}"
                lastStatus = "RX Error: ${e.message}"
            }
        } finally {
            try {
                if (audioRecord.recordingState == AudioRecord.RECORDSTATE_RECORDING) audioRecord.stop()
            } catch (e: IllegalStateException) {
                appendLog("[RX] Error stopping microphone: ${e.message}")
            } finally {
                try {
                    audioRecord.release()
                } finally {
                    rxCalibrated.set(false)
                    rxRunning.set(false)
                    runOnUiThread { isListening = false }
                }
            }
        }
    }

    private fun drainMutedCapture(
        audioRecord: AudioRecord,
        chunk: ShortArray,
        ringBuffer: AudioRingBuffer,
        samplesToDrain: Long
    ) {
        var remaining = samplesToDrain
        while (remaining > 0 && rxRunning.get()) {
            val read = audioRecord.read(chunk, 0, minOf(chunk.size.toLong(), remaining).toInt())
            check(read > 0) { "Microphone capture failed while clearing the ACK echo" }
            ringBuffer.clear()
            remaining -= read
        }
    }

    private fun playAudio(audio: FloatArray) {
        val pcm = ShortArray(audio.size) { i ->
            (audio[i] * 32767).toInt().coerceIn(-32768, 32767).toShort()
        }
        val minBufferSize = AudioTrack.getMinBufferSize(
            ModemCore.SAMPLE_RATE,
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
                    .setSampleRate(ModemCore.SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(maxOf(minBufferSize, pcm.size * 2))
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()

        try {
            check(track.write(pcm, 0, pcm.size) == pcm.size) { "AudioTrack could not queue the complete waveform" }
            track.play()
            val deadline = SystemClock.elapsedRealtime() + audio.size * 1000L / ModemCore.SAMPLE_RATE + 2000L
            while (track.playbackHeadPosition < pcm.size && SystemClock.elapsedRealtime() < deadline) {
                Thread.sleep(10)
            }
            check(track.playbackHeadPosition >= pcm.size) { "AudioTrack playback did not complete" }
            track.stop()
        } finally {
            track.release()
        }
    }

    private fun appendLog(message: String) {
        runOnUiThread { logOutput += "\n$message" }
    }

    private fun updateStatus(message: String) {
        runOnUiThread { lastStatus = message }
    }

    private fun concatenate(first: FloatArray, second: FloatArray): FloatArray = first + second

    private fun tailSamples(previous: FloatArray, current: FloatArray): FloatArray {
        val combined = previous + current
        val length = ((1 shl ModemCore.DEFAULT_SF) * ModemCore.OVERSAMPLE).coerceAtMost(combined.size)
        return combined.copyOfRange(combined.size - length, combined.size)
    }

    private fun packLiveL0(record: CoralRecord): IntArray {
        val bits = ArrayList<Int>(80)
        fun write(value: Int, count: Int) {
            bits.addAll(ModemCore.intToBits(value.coerceIn(0, (1 shl count) - 1), count).toList())
        }
        fun quantize(value: Double, maximum: Double, count: Int): Int =
            (value.coerceIn(0.0, maximum) / maximum * ((1 shl count) - 1)).roundToInt()
        fun quantizeSignedColor(value: Double): Int =
            (value / 4.0).roundToInt().plus(16).coerceIn(0, 31)

        write(record.transect, 6)
        write(record.coral, 8)
        write(quantize(record.depthM, 63.5, 7), 7)
        write(quantize(record.timeS, 5115.0, 10), 10)
        write(quantize(record.bleach, 1.0, 6), 6)
        write(record.health - 1, 3)
        write(record.morph, 2)
        for (index in 0 until 2) {
            val color = record.colors.getOrNull(index) ?: Triple(0.0, 0.0, 0.0)
            write(quantize(color.first, 100.0, 6), 6)
            write(quantizeSignedColor(color.second), 5)
            write(quantizeSignedColor(color.third), 5)
        }
        write(record.patch, 1)
        write(record.evidence, 1)
        write(record.texture, 4)
        return bits.toIntArray()
    }

    private fun hexToBits(hex: String, bitCount: Int): IntArray {
        require(bitCount <= hex.length * 4)
        return IntArray(bitCount) { index ->
            (hex[index / 4].digitToInt(16) ushr (3 - index % 4)) and 1
        }
    }

    override fun onDestroy() {
        rxRunning.set(false)
        rxThread?.interrupt()
        super.onDestroy()
    }

    private companion object {
        const val RX_RING_SECONDS = 18
        const val ACK_WINDOW_MS = 3_000L
        const val ACK_ECHO_SETTLE_MS = 250L
        const val BURST_QUIET_MS = 300L
    }
}
// 6. JETPACK COMPOSE UI (SCROLLABLE LOG & JUSTIFIED LAYOUT)
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
        Text(
            text = "ReefGlyph Acoustic Modem",
            fontSize = 22.sp,
            fontWeight = FontWeight.Bold,
            color = Color(0xFF38BDF8)
        )
        Text(
            text = "Spec v1.1 | 1-4 kHz CSS SF7 | Pure Continuous PHY",
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

        Text(text = "STATUS: $status", fontSize = 12.sp, color = Color(0xFFFBBF24), fontWeight = FontWeight.Bold)

        Spacer(modifier = Modifier.height(6.dp))

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
