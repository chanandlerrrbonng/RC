package org.reefglyph

import android.Manifest
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
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
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.reefglyph.ui.theme.ReefGlyphTheme
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import org.opencv.core.CvException
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread
import kotlin.math.roundToInt

data class LibraryCoral(
    val id: String,
    val filename: String,
    val bleachIndex: Double,
    val flagged: Boolean,
    val healthScore: Int,
    val textureIndex: Int,
    val l0Hex: String,
    val l1Hex: String,
    val l2Hex: String,
    val preview: ImageBitmap
)

private data class TxBurstRequest(
    val layers: Map<Int, IntArray>,
    val description: String,
    val libraryId: String? = null,
    val longUpgrade: Boolean = false
)

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
    private var libraryCorals by mutableStateOf<List<LibraryCoral>>(emptyList())
    private var selectedLibraryCoral by mutableStateOf<LibraryCoral?>(null)
    private var selectedAnalysis by mutableStateOf<AnalysisResult?>(null)
    private var analyzerBusy by mutableStateOf(false)
    private var manualMode by mutableStateOf(false)
    private var receiverL0 by mutableStateOf<CoralRecord?>(null)
    private var receiverL1 by mutableStateOf<LayerRecord?>(null)
    private var receiverL2 by mutableStateOf<LayerRecord?>(null)
    private var receiverAlert by mutableStateOf<String?>(null)
    private var rxCsvPath by mutableStateOf("Unavailable")
    private var rxCsvCount by mutableIntStateOf(0)

    private val rxRunning = AtomicBoolean(false)
    private val rxCalibrated = AtomicBoolean(false)
    private val txActive = AtomicBoolean(false)
    private val ackWindow = AtomicBoolean(false)
    private val sendInProgress = AtomicBoolean(false)
    private val analyzerRunning = AtomicBoolean(false)
    private val activeBurstId = AtomicInteger(-1)
    private val receivedAck = AtomicReference<ModemCore.Ack?>(null)
    private val burstCounter = AtomicInteger(0)
    private val rxCsvLock = Any()
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
        try {
            libraryCorals = loadLibrary()
            selectedLibraryCoral = libraryCorals.firstOrNull()
        } catch (e: IOException) {
            lastStatus = "Library load error: ${e.message}"
            logOutput += "\n[Library Error] ${e.localizedMessage}"
        } catch (e: JSONException) {
            lastStatus = "Library load error: ${e.message}"
            logOutput += "\n[Library Error] ${e.localizedMessage}"
        } catch (e: IllegalArgumentException) {
            lastStatus = "Library load error: ${e.message}"
            logOutput += "\n[Library Error] ${e.localizedMessage}"
        }
        initializeRxCsvState()

        setContent {
            ReefGlyphTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    ReefGlyphMainScreen(
                        record = currentRecord,
                        isTransmitting = isTransmitting,
                        isListening = isListening,
                        status = lastStatus,
                        log = logOutput,
                        library = libraryCorals,
                        selectedLibraryCoral = selectedLibraryCoral,
                        selectedAnalysis = selectedAnalysis,
                        analyzerBusy = analyzerBusy,
                        manualMode = manualMode,
                        receiverL0 = receiverL0,
                        receiverL1 = receiverL1,
                        receiverL2 = receiverL2,
                        receiverAlert = receiverAlert,
                        rxCsvPath = rxCsvPath,
                        rxCsvCount = rxCsvCount,
                        onSendShort = { sendBurst(short = true) },
                        onSendLong = { sendBurst(short = false) },
                        onSendSelectedCoral = { sendSelectedLibraryCoral() },
                        onAnalyzeSelected = { analyzeSelectedLibraryCrop() },
                        onVerify20Crops = { verifyLibraryCrops() },
                        onRunDiveQueue = { budget -> runDiveQueue(budget) },
                        onSelectLibraryCoral = { entry ->
                            selectedLibraryCoral = entry
                            selectedAnalysis = null
                            manualMode = false
                        },
                        onToggleManualMode = { manualMode = !manualMode },
                        onDismissReceiverAlert = { receiverAlert = null },
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

    private fun loadLibrary(): List<LibraryCoral> {
        val items = assets.open("library/library.json").bufferedReader().use {
            JSONArray(it.readText())
        }
        return List(items.length()) { index ->
            val item = items.getJSONObject(index)
            val id = item.getString("id")
            val l0Hex = item.getString("l0_hex")
            val l1Hex = item.getString("l1_hex")
            val l2Hex = item.getString("l2_hex")
            hexToBits(l0Hex, 80)
            hexToBits(l1Hex, LayerCodec.L1_BITS)
            hexToBits(l2Hex, LayerCodec.L2_BITS)
            val preview = assets.open("library/$id.jpg").use { stream ->
                BitmapFactory.decodeStream(stream)
            } ?: throw IllegalArgumentException("Invalid preview image for $id")
            LibraryCoral(
                id = id,
                filename = item.getString("filename"),
                bleachIndex = item.getDouble("bleach_index"),
                flagged = item.getBoolean("flagged"),
                healthScore = item.getInt("health_score"),
                textureIndex = item.getInt("texture_idx"),
                l0Hex = l0Hex,
                l1Hex = l1Hex,
                l2Hex = l2Hex,
                preview = preview.asImageBitmap()
            )
        }
    }

    private fun libraryCropName(entry: LibraryCoral): String {
        val index = libraryCorals.indexOfFirst { it.id == entry.id }
        return if (index >= 0) {
            "Crop %02d".format(Locale.US, index + 1)
        } else {
            "Crop ${entry.id.substringAfterLast('-')}"
        }
    }

    private var rxCsvRows = 0

    private fun initializeRxCsvState() {
        val directory = getExternalFilesDir(null)
        if (directory == null) {
            rxCsvPath = "External files directory unavailable"
            return
        }
        val file = File(directory, "reefglyph_rx_log.csv")
        rxCsvPath = file.absolutePath
        if (file.isFile) {
            try {
                rxCsvRows = file.bufferedReader().useLines { lines ->
                    (lines.count() - 1).coerceAtLeast(0)
                }
                rxCsvCount = rxCsvRows
            } catch (e: IOException) {
                lastStatus = "RX CSV read error: ${e.message}"
                logOutput += "\n[RX CSV Error] ${e.localizedMessage}"
            }
        }
    }

    private fun appendRxCsvRow(
        burstId: Int,
        mode: String,
        info: ModemCore.SyncInfo?,
        crcPassFrames: Int,
        totalFrames: Int,
        record: CoralRecord?
    ) {
        try {
            val directory = getExternalFilesDir(null)
                ?: throw IOException("External files directory is unavailable")
            val file = File(directory, "reefglyph_rx_log.csv")
            synchronized(rxCsvLock) {
                FileOutputStream(file, true).bufferedWriter(Charsets.UTF_8).use { writer ->
                    if (!file.exists() || file.length() == 0L) {
                        writer.appendLine(
                            "timestamp_ms,burst_id,mode,snr_db,snr4,cfo_hz,crc_pass_frames,total_frames,coral_id,bleach_idx,texture_idx"
                        )
                    }
                    val cfoHz = info?.let {
                        it.cfo * ModemCore.BANDWIDTH / (1 shl ModemCore.DEFAULT_SF)
                    }
                    val values = listOf(
                        System.currentTimeMillis().toString(),
                        burstId.toString(),
                        mode,
                        info?.snrDb?.let { "%.2f".format(Locale.US, it) }.orEmpty(),
                        info?.snr4?.toString().orEmpty(),
                        cfoHz?.let { "%.2f".format(Locale.US, it) }.orEmpty(),
                        crcPassFrames.toString(),
                        totalFrames.toString(),
                        record?.coral?.toString().orEmpty(),
                        record?.bleach?.let { "%.6f".format(Locale.US, it) }.orEmpty(),
                        record?.texture?.toString().orEmpty()
                    )
                    writer.appendLine(values.joinToString(","))
                }
                rxCsvRows++
                val count = rxCsvRows
                runOnUiThread {
                    rxCsvPath = file.absolutePath
                    rxCsvCount = count
                }
            }
        } catch (e: IOException) {
            appendLog("[RX CSV Error] ${e.localizedMessage}")
            updateStatus("RX CSV write error: ${e.message}")
        } catch (e: SecurityException) {
            appendLog("[RX CSV Error] ${e.localizedMessage}")
            updateStatus("RX CSV write error: ${e.message}")
        }
    }

    private fun sendBurst(short: Boolean) {
        val record = currentRecord.copy()
        val l0 = packLiveL0(record)
        val layers = linkedMapOf(0 to l0)
        if (!short) {
            val selectedLayers = selectedLibraryCoral?.let { entry ->
                mapOf(
                    1 to hexToBits(entry.l1Hex, LayerCodec.L1_BITS),
                    2 to hexToBits(entry.l2Hex, LayerCodec.L2_BITS)
                )
            } ?: sampleLayers
            layers.putAll(selectedLayers)
        }
        startBurstSequence(
            listOf(
                TxBurstRequest(
                    layers,
                    selectedLibraryCoral?.let { "${libraryCropName(it)} manual coral #${record.coral}" }
                        ?: "manual coral #${record.coral}"
                )
            ),
            "Manual transmission"
        )
    }

    private fun sendSelectedLibraryCoral() {
        val entry = selectedLibraryCoral
        if (entry == null) {
            updateStatus("Select a library coral before sending.")
            return
        }
        val analysis = selectedAnalysis
        val analyzedRecord = analysis?.let {
            currentRecord.copy(
                bleach = it.bleachIndex,
                health = it.healthScore,
                morph = 3,
                colors = it.dominantColors.map { color -> Triple(color.l, color.a, color.b) },
                evidence = if (it.flagged) 1 else 0,
                texture = it.textureBin
            )
        }
        val short = !(analysis?.flagged ?: entry.flagged)
        val l0 = analyzedRecord?.let { packLiveL0(it) } ?: hexToBits(entry.l0Hex, 80)
        val layers = linkedMapOf(0 to l0)
        if (!short) {
            layers[1] = hexToBits(entry.l1Hex, LayerCodec.L1_BITS)
            layers[2] = hexToBits(entry.l2Hex, LayerCodec.L2_BITS)
        }
        if (analysis != null) {
            appendLog("[TX] ${libraryCropName(entry)}: on-device L0 triage; using this crop's library L1/L2 when flagged.")
        }
        startBurstSequence(
            listOf(TxBurstRequest(layers, "${libraryCropName(entry)} (${if (short) "SHORT" else "LONG"})")),
            "Library transmission"
        )
    }

    private fun analyzeSelectedLibraryCrop() {
        val entry = selectedLibraryCoral
        if (entry == null) {
            updateStatus("Select a library coral before running on-device analysis.")
            return
        }
        if (!analyzerRunning.compareAndSet(false, true)) return
        analyzerBusy = true
        thread(name = "CoralL0Analysis") {
            try {
                CoralAnalyzer.initialize(applicationContext)
                val bitmap = loadLibraryBitmap(entry)
                val result = try {
                    CoralAnalyzer.analyzeL0(bitmap, libraryAnalysisRect(bitmap))
                } finally {
                    bitmap.recycle()
                }
                runOnUiThread {
                    if (selectedLibraryCoral?.id == entry.id) selectedAnalysis = result
                    val cropName = libraryCropName(entry)
                    lastStatus = "$cropName: on-device L0 ${if (result.flagged) "flagged" else "not flagged"}"
                    logOutput += "\n[L0 On-Device] $cropName: bleach %.3f, H%d, T%d, %s, %.2f ms."
                        .format(
                            Locale.US,
                            result.bleachIndex,
                            result.healthScore,
                            result.textureBin,
                            if (result.flagged) "FLAGGED" else "clear",
                            result.elapsedMs
                        )
                    logOutput += "\n[L0 On-Device] Dominant Lab: " +
                        result.dominantColors.joinToString(" / ") {
                            "(%.1f, %.1f, %.1f)".format(Locale.US, it.l, it.a, it.b)
                        } + "; this crop's precomputed L1/L2 provide contour and thumbnail detail on LONG."
                }
            } catch (e: IOException) {
                reportAnalyzerFailure("On-device analysis", e)
            } catch (e: JSONException) {
                reportAnalyzerFailure("On-device analysis", e)
            } catch (e: IllegalArgumentException) {
                reportAnalyzerFailure("On-device analysis", e)
            } catch (e: IllegalStateException) {
                reportAnalyzerFailure("On-device analysis", e)
            } catch (e: CvException) {
                reportAnalyzerFailure("On-device analysis", e)
            } finally {
                analyzerRunning.set(false)
                runOnUiThread { analyzerBusy = false }
            }
        }
    }

    private fun verifyLibraryCrops() {
        if (libraryCorals.size < 20) {
            updateStatus("Verification needs at least 20 library crops.")
            return
        }
        runLibraryAnalyzerBenchmark(libraryCorals.take(20), passes = 1)
    }

    private fun runLibraryAnalyzerBenchmark(entries: List<LibraryCoral>, passes: Int = 1) {
        require(entries.isNotEmpty()) { "Benchmark requires library crops" }
        require(passes > 0) { "Benchmark pass count must be positive" }
        if (!analyzerRunning.compareAndSet(false, true)) return
        analyzerBusy = true
        thread(name = "CoralL0Benchmark") {
            try {
                CoralAnalyzer.initialize(applicationContext)
                val latencies = ArrayList<Double>(entries.size * passes)
                var agreements = 0
                repeat(passes) {
                    entries.forEach { entry ->
                        val bitmap = loadLibraryBitmap(entry)
                        val result = try {
                            CoralAnalyzer.analyzeL0(bitmap, libraryAnalysisRect(bitmap))
                        } finally {
                            bitmap.recycle()
                        }
                        latencies += result.elapsedMs
                        if (result.flagged == entry.flagged) agreements++
                    }
                }
                val meanMs = latencies.average()
                val maxMs = latencies.maxOrNull() ?: 0.0
                val agreementRate = agreements * 100.0 / latencies.size
                runOnUiThread {
                    lastStatus = "L0 verification: %.1f%% flag agreement".format(Locale.US, agreementRate)
                    logOutput += "\n[L0 Verify] ${entries.size} crops x $passes pass(es): " +
                        "%.1f%% flag agreement vs library.json; mean %.2f ms, max %.2f ms (%d analyses)."
                            .format(Locale.US, agreementRate, meanMs, maxMs, latencies.size)
                }
            } catch (e: IOException) {
                reportAnalyzerFailure("L0 verification", e)
            } catch (e: JSONException) {
                reportAnalyzerFailure("L0 verification", e)
            } catch (e: IllegalArgumentException) {
                reportAnalyzerFailure("L0 verification", e)
            } catch (e: IllegalStateException) {
                reportAnalyzerFailure("L0 verification", e)
            } catch (e: CvException) {
                reportAnalyzerFailure("L0 verification", e)
            } finally {
                analyzerRunning.set(false)
                runOnUiThread { analyzerBusy = false }
            }
        }
    }

    private fun loadLibraryBitmap(entry: LibraryCoral): Bitmap =
        assets.open("library/${entry.id}.jpg").use { stream ->
            BitmapFactory.decodeStream(stream)
                ?: throw IllegalArgumentException("Invalid library crop for ${entry.id}")
        }

    private fun libraryAnalysisRect(bitmap: Bitmap): Rect {
        val left = bitmap.width / 10
        val top = bitmap.height / 10
        return Rect(left, top, left + bitmap.width * 8 / 10, top + bitmap.height * 8 / 10)
    }

    private fun reportAnalyzerFailure(operation: String, error: Exception) {
        runOnUiThread {
            lastStatus = "$operation failed: ${error.message}"
            logOutput += "\n[$operation Error] ${error.localizedMessage ?: error.message}"
        }
    }

    private fun runDiveQueue(budgetSec: Double) {
        if (!budgetSec.isFinite() || budgetSec <= 0.0) {
            updateStatus("Enter a positive airtime budget in seconds.")
            return
        }
        if (libraryCorals.isEmpty()) {
            updateStatus("The coral library is unavailable.")
            return
        }

        val queueIdToCoral = libraryCorals.withIndex().associate { it.index to it.value }
        val items = libraryCorals.mapIndexed { index, entry ->
            AirtimeItem(index, entry.bleachIndex, entry.flagged)
        }
        val queue = try {
            LayerCodec.buildAirtimeQueue(items, budgetSec)
        } catch (e: IllegalArgumentException) {
            updateStatus("Could not build dive queue: ${e.message}")
            return
        }
        if (queue.isEmpty()) {
            updateStatus("Budget is too small for a SHORT burst (${LayerCodec.SHORT_AIRTIME_SEC}s minimum).")
            return
        }

        val upgradePreambleOverhead =
            (ModemCore.frameSymbols(first = true) - ModemCore.frameSymbols(first = false)) *
                (1 shl ModemCore.DEFAULT_SF).toDouble() / ModemCore.BANDWIDTH
        val selectedQueue = mutableListOf<AirtimeQueueEntry>()
        var plannedAirtime = 0.0
        for (scheduled in queue) {
            val cost = scheduled.airtimeSec +
                if (scheduled.burst == AirtimeBurst.LONG_UPGRADE) upgradePreambleOverhead else 0.0
            if (plannedAirtime + cost > budgetSec + 1e-12) break
            plannedAirtime += cost
            selectedQueue += scheduled
        }
        val requests = selectedQueue.map { scheduled ->
            val entry = queueIdToCoral.getValue(scheduled.coralId)
            val short = scheduled.burst == AirtimeBurst.SHORT
            val layers = linkedMapOf(0 to hexToBits(entry.l0Hex, 80))
            if (!short) {
                layers[1] = hexToBits(entry.l1Hex, LayerCodec.L1_BITS)
                layers[2] = hexToBits(entry.l2Hex, LayerCodec.L2_BITS)
            }
            TxBurstRequest(
                layers = layers,
                description = "${libraryCropName(entry)} ${if (short) "SHORT" else "LONG"}",
                libraryId = entry.id,
                longUpgrade = !short
            )
        }
        startBurstSequence(
            requests,
            "Dive queue: ${requests.size} bursts, %.1f/%.1f s planned".format(
                Locale.US,
                plannedAirtime,
                budgetSec
            )
        )
    }

    private fun startBurstSequence(requests: List<TxBurstRequest>, description: String) {
        if (requests.isEmpty()) return
        if (!sendInProgress.compareAndSet(false, true)) return
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissionsIfNeeded()
            appendLog("[TX] Audio permission is required for ACK reception.")
            sendInProgress.set(false)
            return
        }

        isTransmitting = true
        lastStatus = "Preparing $description..."

        thread(name = "AcousticTxThread") {
            try {
                var acknowledgedBursts = 0
                val queueBurstIds = mutableMapOf<String, Int>()
                val acknowledgedShorts = mutableSetOf<String>()
                requests.forEachIndexed { index, request ->
                    val burstId = if (request.longUpgrade) {
                        request.libraryId?.let { queueBurstIds[it] }
                            ?: burstCounter.getAndUpdate { (it + 1) and 0x3F }
                    } else {
                        burstCounter.getAndUpdate { (it + 1) and 0x3F }
                    }
                    if (!request.longUpgrade) {
                        request.libraryId?.let { queueBurstIds[it] = burstId }
                    }
                    val originalFrames = ModemCore.packetize(request.layers, burstId)
                    val canUpgradeShort = request.longUpgrade &&
                        request.libraryId?.let { it in acknowledgedShorts } == true
                    var framesToSend = if (canUpgradeShort) originalFrames.drop(1) else originalFrames
                    if (canUpgradeShort) {
                        appendLog("[TX] ${request.description}: reusing acknowledged F0; sending F1-F8.")
                    }
                    var complete = false
                    for (attempt in 1..ModemCore.MAX_ATTEMPTS) {
                        activeBurstId.set(burstId)
                        receivedAck.set(null)
                        ackWindow.set(false)
                        txActive.set(true)
                        updateStatus("${request.description}: attempt $attempt/${ModemCore.MAX_ATTEMPTS}...")
                        appendLog("[TX] Burst $burstId (${index + 1}/${requests.size}, ${request.description}): attempt $attempt, ${framesToSend.size} frame(s).")
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
                            acknowledgedBursts++
                            if (!request.longUpgrade) {
                                request.libraryId?.let { acknowledgedShorts += it }
                            }
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
                    if (!complete) appendLog("[TX] ${request.description} finished without full ACK.")
                }
                updateStatus(
                    if (requests.size == 1) {
                        if (acknowledgedBursts == 1) "Transmission acknowledged."
                        else "Transmission finished; ACK incomplete or timed out."
                    } else {
                        "Dive queue finished: $acknowledgedBursts/${requests.size} bursts acknowledged."
                    }
                )
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
            val publishedLayerByBurst = mutableMapOf<Int, Int>()
            val alertedBursts = mutableSetOf<Int>()
            var noiseFloor: ModemCore.RunningNoiseFloor? = null
            var calibrationPower = 0.0
            var calibrationChunks = 0
            var calibrationSamples = 0
            var previousGateSamples = FloatArray(0)
            var candidate = false
            var lastSignalAt = 0L
            var lastUngatedSearchAt = 0L
            var lastProgressiveDecodeAt = 0L
            var candidateBurstId: Int? = null
            var candidateFrameCount = 0
            var candidateSyncInfo: ModemCore.SyncInfo? = null

            while (rxRunning.get()) {
                val read = audioRecord.read(chunk, 0, chunk.size)
                if (read <= 0) continue
                if (txActive.get()) {
                    ringBuffer.clear()
                    candidate = false
                    candidateBurstId = null
                    candidateFrameCount = 0
                    candidateSyncInfo = null
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
                    lastProgressiveDecodeAt = 0L
                    candidateBurstId = null
                    candidateFrameCount = 0
                    candidateSyncInfo = null
                    appendLog("[RX] Stage 1 energy and Stage 2 up-chirp gates passed.")
                } else if (!candidate && !gatedListen && ringBuffer.size() >= sampleRate && now - lastUngatedSearchAt >= 500) {
                    lastUngatedSearchAt = now
                    val result = ModemCore.demodulate(ringBuffer.snapshot())
                    reassembler.add(result.frames)
                    if (result.info != null) {
                        candidate = true
                        lastSignalAt = now
                        lastProgressiveDecodeAt = now
                        candidateSyncInfo = result.info
                        result.frames.firstOrNull { it.ok }?.let { frame ->
                            candidateBurstId = frame.burstId
                            candidateFrameCount = frame.n
                        }
                        publishReceiverProgress(
                            result.frames,
                            reassembler,
                            publishedLayerByBurst,
                            alertedBursts,
                            result.info
                        )
                        appendLog("[RX] Demodulator synchronized (gating disabled).")
                    }
                }

                val layer2Complete = candidateBurstId?.let {
                    2 in reassembler.layers(it)
                } == true
                if (candidate && !layer2Complete && now - lastProgressiveDecodeAt >= 750) {
                    lastProgressiveDecodeAt = now
                    val partial = ModemCore.demodulate(ringBuffer.snapshot())
                    reassembler.add(partial.frames)
                    partial.info?.let { candidateSyncInfo = it }
                    partial.frames.firstOrNull { it.ok }?.let { frame ->
                        candidateBurstId = frame.burstId
                        candidateFrameCount = frame.n
                    }
                    publishReceiverProgress(
                        partial.frames,
                        reassembler,
                        publishedLayerByBurst,
                        alertedBursts,
                        partial.info
                    )
                }

                if (candidate && now - lastSignalAt >= BURST_QUIET_MS) {
                    val result = ModemCore.demodulate(ringBuffer.snapshot())
                    reassembler.add(result.frames)
                    val validFrame = result.frames.firstOrNull { it.ok }
                    result.info?.let { candidateSyncInfo = it }
                    validFrame?.let {
                        candidateBurstId = it.burstId
                        candidateFrameCount = it.n
                    }
                    publishReceiverProgress(
                        result.frames,
                        reassembler,
                        publishedLayerByBurst,
                        alertedBursts,
                        result.info
                    )
                    val burstId = validFrame?.burstId ?: candidateBurstId
                    if (burstId != null) {
                        val bitmap = reassembler.bitmap(burstId)
                        val missing = reassembler.missing(burstId)
                        val layer0 = reassembler.layers(burstId)[0]
                        val decoded = layer0?.let { ModemCore.unpackL0(it) }
                        val totalFrames = validFrame?.n ?: candidateFrameCount
                        val burstKind = if (totalFrames == 1) "SHORT" else "LONG"
                        val logMode = when {
                            missing.isNotEmpty() -> "${burstKind}_TIMEOUT"
                            else -> burstKind
                        }
                        appendRxCsvRow(
                            burstId = burstId,
                            mode = logMode,
                            info = result.info ?: candidateSyncInfo,
                            crcPassFrames = bitmap.count { it },
                            totalFrames = totalFrames,
                            record = decoded
                        )
                        if (layer0 != null) {
                            val decodedRecord = decoded ?: ModemCore.unpackL0(layer0)
                            runOnUiThread {
                                currentRecord = decodedRecord
                                lastStatus = "DECODED: Coral #${decodedRecord.coral} ($burstKind)"
                                logOutput += "\n----------------------------------------"
                                logOutput += "\n[RX CRC16: PASS] ${bitmap.count { it }} of $totalFrames frame(s) decoded."
                                logOutput += "\n  Burst Mode : $burstKind ($totalFrames frame${if (totalFrames != 1) "s" else ""})"
                                logOutput += "\n  Coral ID   : #${decodedRecord.coral}"
                                logOutput += "\n  Bleach     : %.2f (Index)".format(decodedRecord.bleach)
                                logOutput += "\n  Health     : Score H%d".format(decodedRecord.health)
                                logOutput += "\n  Texture    : Level T%d / 15".format(decodedRecord.texture)
                                result.info?.let { logOutput += "\n  Signal SNR : %.1f dB".format(it.snrDb) }
                                logOutput += "\n  Missing    : ${if (missing.isEmpty()) "none" else missing}"
                                logOutput += "\n----------------------------------------"
                            }
                        } else {
                            appendLog("[RX] Burst $burstId: ${bitmap.count { it }} valid frame(s), missing $missing.")
                        }
                        val ackBits = ModemCore.packAck(
                            burstId,
                            bitmap,
                            (result.info ?: candidateSyncInfo)?.snr4 ?: 0
                        )
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
                        appendRxCsvRow(
                            burstId = -1,
                            mode = "TIMEOUT",
                            info = result.info ?: candidateSyncInfo,
                            crcPassFrames = 0,
                            totalFrames = 0,
                            record = null
                        )
                        ringBuffer.clear()
                    }
                    candidate = false
                    candidateBurstId = null
                    candidateFrameCount = 0
                    candidateSyncInfo = null
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

    private fun publishReceiverProgress(
        frames: List<ModemCore.Frame>,
        reassembler: ModemCore.Reassembler,
        publishedLayerByBurst: MutableMap<Int, Int>,
        alertedBursts: MutableSet<Int>,
        info: ModemCore.SyncInfo?
    ) {
        frames.filter { it.ok }.map { it.burstId }.distinct().forEach { burstId ->
            val layers = reassembler.layers(burstId)
            val level = when {
                2 in layers -> 2
                1 in layers -> 1
                0 in layers -> 0
                else -> -1
            }
            if (level <= (publishedLayerByBurst[burstId] ?: -1)) return@forEach
            val layer0 = layers[0] ?: return@forEach
            val decodedL0 = ModemCore.unpackL0(layer0)
            val decodedL1 = layers[1]?.let { LayerCodec.unpackL1(it) }
            val decodedL2 = layers[2]?.let { LayerCodec.unpackL2(it) }
            val showAlert = alertedBursts.add(burstId)
            publishedLayerByBurst[burstId] = level
            runOnUiThread {
                receiverL0 = decodedL0
                receiverL1 = decodedL1
                receiverL2 = decodedL2
                currentRecord = decodedL0
                lastStatus = when (level) {
                    0 -> "Coral #${decodedL0.coral}: L0 decoded"
                    1 -> "Coral #${decodedL0.coral}: L1 contour decoded"
                    else -> "Coral #${decodedL0.coral}: L2 thumbnail decoded"
                }
                if (showAlert) {
                    val decision = if (decodedL0.evidence != 0) "flagged" else "healthy"
                    receiverAlert = "Coral #${decodedL0.coral} is $decision (bleach %.2f).".format(
                        Locale.US,
                        decodedL0.bleach
                    )
                }
                info?.let {
                    logOutput += "\n[RX] Burst $burstId: L$level complete; SNR %.1f dB.".format(it.snrDb)
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
        require(bitCount > 0 && hex.length >= (bitCount + 3) / 4) {
            "Expected at least ${(bitCount + 3) / 4} hex characters for $bitCount bits"
        }
        require(hex.all { it.digitToIntOrNull(16) != null }) { "Layer data must be hexadecimal" }
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
    library: List<LibraryCoral>,
    selectedLibraryCoral: LibraryCoral?,
    selectedAnalysis: AnalysisResult?,
    analyzerBusy: Boolean,
    manualMode: Boolean,
    receiverL0: CoralRecord?,
    receiverL1: LayerRecord?,
    receiverL2: LayerRecord?,
    receiverAlert: String?,
    rxCsvPath: String,
    rxCsvCount: Int,
    onSendShort: () -> Unit,
    onSendLong: () -> Unit,
    onSendSelectedCoral: () -> Unit,
    onAnalyzeSelected: () -> Unit,
    onVerify20Crops: () -> Unit,
    onRunDiveQueue: (Double) -> Unit,
    onSelectLibraryCoral: (LibraryCoral) -> Unit,
    onToggleManualMode: () -> Unit,
    onDismissReceiverAlert: () -> Unit,
    onToggleListening: () -> Unit,
    onRecordChange: (CoralRecord) -> Unit,
    modifier: Modifier = Modifier
) {
    val screenScrollState = rememberScrollState()
    val terminalScrollState = rememberScrollState()
    var budgetText by remember { mutableStateOf("120") }

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

        Card(
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text(
                    text = "CORAL LIBRARY & TRIAGE",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF94A3B8)
                )
                Spacer(modifier = Modifier.height(8.dp))
                LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    itemsIndexed(library, key = { _, coral -> coral.id }) { index, coral ->
                        val selected = coral.id == selectedLibraryCoral?.id
                        Column(
                            modifier = Modifier
                                .width(112.dp)
                                .clip(RoundedCornerShape(8.dp))
                                .background(if (selected) Color(0xFF334155) else Color(0xFF0F172A))
                                .clickable { onSelectLibraryCoral(coral) }
                                .padding(6.dp)
                        ) {
                            Image(
                                bitmap = coral.preview,
                                contentDescription = "Preview of crop ${index + 1}",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(70.dp)
                                    .clip(RoundedCornerShape(6.dp))
                            )
                            Text(
                                "Crop %02d".format(Locale.US, index + 1),
                                color = Color.White,
                                fontSize = 11.sp,
                                maxLines = 1
                            )
                            Text(
                                if (coral.flagged) "FLAGGED" else "HEALTHY",
                                color = if (coral.flagged) Color(0xFFF87171) else Color(0xFF4ADE80),
                                fontSize = 10.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }
                selectedLibraryCoral?.let { coral ->
                    val cropIndex = library.indexOfFirst { it.id == coral.id } + 1
                    val triageFlagged = selectedAnalysis?.flagged ?: coral.flagged
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Image(
                            bitmap = coral.preview,
                            contentDescription = "Selected crop $cropIndex",
                            contentScale = ContentScale.Crop,
                            modifier = Modifier
                                .size(width = 116.dp, height = 88.dp)
                                .clip(RoundedCornerShape(8.dp))
                        )
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                "Crop %02d".format(Locale.US, cropIndex.coerceAtLeast(1)),
                                color = Color.White,
                                fontWeight = FontWeight.Bold
                            )
                            Text(coral.filename, color = Color(0xFF94A3B8), fontSize = 10.sp, maxLines = 1)
                            Text(
                                "${if (triageFlagged) "Flagged · LONG" else "Healthy · SHORT"}  " +
                                    "bleach %.2f · H%d · T%d".format(
                                        Locale.US,
                                        selectedAnalysis?.bleachIndex ?: coral.bleachIndex,
                                        selectedAnalysis?.healthScore ?: coral.healthScore,
                                        selectedAnalysis?.textureBin ?: coral.textureIndex
                                    ),
                                color = if (triageFlagged) Color(0xFFF87171) else Color(0xFF4ADE80),
                                fontSize = 11.sp
                            )
                            Button(
                                onClick = onSendSelectedCoral,
                                enabled = !isTransmitting && !analyzerBusy,
                                modifier = Modifier.padding(top = 4.dp)
                            ) {
                                Text(
                                    if (triageFlagged) "SEND LONG (14.08s)" else "SEND SHORT (2.13s)",
                                    fontSize = 11.sp
                                )
                            }
                            Row(
                                modifier = Modifier.padding(top = 4.dp),
                                horizontalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                OutlinedButton(
                                    onClick = onAnalyzeSelected,
                                    enabled = !analyzerBusy,
                                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp)
                                ) {
                                    Text(
                                        if (analyzerBusy) "Analyzing..." else "Analyze On-Device",
                                        fontSize = 10.sp
                                    )
                                }
                                OutlinedButton(
                                    onClick = onVerify20Crops,
                                    enabled = !analyzerBusy,
                                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp)
                                ) {
                                    Text("Verify 20 Crops", fontSize = 10.sp)
                                }
                            }
                        }
                    }
                }
                TextButton(onClick = onToggleManualMode) {
                    Text(
                        if (manualMode) "Back to coral library controls" else "Use manual sliders",
                        color = Color(0xFF38BDF8)
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        Card(
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text(
                    text = "RECEIVED GLYPH",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF94A3B8)
                )
                Spacer(modifier = Modifier.height(8.dp))
                CoralGlyphCanvas(
                    l0 = receiverL0,
                    l1 = receiverL1,
                    l2 = receiverL2,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(170.dp)
                        .clip(RoundedCornerShape(8.dp))
                )
                Text(
                    text = when {
                        receiverL2 != null -> "Full glyph: L0 summary + L1 contour + L2 colour thumbnail"
                        receiverL1 != null -> "L0 summary + L1 contour received; waiting for L2 thumbnail"
                        receiverL0 != null -> "L0 summary only: no contour or image thumbnail"
                        else -> "Waiting for a CRC-valid F0 (L0) frame"
                    },
                    color = Color(0xFF94A3B8),
                    fontSize = 11.sp,
                    modifier = Modifier.padding(top = 5.dp)
                )
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        if (manualMode) {
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
        }

        Spacer(modifier = Modifier.height(12.dp))

        Card(
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text(
                    text = "AIRTIME-BUDGET SCHEDULER",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF94A3B8)
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    OutlinedTextField(
                        value = budgetText,
                        onValueChange = { value ->
                            if (value.matches(Regex("""\d*\.?\d*"""))) budgetText = value
                        },
                        label = { Text("Budget (seconds)") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
                        singleLine = true,
                        modifier = Modifier.weight(1f)
                    )
                    Button(
                        onClick = { budgetText.toDoubleOrNull()?.let(onRunDiveQueue) },
                        enabled = !isTransmitting &&
                            library.isNotEmpty() &&
                            (budgetText.toDoubleOrNull()?.let { it.isFinite() && it > 0.0 } == true)
                    ) {
                        Text("Run Dive Queue")
                    }
                }
                Text(
                    text = "Schedules SHORT bursts within budget, then prioritizes flagged LONG upgrades.",
                    color = Color(0xFF94A3B8),
                    fontSize = 10.sp
                )
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

        Text(
            text = "RX CSV: $rxCsvPath ($rxCsvCount rows)",
            color = Color(0xFF94A3B8),
            fontSize = 10.sp
        )

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

    receiverAlert?.let { message ->
        AlertDialog(
            onDismissRequest = onDismissReceiverAlert,
            title = { Text("Coral received") },
            text = { Text(message) },
            confirmButton = {
                TextButton(onClick = onDismissReceiverAlert) {
                    Text("OK")
                }
            }
        )
    }
}
