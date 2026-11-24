package org.reefglyph

import java.nio.charset.StandardCharsets
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ModemCoreTest {
    @Test
    fun goldenWavsMatchPythonFramesAndSnr() {
        val manifest = JSONObject(String(resourceBytes("golden/manifest.json"), StandardCharsets.UTF_8))
        val cases = manifest.getJSONObject("cases")
        val names = cases.keys()
        while (names.hasNext()) {
            val name = names.next()
            val case = cases.getJSONObject(name)
            val expected = case.getJSONArray("expected_decoded_frames_hex")
            val audio = readPcm16Wav(case.getString("filename"))
            val expectedSnr = case.getInt("expected_snr4")
            val decodedHex: List<String>
            val snr4: Int

            if (case.getString("kind") == "ack") {
                val (bits, info) = ModemCore.demodulateAck(audio)
                assertNotNull("$name: preamble", info)
                assertNotNull("$name: ACK bits", bits)
                assertTrue("$name: ACK CRC", ModemCore.unpackAck(bits!!).ok)
                decodedHex = listOf(bitsToHex(bits))
                snr4 = info!!.snr4
            } else {
                val result = ModemCore.demodulate(audio, maxFrames = expected.length())
                assertNotNull("$name: preamble", result.info)
                assertEquals("$name: decoded frame count", expected.length(), result.frames.size)
                assertTrue("$name: frame CRC", result.frames.all { it.ok })
                decodedHex = result.frames.map { bitsToHex(ModemCore.packFrame(it)) }
                snr4 = result.info!!.snr4
            }

            assertEquals("$name: decoded bytes", expected.toStringList(), decodedHex)
            assertEquals("$name: quantized SNR", expectedSnr, snr4)
        }
    }

    @Test
    fun ackAndSelectiveResendReassembleMissingFrames() {
        val manifest = JSONObject(String(resourceBytes("golden/manifest.json"), StandardCharsets.UTF_8))
        val encodedLayers = manifest.getJSONObject("sample_layers")
        val layers = mapOf(
            0 to hexToBits(encodedLayers.getString("L0"), 80),
            1 to hexToBits(encodedLayers.getString("L1"), 175),
            2 to hexToBits(encodedLayers.getString("L2"), 576)
        )
        val burstId = 23
        val original = ModemCore.packetize(layers, burstId)
        assertEquals(9, original.size)
        val expectedFrames = manifest.getJSONObject("cases")
            .getJSONObject("long_clean")
            .getJSONArray("expected_decoded_frames_hex")
            .toStringList()
        assertEquals(expectedFrames, original.map { bitsToHex(it) })

        val reassembler = ModemCore.Reassembler()
        val firstAttempt = original.mapIndexedNotNull { index, bits ->
            if (index in 3..5) null else ModemCore.unpackFrame(bits)
        }
        reassembler.add(firstAttempt)

        val ackBits = ModemCore.packAck(burstId, reassembler.bitmap(), 9)
        assertEquals(ModemCore.ACK_BITS, ackBits.size)
        val ack = ModemCore.unpackAck(ackBits)
        assertTrue(ack.ok)
        assertEquals(burstId, ack.burstId)
        assertEquals(listOf(3, 4, 5), reassembler.missing())

        val resend = ModemCore.buildSelectiveResendBurst(original, ack.bitmap, attempt = 1)
        assertEquals(listOf(3, 4, 5), resend.map { ModemCore.unpackFrame(it).idx })
        assertTrue(resend.all { ModemCore.unpackFrame(it).burstId == burstId })
        reassembler.add(resend.map { ModemCore.unpackFrame(it) })

        assertTrue(reassembler.bitmap().all { it })
        assertTrue(reassembler.missing().isEmpty())
        val assembledLayers = reassembler.layers()
        assertEquals(setOf(0, 1, 2), assembledLayers.keys)
        for ((layer, bits) in layers) assertArrayEquals(bits, assembledLayers.getValue(layer))
    }

    @Test
    fun audioUpChirpGateAcceptsModemAudioAndRejectsSilence() {
        val frame = ModemCore.packFrame(
            ModemCore.Frame(0, 0, 1, 1, IntArray(ModemCore.PAYLOAD_BITS))
        )
        val audio = ModemCore.modulate(listOf(frame))

        assertTrue(ModemCore.stage2AudioUpChirpGate(audio.copyOfRange(0, ModemCore.SAMPLE_RATE / 10)))
        assertTrue(!ModemCore.stage2AudioUpChirpGate(FloatArray(ModemCore.SAMPLE_RATE / 10)))
    }

    @Test
    fun unpackL0ReadsManifestSamplePayload() {
        val manifest = JSONObject(String(resourceBytes("golden/manifest.json"), StandardCharsets.UTF_8))
        val sampleL0 = hexToBits(manifest.getJSONObject("sample_layers").getString("L0"), 80)

        val decoded = ModemCore.unpackL0(sampleL0)

        assertEquals(17, decoded.transect)
        assertEquals(42, decoded.coral)
    }

    @Test
    fun layerCodecRoundTripsManifestSamplesBitExactly() {
        val manifest = JSONObject(String(resourceBytes("golden/manifest.json"), StandardCharsets.UTF_8))
            .getJSONObject("sample_layers")
        val sampleL1 = hexToBits(manifest.getString("L1"), LayerCodec.L1_BITS)
        val sampleL2 = hexToBits(manifest.getString("L2"), LayerCodec.L2_BITS)

        assertArrayEquals(sampleL1, LayerCodec.packL1(LayerCodec.unpackL1(sampleL1)))
        assertArrayEquals(sampleL2, LayerCodec.packL2(LayerCodec.unpackL2(sampleL2)))
    }

    @Test
    fun airtimeQueueUsesBudgetAndUpgradesHighestBleachFirst() {
        val items = listOf(
            AirtimeItem(coralId = 1, bleachIndex = 0.2, flagged = true),
            AirtimeItem(coralId = 2, bleachIndex = 0.9, flagged = true),
            AirtimeItem(coralId = 3, bleachIndex = 0.6, flagged = true),
            AirtimeItem(coralId = 4, bleachIndex = 0.1, flagged = false)
        )
        val budget = items.size * LayerCodec.SHORT_AIRTIME_SEC + LayerCodec.LONG_UPGRADE_AIRTIME_SEC + 0.01

        val queue = LayerCodec.buildAirtimeQueue(items, budget)
        val upgrades = queue.filter { it.burst == AirtimeBurst.LONG_UPGRADE }

        assertEquals(items.size, queue.count { it.burst == AirtimeBurst.SHORT })
        assertEquals(listOf(2), upgrades.map { it.coralId })
        assertTrue(queue.sumOf { it.airtimeSec } <= budget)
    }

    private fun JSONArray.toStringList(): List<String> =
        List(length()) { index -> getString(index) }

    private fun resourceBytes(path: String): ByteArray =
        requireNotNull(javaClass.classLoader?.getResourceAsStream(path)) { "Missing test resource: $path" }
            .use { it.readBytes() }

    private fun readPcm16Wav(filename: String): FloatArray {
        val bytes = resourceBytes("golden/$filename")
        require(bytes.size >= 44 && bytes.ascii(0, 4) == "RIFF" && bytes.ascii(8, 4) == "WAVE") {
            "Invalid WAV file: $filename"
        }

        var format = -1
        var channels = -1
        var sampleRate = -1
        var bitsPerSample = -1
        var dataOffset = -1
        var dataSize = -1
        var offset = 12
        while (offset + 8 <= bytes.size) {
            val chunkName = bytes.ascii(offset, 4)
            val chunkSize = bytes.int32LE(offset + 4)
            val content = offset + 8
            require(chunkSize >= 0 && content + chunkSize <= bytes.size) { "Invalid WAV chunk: $chunkName" }
            when (chunkName) {
                "fmt " -> {
                    format = bytes.uint16LE(content)
                    channels = bytes.uint16LE(content + 2)
                    sampleRate = bytes.int32LE(content + 4)
                    bitsPerSample = bytes.uint16LE(content + 14)
                }
                "data" -> {
                    dataOffset = content
                    dataSize = chunkSize
                }
            }
            offset = content + chunkSize + (chunkSize and 1)
        }
        require(format == 1 && channels == 1 && bitsPerSample == 16 && dataOffset >= 0) {
            "Expected mono PCM16 WAV: $filename"
        }
        assertEquals(ModemCore.SAMPLE_RATE, sampleRate)
        return FloatArray(dataSize / 2) { index ->
            val raw = bytes.uint16LE(dataOffset + index * 2)
            val signed = if (raw >= 0x8000) raw - 0x10000 else raw
            signed / 32767.0f
        }
    }

    private fun ByteArray.ascii(offset: Int, length: Int): String =
        String(this, offset, length, StandardCharsets.US_ASCII)

    private fun ByteArray.uint16LE(offset: Int): Int =
        (this[offset].toInt() and 0xff) or ((this[offset + 1].toInt() and 0xff) shl 8)

    private fun ByteArray.int32LE(offset: Int): Int =
        uint16LE(offset) or (uint16LE(offset + 2) shl 16)

    private fun bitsToHex(bits: IntArray): String = buildString {
        for (start in bits.indices step 8) {
            var value = 0
            for (offset in 0 until 8) {
                value = value shl 1
                if (start + offset < bits.size) value = value or bits[start + offset]
            }
            append("%02x".format(value))
        }
    }

    private fun hexToBits(hex: String, bitCount: Int): IntArray {
        val bits = IntArray(hex.length * 4) { index ->
            (hex[index / 4].digitToInt(16) ushr (3 - index % 4)) and 1
        }
        require(bitCount <= bits.size)
        return bits.copyOf(bitCount)
    }
}
