package org.reefglyph

import android.graphics.Paint as NativePaint
import android.graphics.Typeface
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.unit.dp
import androidx.compose.ui.platform.LocalDensity
import java.util.Locale
import kotlin.math.min

typealias L0Record = CoralRecord
typealias L1Record = LayerRecord
typealias L2Record = LayerRecord

@Composable
fun CoralGlyphCanvas(
    l0: L0Record?,
    l1: L1Record?,
    l2: L2Record?,
    modifier: Modifier
) {
    val density = LocalDensity.current
    val badgeHeight = with(density) { 26.dp.toPx() }
    val padding = with(density) { 8.dp.toPx() }
    val textSize = with(density) { 12.dp.toPx() }

    Canvas(modifier = modifier) {
        drawRect(Color.White)
        if (l0 == null) return@Canvas

        val colors = listOf(
            l0.colors.getOrNull(0)?.let(LayerCodec::labToRgb)?.toComposeColor()
                ?: Color.LightGray,
            l0.colors.getOrNull(1)?.let(LayerCodec::labToRgb)?.toComposeColor()
                ?: Color.Gray
        )

        val aspect = (l1?.aspect ?: 1.0).toFloat().coerceIn(0.25f, 4f)
        val availableWidth = (size.width - padding * 2).coerceAtLeast(1f)
        val availableHeight = (size.height - badgeHeight - padding * 2).coerceAtLeast(1f)
        val frameSize = min(availableWidth, availableHeight)
        val glyphWidth = if (aspect >= 1f) frameSize else frameSize * aspect
        val glyphHeight = if (aspect >= 1f) frameSize / aspect else frameSize
        val left = (size.width - glyphWidth) / 2f
        val top = badgeHeight + padding + (availableHeight - glyphHeight) / 2f

        if (l1?.efd == null) {
            val swatchSize = min(frameSize * 0.25f, with(density) { 56.dp.toPx() })
            val swatchGap = padding
            val swatchLeft = (size.width - swatchSize * 2 - swatchGap) / 2f
            val swatchTop = badgeHeight + padding + (availableHeight - swatchSize) / 2f
            drawRect(
                color = colors[0],
                topLeft = androidx.compose.ui.geometry.Offset(swatchLeft, swatchTop),
                size = androidx.compose.ui.geometry.Size(swatchSize, swatchSize)
            )
            drawRect(
                color = colors[1],
                topLeft = androidx.compose.ui.geometry.Offset(swatchLeft + swatchSize + swatchGap, swatchTop),
                size = androidx.compose.ui.geometry.Size(swatchSize, swatchSize)
            )
            drawIntoCanvas { canvas ->
                val paint = NativePaint(NativePaint.ANTI_ALIAS_FLAG).apply {
                    color = android.graphics.Color.DKGRAY
                    this.textSize = with(density) { 11.dp.toPx() }
                    typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
                }
                val label = "L0 SUMMARY ONLY"
                canvas.nativeCanvas.drawText(
                    label,
                    (size.width - paint.measureText(label)) / 2f,
                    badgeHeight + padding + paint.textSize,
                    paint
                )
            }
        } else {
            val contour = contourPath(l1, left, top, glyphWidth, glyphHeight)
            clipPath(contour) {
                if (l2?.thumb != null && l2.palette != null) {
                    val thumbnail = l2.thumb!!
                    val palette = l2.palette!!
                    for (y in 0 until 16) {
                        for (x in 0 until 16) {
                            val paletteIndex = thumbnail[y * 16 + x].coerceIn(0, 3)
                            drawRect(
                                color = palette[paletteIndex].toComposeColor(),
                                topLeft = androidx.compose.ui.geometry.Offset(
                                    left + glyphWidth * x / 16f,
                                    top + glyphHeight * y / 16f
                                ),
                                size = androidx.compose.ui.geometry.Size(
                                    glyphWidth / 16f + 0.5f,
                                    glyphHeight / 16f + 0.5f
                                )
                            )
                        }
                    }
                } else {
                    drawRect(
                        color = colors[0],
                        topLeft = androidx.compose.ui.geometry.Offset(left, top),
                        size = androidx.compose.ui.geometry.Size(glyphWidth / 2f, glyphHeight)
                    )
                    drawRect(
                        color = colors[1],
                        topLeft = androidx.compose.ui.geometry.Offset(left + glyphWidth / 2f, top),
                        size = androidx.compose.ui.geometry.Size(glyphWidth / 2f, glyphHeight)
                    )
                }
            }
            drawPath(
                path = contour,
                color = Color.Black.copy(alpha = 0.35f),
                style = Stroke(width = with(density) { 1.dp.toPx() })
            )
        }

        val badgeColor = if (l0.evidence != 0) Color(0xFFDC2626) else Color(0xFF16A34A)
        drawRect(color = badgeColor, size = androidx.compose.ui.geometry.Size(size.width, badgeHeight))
        drawIntoCanvas { canvas ->
            val paint = NativePaint(NativePaint.ANTI_ALIAS_FLAG).apply {
                color = android.graphics.Color.WHITE
                this.textSize = textSize
                typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            }
            val baseline = badgeHeight * 0.68f
            canvas.nativeCanvas.drawText(
                "BLEACH %.2f".format(Locale.US, l0.bleach),
                padding,
                baseline,
                paint
            )
            val texture = "T${l0.texture.coerceIn(0, 15)}"
            canvas.nativeCanvas.drawText(
                texture,
                size.width - padding - paint.measureText(texture),
                baseline,
                paint
            )
        }
    }
}

private fun contourPath(
    record: L1Record,
    left: Float,
    top: Float,
    width: Float,
    height: Float
): Path {
    val points = LayerCodec.reconstructContour(record, nPoints = 200)
    return Path().apply {
        points.forEachIndexed { index, point ->
            val x = left + point[0].coerceIn(0.0, 1.0).toFloat() * width
            val y = top + point[1].coerceIn(0.0, 1.0).toFloat() * height
            if (index == 0) moveTo(x, y) else lineTo(x, y)
        }
        close()
    }
}

private fun IntArray.toComposeColor(): Color =
    Color(
        red = getOrElse(0) { 0 }.coerceIn(0, 255) / 255f,
        green = getOrElse(1) { 0 }.coerceIn(0, 255) / 255f,
        blue = getOrElse(2) { 0 }.coerceIn(0, 255) / 255f
    )
