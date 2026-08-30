package com.federated.copilot

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint

object OcrImagePreprocessor {

    /**
     * Conservative preprocessing fallback:
     * 1. Grayscale conversion via ColorMatrix
     * 2. Moderate contrast enhancement (1.35x contrast)
     * 3. Mild sharpening via 3x3 convolution kernel
     *
     * Used ONLY as a fallback pass when raw image ML Kit OCR yields low text or field count.
     * Does NOT aggressively threshold or distort original document geometry.
     */
    fun preprocessBitmap(original: Bitmap): Bitmap {
        return try {
            val width = original.width
            val height = original.height

            // Step 1 & 2: Grayscale + Moderate Contrast Enhancement
            val preprocessed = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(preprocessed)
            val paint = Paint(Paint.ANTI_ALIAS_FLAG)

            // ColorMatrix: Grayscale + Contrast scale 1.35 with mid-tone balance
            val contrast = 1.35f
            val translate = (-0.5f * contrast + 0.5f) * 255f
            val cm = ColorMatrix(floatArrayOf(
                0.3f * contrast, 0.59f * contrast, 0.11f * contrast, 0f, translate,
                0.3f * contrast, 0.59f * contrast, 0.11f * contrast, 0f, translate,
                0.3f * contrast, 0.59f * contrast, 0.11f * contrast, 0f, translate,
                0f, 0f, 0f, 1f, 0f
            ))

            paint.colorFilter = ColorMatrixColorFilter(cm)
            canvas.drawBitmap(original, 0f, 0f, paint)

            // Step 3: Mild Sharpening Filter (Pixel Buffer Convolution)
            sharpenBitmap(preprocessed)
        } catch (e: Exception) {
            // Fallback safely to original bitmap if memory or allocation fails
            original
        }
    }

    private fun sharpenBitmap(src: Bitmap): Bitmap {
        val width = src.width
        val height = src.height
        val pixels = IntArray(width * height)
        src.getPixels(pixels, 0, width, 0, 0, width, height)

        val outputPixels = IntArray(width * height)

        // 3x3 mild sharpening kernel:
        //  0  -1   0
        // -1   5  -1
        //  0  -1   0
        for (y in 1 until height - 1) {
            val rowOffset = y * width
            for (x in 1 until width - 1) {
                val idx = rowOffset + x

                val cCenter = pixels[idx]
                val cUp = pixels[idx - width]
                val cDown = pixels[idx + width]
                val cLeft = pixels[idx - 1]
                val cRight = pixels[idx + 1]

                val r = (5 * ((cCenter shr 16) and 0xFF)
                        - ((cUp shr 16) and 0xFF) - ((cDown shr 16) and 0xFF)
                        - ((cLeft shr 16) and 0xFF) - ((cRight shr 16) and 0xFF)).coerceIn(0, 255)

                val g = (5 * ((cCenter shr 8) and 0xFF)
                        - ((cUp shr 8) and 0xFF) - ((cDown shr 8) and 0xFF)
                        - ((cLeft shr 8) and 0xFF) - ((cRight shr 8) and 0xFF)).coerceIn(0, 255)

                val b = (5 * (cCenter and 0xFF)
                        - (cUp and 0xFF) - (cDown and 0xFF)
                        - (cLeft and 0xFF) - (cRight and 0xFF)).coerceIn(0, 255)

                outputPixels[idx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        // Copy border pixels unchanged
        System.arraycopy(pixels, 0, outputPixels, 0, width)
        System.arraycopy(pixels, (height - 1) * width, outputPixels, (height - 1) * width, width)

        val sharpened = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        sharpened.setPixels(outputPixels, 0, width, 0, 0, width, height)
        return sharpened
    }
}
