package com.federated.copilot

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.util.Log
import java.io.File
import java.io.FileOutputStream

object PdfOcrExtractor {

    private const val TAG = "CardioSense_OCR"

    /**
     * Renders pages of a PDF Uri into Bitmap objects locally on device using PdfRenderer.
     * Fills Canvas with solid white background to ensure high-contrast text for ML Kit OCR.
     */
    fun renderPdfToBitmaps(context: Context, pdfUri: Uri, maxPages: Int = 5): List<Bitmap> {
        val bitmaps = mutableListOf<Bitmap>()
        var fileDescriptor: ParcelFileDescriptor? = null
        var renderer: PdfRenderer? = null
        var tempFile: File? = null

        try {
            val contentResolver = context.contentResolver
            val inputStream = contentResolver.openInputStream(pdfUri)
            if (inputStream == null) {
                Log.e(TAG, "[STAGE B - PDF DECODING FAILED] InputStream is null for URI: $pdfUri")
                return emptyList()
            }

            tempFile = File(context.cacheDir, "temp_report_${System.currentTimeMillis()}.pdf")
            FileOutputStream(tempFile).use { outputStream ->
                inputStream.copyTo(outputStream)
            }

            fileDescriptor = ParcelFileDescriptor.open(tempFile, ParcelFileDescriptor.MODE_READ_ONLY)
            renderer = PdfRenderer(fileDescriptor)

            val pageCount = Math.min(renderer.pageCount, maxPages)
            Log.d(TAG, "[STAGE B - PDF DECODING SUCCESS] Total Pages: ${renderer.pageCount}, Rendering up to: $pageCount pages")

            for (i in 0 until pageCount) {
                val page = renderer.openPage(i)
                val width = page.width * 2 // 2x scale for crisp text recognition
                val height = page.height * 2

                val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                val canvas = Canvas(bitmap)
                canvas.drawColor(Color.WHITE) // Fill white background to prevent black-on-transparent rendering

                page.render(bitmap, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
                bitmaps.add(bitmap)
                page.close()
                Log.d(TAG, "[STAGE B - PDF PAGE $i RENDERED] Bitmap Size: ${width}x${height}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "[STAGE B - PDF RENDER ERROR] ${e.message}", e)
        } finally {
            try {
                renderer?.close()
                fileDescriptor?.close()
                tempFile?.delete()
            } catch (_: Exception) {}
        }
        return bitmaps
    }
}
