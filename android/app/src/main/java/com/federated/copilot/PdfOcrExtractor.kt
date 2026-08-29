package com.federated.copilot

import android.content.Context
import android.graphics.Bitmap
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import android.os.ParcelFileDescriptor
import java.io.File
import java.io.FileOutputStream

object PdfOcrExtractor {
    /**
     * Renders pages of a PDF Uri into Bitmap objects locally on device using PdfRenderer.
     */
    fun renderPdfToBitmaps(context: Context, pdfUri: Uri, maxPages: Int = 3): List<Bitmap> {
        val bitmaps = mutableListOf<Bitmap>()
        var fileDescriptor: ParcelFileDescriptor? = null
        var renderer: PdfRenderer? = null

        try {
            val contentResolver = context.contentResolver
            val inputStream = contentResolver.openInputStream(pdfUri) ?: return emptyList()

            // Copy to temp file so ParcelFileDescriptor can read it
            val tempFile = File(context.cacheDir, "temp_report_${System.currentTimeMillis()}.pdf")
            FileOutputStream(tempFile).use { outputStream ->
                inputStream.copyTo(outputStream)
            }

            fileDescriptor = ParcelFileDescriptor.open(tempFile, ParcelFileDescriptor.MODE_READ_ONLY)
            renderer = PdfRenderer(fileDescriptor)

            val pageCount = Math.min(renderer.pageCount, maxPages)
            for (i in 0 until pageCount) {
                val page = renderer.openPage(i)
                val width = page.width * 2 // 2x scale for crisp text recognition
                val height = page.height * 2

                val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                page.render(bitmap, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
                bitmaps.add(bitmap)
                page.close()
            }
            tempFile.delete()
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            try {
                renderer?.close()
                fileDescriptor?.close()
            } catch (e: Exception) {
                // Ignore close errors
            }
        }
        return bitmaps
    }
}
