package com.federated.copilot

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import org.json.JSONObject
import java.io.File
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.exp
import kotlin.math.max

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "CardioSense_OCR"
    }

    private var cameraPhotoUri: Uri? = null

    // Screens
    private lateinit var screenWelcome: View
    private lateinit var screenDashboard: View
    private lateinit var screenOcrUpload: View
    private lateinit var screenOcrReview: View
    private lateinit var screenAssessment: View
    private lateinit var screenResult: View
    private lateinit var screenHealthTips: View
    private lateinit var screenHowItWorks: View
    private lateinit var screenHistory: View

    // Navigation & Header Status
    private lateinit var tvHeaderStatus: TextView
    private lateinit var spinnerMode: Spinner
    private lateinit var tvModeDescription: TextView

    // Bottom Nav Labels
    private lateinit var navLblHome: TextView
    private lateinit var navLblAssess: TextView
    private lateinit var navLblHistory: TextView
    private lateinit var navLblTips: TextView

    // Bottom Navigation Buttons
    private lateinit var navBtnHome: View
    private lateinit var navBtnAssess: View
    private lateinit var navBtnHistory: View
    private lateinit var navBtnTips: View

    // Assessment Inputs
    private lateinit var etAge: EditText
    private lateinit var etHeight: EditText
    private lateinit var etWeight: EditText
    private lateinit var etApHi: EditText
    private lateinit var etApLo: EditText

    private lateinit var spinnerGender: Spinner
    private lateinit var spinnerCholesterol: Spinner
    private lateinit var spinnerGlucose: Spinner
    private lateinit var spinnerSmoke: Spinner
    private lateinit var spinnerAlco: Spinner
    private lateinit var spinnerActive: Spinner

    // Result Views
    private lateinit var cardRiskPrimary: LinearLayout
    private lateinit var tvRiskTag: TextView
    private lateinit var tvProbability: TextView
    private lateinit var tvContributingFactors: TextView

    // Scale Views
    private lateinit var vScaleLow: View
    private lateinit var vScaleMod: View
    private lateinit var vScaleElev: View
    private lateinit var vScaleHigh: View

    private lateinit var tvLblLow: TextView
    private lateinit var tvLblMod: TextView
    private lateinit var tvLblElev: TextView
    private lateinit var tvLblHigh: TextView

    // Summary Snapshot Views
    private lateinit var tvResAge: TextView
    private lateinit var tvResBp: TextView
    private lateinit var tvResBmi: TextView
    private lateinit var tvResCholesterol: TextView
    private lateinit var tvResGlucose: TextView

    // History Views
    private lateinit var tvHistoryEmpty: TextView
    private lateinit var containerHistoryList: LinearLayout

    // OCR Alert
    private lateinit var tvOcrStatusAlert: TextView

    // State Variables
    private var lastComputedProb: Double = 0.0
    private var lastComputedCategory: String = "Unknown"
    private var lastComputedApHi: Int = 0
    private var lastComputedApLo: Int = 0
    private var lastComputedAge: Double = 0.0

    // Offline Assets & Executor
    private var weightsJson: JSONObject? = null
    private var scalerJson: JSONObject? = null
    private val executor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    // Activity Result Launchers


    // 1. Take Photo (Full Resolution JPEG via FileProvider)
    private val takePhotoLauncher = registerForActivityResult(ActivityResultContracts.TakePicture()) { success: Boolean ->
        if (success && cameraPhotoUri != null) {
            Log.d(TAG, "========== [OCR-A] INPUT ==========")
            Log.d(TAG, "URI: $cameraPhotoUri")
            Log.d(TAG, "TYPE: image/jpeg")
            Log.d(TAG, "SOURCE: Camera (Full-resolution)")
            Log.d(TAG, "===================================")
            processOcrImageUri(cameraPhotoUri!!)
        } else {
            Log.w(TAG, "[OCR-A] Camera capture cancelled or failed")
        }
    }


    // 2. Choose Image (JPG, PNG, WEBP, HEIC - ContentResolver openInputStream)
    private val selectImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) {
            processOcrImageUri(uri)
        } else {
            Log.w(TAG, "[STAGE A - IMAGE INPUT] Selection returned null Uri")
        }
    }

    // 3. Choose PDF Document
    private val selectPdfLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) {
            val mime = contentResolver.getType(uri) ?: ""
            if (mime.contains("pdf", ignoreCase = true) || uri.toString().endsWith(".pdf", ignoreCase = true)) {
                processOcrPdf(uri)
            } else {
                processOcrImageUri(uri)
            }
        } else {
            Log.w(TAG, "[STAGE A - PDF INPUT] Selection returned null Uri")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        setupSpinners()
        loadLocalAssetsAsync()
        runDeveloperParserUnitTest()
    }

    private fun runDeveloperParserUnitTest() {
        val testReportStr1 = """
            Patient Medical Report

            Age: 34 years
            Height: 172 cm
            Weight: 68 kg
            Blood Pressure: 118/76 mmHg
            Total Cholesterol: 182 mg/dL
            Smoking Status: No
            Alcohol Consumption: No
            Physical Activity: Yes
        """.trimIndent()

        val disconnectedColumnOcrText = """
            Patient Information
            Patient Name
            Age
            Gender
            Height
            Weight
            Clinical Measurements
            Blood Pressure
            Cholesterol
            Glucose
            Lifestyle Information
            Smoking
            Alcohol Consumption
            Physical Activity

            Rahul Sharma
            55 years
            Female
            165 cm
            70 kg
            140 / 90 mmHg
            Level 2
            Level 1
            No
            No
            Yes
        """.trimIndent()

        Log.d(TAG, "==================== MANDATORY PARSER UNIT TEST ====================")
        val ocrResult1 = OcrReportParser.parseText(testReportStr1)
        val data1 = ocrResult1.parsedData
        val passed1 = data1.age == 34.0 && data1.height == 172.0 && data1.weight == 68.0 &&
                data1.apHi == 118 && data1.apLo == 76 && data1.cholesterol == 1 &&
                data1.smoke == 0 && data1.alco == 0 && data1.active == 1

        Log.d(TAG, "[TEST REPORT 1 - INLINE]: ${if (passed1) "PASS" else "FAIL"} (${data1.fieldsFoundCount()}/10 fields matched)")

        val ocrResult2 = OcrReportParser.parseText(disconnectedColumnOcrText)
        val data2 = ocrResult2.parsedData
        val passed2 = data2.age == 55.0 && data2.gender == 1 && data2.height == 165.0 && data2.weight == 70.0 &&
                data2.apHi == 140 && data2.apLo == 90 && data2.cholesterol == 2 && data2.glucose == 1 &&
                data2.smoke == 0 && data2.alco == 0 && data2.active == 1

        Log.d(TAG, "[TEST REPORT 2 - 2-COLUMN DISCONNECTED TABLE]: ${if (passed2) "PASS (10/10 MATCHED!)" else "FAIL: $data2"} (${data2.fieldsFoundCount()}/10 fields matched)")
        Log.d(TAG, "==========================================================================")
    }

    private fun initViews() {
        screenWelcome = findViewById(R.id.screen_welcome)
        screenDashboard = findViewById(R.id.screen_dashboard)
        screenOcrUpload = findViewById(R.id.screen_ocr_upload)
        screenOcrReview = findViewById(R.id.screen_ocr_review)
        screenAssessment = findViewById(R.id.screen_assessment)
        screenResult = findViewById(R.id.screen_result)
        screenHealthTips = findViewById(R.id.screen_health_tips)
        screenHowItWorks = findViewById(R.id.screen_how_it_works)
        screenHistory = findViewById(R.id.screen_history)

        tvHeaderStatus = findViewById(R.id.tv_header_status)
        spinnerMode = findViewById(R.id.spinner_inference_mode)
        tvModeDescription = findViewById(R.id.tv_mode_description)

        navLblHome = findViewById(R.id.nav_lbl_home)
        navLblAssess = findViewById(R.id.nav_lbl_assess)
        navLblHistory = findViewById(R.id.nav_lbl_history)
        navLblTips = findViewById(R.id.nav_lbl_tips)

        navBtnHome = findViewById(R.id.nav_btn_home)
        navBtnAssess = findViewById(R.id.nav_btn_assess)
        navBtnHistory = findViewById(R.id.nav_btn_history)
        navBtnTips = findViewById(R.id.nav_btn_tips)

        etAge = findViewById(R.id.et_age)
        etHeight = findViewById(R.id.et_height)
        etWeight = findViewById(R.id.et_weight)
        etApHi = findViewById(R.id.et_ap_hi)
        etApLo = findViewById(R.id.et_ap_lo)

        spinnerGender = findViewById(R.id.spinner_gender)
        spinnerCholesterol = findViewById(R.id.spinner_cholesterol)
        spinnerGlucose = findViewById(R.id.spinner_glucose)
        spinnerSmoke = findViewById(R.id.spinner_smoke)
        spinnerAlco = findViewById(R.id.spinner_alco)
        spinnerActive = findViewById(R.id.spinner_active)

        cardRiskPrimary = findViewById(R.id.card_risk_primary)
        tvRiskTag = findViewById(R.id.tv_risk_tag)
        tvProbability = findViewById(R.id.tv_probability)
        tvContributingFactors = findViewById(R.id.tv_contributing_factors)

        vScaleLow = findViewById(R.id.v_scale_low)
        vScaleMod = findViewById(R.id.v_scale_mod)
        vScaleElev = findViewById(R.id.v_scale_elev)
        vScaleHigh = findViewById(R.id.v_scale_high)

        tvLblLow = findViewById(R.id.tv_lbl_low)
        tvLblMod = findViewById(R.id.tv_lbl_mod)
        tvLblElev = findViewById(R.id.tv_lbl_elev)
        tvLblHigh = findViewById(R.id.tv_lbl_high)

        tvResAge = findViewById(R.id.tv_res_age)
        tvResBp = findViewById(R.id.tv_res_bp)
        tvResBmi = findViewById(R.id.tv_res_bmi)
        tvResCholesterol = findViewById(R.id.tv_res_cholesterol)
        tvResGlucose = findViewById(R.id.tv_res_glucose)

        tvHistoryEmpty = findViewById(R.id.tv_history_empty)
        containerHistoryList = findViewById(R.id.container_history_list)
        tvOcrStatusAlert = findViewById(R.id.tv_ocr_status_alert)

        // Welcome listeners
        findViewById<Button>(R.id.btn_welcome_start).setOnClickListener {
            showScreen(screenDashboard, navLblHome)
        }
        findViewById<TextView>(R.id.btn_welcome_how).setOnClickListener {
            showScreen(screenHowItWorks, navLblHome)
        }

        // Dashboard cards
        findViewById<View>(R.id.dash_card_assess).setOnClickListener {
            resetAssessmentForm()
            showScreen(screenAssessment, navLblAssess)
        }
        findViewById<View>(R.id.dash_card_ocr).setOnClickListener {
            showScreen(screenOcrUpload, navLblAssess)
        }
        findViewById<View>(R.id.dash_card_tips).setOnClickListener {
            showScreen(screenHealthTips, navLblTips)
        }
        findViewById<View>(R.id.dash_card_history).setOnClickListener {
            loadAndDisplayHistory()
            showScreen(screenHistory, navLblHistory)
        }

        // Report Upload Actions
        findViewById<Button>(R.id.btn_select_ocr_image).setOnClickListener {
            selectImageLauncher.launch("image/*")
        }
        findViewById<Button>(R.id.btn_select_ocr_pdf).setOnClickListener {
            selectPdfLauncher.launch("application/pdf")
        }
        findViewById<Button>(R.id.btn_take_photo_ocr).setOnClickListener {
            try {
                val photoFile = File(cacheDir, "camera_photo_${System.currentTimeMillis()}.jpg")
                cameraPhotoUri = FileProvider.getUriForFile(
                    this,
                    "$packageName.fileprovider",
                    photoFile
                )
                takePhotoLauncher.launch(cameraPhotoUri)
            } catch (e: Exception) {
                Log.e(TAG, "[OCR-A] Failed to create camera file URI: ${e.message}", e)
                Toast.makeText(this, "Camera error: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }


        // OCR Review Action
        findViewById<Button>(R.id.btn_confirm_ocr_review).setOnClickListener {
            showScreen(screenAssessment, navLblAssess)
        }

        // Assessment Actions
        findViewById<TextView>(R.id.btn_cancel_assessment).setOnClickListener {
            showScreen(screenDashboard, navLblHome)
        }
        findViewById<Button>(R.id.btn_analyze_patient).setOnClickListener {
            handleAnalyzePatient()
        }

        // Result Actions
        findViewById<Button>(R.id.btn_save_history).setOnClickListener {
            HistoryManager.saveAssessment(
                this,
                lastComputedProb,
                lastComputedCategory,
                lastComputedApHi,
                lastComputedApLo,
                lastComputedAge
            )
            Toast.makeText(this, "Assessment saved to history", Toast.LENGTH_SHORT).show()
            loadAndDisplayHistory()
            showScreen(screenHistory, navLblHistory)
        }

        findViewById<Button>(R.id.btn_new_assessment).setOnClickListener {
            resetAssessmentForm()
            showScreen(screenAssessment, navLblAssess)
        }

        // Bottom Navigation Bar Listeners
        navBtnHome.setOnClickListener { showScreen(screenDashboard, navLblHome) }
        navBtnAssess.setOnClickListener { showScreen(screenAssessment, navLblAssess) }
        navBtnHistory.setOnClickListener {
            loadAndDisplayHistory()
            showScreen(screenHistory, navLblHistory)
        }
        navBtnTips.setOnClickListener { showScreen(screenHealthTips, navLblTips) }
    }

    private fun showScreen(target: View, activeNavLabel: TextView? = null) {
        screenWelcome.visibility = View.GONE
        screenDashboard.visibility = View.GONE
        screenOcrUpload.visibility = View.GONE
        screenOcrReview.visibility = View.GONE
        screenAssessment.visibility = View.GONE
        screenResult.visibility = View.GONE
        screenHealthTips.visibility = View.GONE
        screenHowItWorks.visibility = View.GONE
        screenHistory.visibility = View.GONE

        target.visibility = View.VISIBLE

        navLblHome.setTextColor(Color.parseColor("#64748B"))
        navLblAssess.setTextColor(Color.parseColor("#64748B"))
        navLblHistory.setTextColor(Color.parseColor("#64748B"))
        navLblTips.setTextColor(Color.parseColor("#64748B"))

        activeNavLabel?.setTextColor(Color.parseColor("#0D9488"))
    }

    private fun createCustomAdapter(items: Array<String>): ArrayAdapter<String> {
        return ArrayAdapter(this, R.layout.spinner_item, items).apply {
            setDropDownViewResource(R.layout.spinner_dropdown_item)
        }
    }

    private fun setupSpinners() {
        val modeOptions = arrayOf("Offline Mode", "Online Mode")
        spinnerMode.adapter = createCustomAdapter(modeOptions)

        spinnerMode.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (position == 1) {
                    tvHeaderStatus.text = "[ Online ]"
                    tvHeaderStatus.setBackgroundColor(Color.parseColor("#10B981"))
                    tvModeDescription.text = "Connected to central server"
                } else {
                    tvHeaderStatus.text = "[ Offline ]"
                    tvHeaderStatus.setBackgroundColor(Color.parseColor("#0F766E"))
                    tvModeDescription.text = "On-device local engine"
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        spinnerGender.adapter = createCustomAdapter(arrayOf("-- Select Gender --", "Female", "Male"))
        spinnerCholesterol.adapter = createCustomAdapter(arrayOf("-- Select Cholesterol --", "Normal (< 200 mg/dL)", "Above Normal (200-239 mg/dL)", "Well Above Normal (≥ 240 mg/dL)"))
        spinnerGlucose.adapter = createCustomAdapter(arrayOf("-- Select Glucose --", "Normal (< 100 mg/dL)", "Above Normal (100-125 mg/dL)", "Well Above Normal (≥ 126 mg/dL)"))
        spinnerSmoke.adapter = createCustomAdapter(arrayOf("-- Select Smoking Status --", "Non-Smoker", "Active Smoker"))
        spinnerAlco.adapter = createCustomAdapter(arrayOf("-- Select Alcohol Intake --", "Non-Drinker", "Regular Drinker"))
        spinnerActive.adapter = createCustomAdapter(arrayOf("-- Select Physical Activity --", "Physically Inactive", "Physically Active"))

        resetAssessmentForm()
    }

    private fun resetAssessmentForm() {
        etAge.setText("")
        etHeight.setText("")
        etWeight.setText("")
        etApHi.setText("")
        etApLo.setText("")
        spinnerGender.setSelection(0)
        spinnerCholesterol.setSelection(0)
        spinnerGlucose.setSelection(0)
        spinnerSmoke.setSelection(0)
        spinnerAlco.setSelection(0)
        spinnerActive.setSelection(0)
    }

    private fun loadLocalAssetsAsync() {
        executor.execute {
            try {
                val wStream: InputStream = assets.open("global_model_weights.json")
                val wText = wStream.bufferedReader().use { it.readText() }
                weightsJson = JSONObject(wText)

                val sStream: InputStream = assets.open("scaler_params.json")
                val sText = sStream.bufferedReader().use { it.readText() }
                scalerJson = JSONObject(sText)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private fun loadBitmapFromUri(context: Context, uri: Uri): Bitmap? {
        return try {
            val contentResolver = context.contentResolver

            // 1. Check bounds and downsample if image is excessively large to prevent OOM
            var inSampleSize = 1
            val boundsOptions = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            contentResolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream, null, boundsOptions)
            }
            val maxDim = Math.max(boundsOptions.outWidth, boundsOptions.outHeight)
            if (maxDim > 3000) {
                inSampleSize = 2
            }

            val decodeOptions = BitmapFactory.Options().apply {
                this.inSampleSize = inSampleSize
                this.inPreferredConfig = Bitmap.Config.ARGB_8888
            }

            var bitmap = contentResolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream, null, decodeOptions)
            }

            if (bitmap == null) {
                Log.e(TAG, "========== [OCR-B] BITMAP ==========")
                Log.e(TAG, "WIDTH: 0")
                Log.e(TAG, "HEIGHT: 0")
                Log.e(TAG, "VALID: false")
                Log.e(TAG, "ORIENTATION: 0°")
                Log.e(TAG, "=====================================")
                return null
            }

            // 2. Read EXIF Orientation and rotate if necessary
            var rotationDegrees = 0f
            try {
                contentResolver.openInputStream(uri)?.use { stream ->
                    val exif = ExifInterface(stream)
                    val orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED)
                    rotationDegrees = when (orientation) {
                        ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                        ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                        ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                        else -> 0f
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "[OCR-B] EXIF reading exception: ${e.message}")
            }

            val finalBitmap = if (rotationDegrees != 0f) {
                val matrix = Matrix().apply { postRotate(rotationDegrees) }
                val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                bitmap.recycle()
                rotated
            } else {
                bitmap
            }

            Log.d(TAG, "========== [OCR-B] BITMAP ==========")
            Log.d(TAG, "WIDTH: ${finalBitmap.width}")
            Log.d(TAG, "HEIGHT: ${finalBitmap.height}")
            Log.d(TAG, "VALID: true")
            Log.d(TAG, "ORIENTATION: ${rotationDegrees.toInt()}°")
            Log.d(TAG, "=====================================")

            finalBitmap
        } catch (e: Exception) {
            Log.e(TAG, "[OCR-B ERROR] Failed to decode Uri $uri: ${e.message}", e)
            null
        }
    }

    private fun processOcrImageUri(uri: Uri) {
        try {
            val mime = contentResolver.getType(uri) ?: ""
            Log.d(TAG, "========== [OCR-A] INPUT ==========")
            Log.d(TAG, "URI: $uri")
            Log.d(TAG, "TYPE: $mime")
            Log.d(TAG, "SOURCE: Gallery / Photo Picker / Uri")
            Log.d(TAG, "===================================")

            if (mime.contains("pdf", ignoreCase = true) || uri.toString().endsWith(".pdf", ignoreCase = true)) {
                processOcrPdf(uri)
                return
            }

            Toast.makeText(this, "Processing medical report on-device...", Toast.LENGTH_SHORT).show()

            val bitmap = loadBitmapFromUri(this, uri)
            if (bitmap == null) {
                val failedResult = OcrResult(
                    success = false,
                    rawText = "",
                    charCount = 0,
                    parsedData = OcrExtractedPatient(),
                    fieldsFoundCount = 0,
                    statusMessage = "Image Decoding Failed: unable to read Uri $uri",
                    errorMsg = "Unable to decode image from Uri"
                )
                populateAssessmentFromOcr(failedResult)
                showScreen(screenAssessment, navLblAssess)
                return
            }

            processOcrBitmap(bitmap)
        } catch (e: Exception) {
            Log.e(TAG, "[OCR-A/B ERROR] ${e.message}", e)
            val errResult = OcrResult(
                success = false,
                rawText = "",
                charCount = 0,
                parsedData = OcrExtractedPatient(),
                fieldsFoundCount = 0,
                statusMessage = "Error processing image: ${e.message}",
                errorMsg = e.message
            )
            populateAssessmentFromOcr(errResult)
            showScreen(screenAssessment, navLblAssess)
        }
    }

    private fun processOcrBitmap(bitmap: Bitmap) {
        try {
            val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
            val pass1Image = InputImage.fromBitmap(bitmap, 0)

            // PASS 1: ML Kit OCR on original properly oriented bitmap
            recognizer.process(pass1Image)
                .addOnSuccessListener { visionText1 ->
                    val rawText1 = visionText1.text
                    val blockCount1 = visionText1.textBlocks.size
                    val result1 = OcrReportParser.parseText(rawText1)

                    Log.d(TAG, "========== [OCR-C] ML KIT (PASS 1 - RAW IMAGE) ==========")
                    Log.d(TAG, "STATUS: SUCCESS")
                    Log.d(TAG, "CHARACTER COUNT: ${rawText1.length}")
                    Log.d(TAG, "BLOCK COUNT: $blockCount1")
                    Log.d(TAG, "FIELDS DETECTED: ${result1.fieldsFoundCount}/10")
                    Log.d(TAG, "==========================================================")

                    // If Pass 1 extracted >= 7 fields or ample text, accept immediately
                    if (result1.fieldsFoundCount >= 7 || (result1.fieldsFoundCount >= 4 && rawText1.length > 150)) {
                        Log.d(TAG, "========== [OCR-D] RAW TEXT ==========\n$rawText1\n=======================================")
                        populateAssessmentFromOcr(result1)
                        showScreen(screenAssessment, navLblAssess)
                        return@addOnSuccessListener
                    }

                    // PASS 2: Conservative Image Preprocessing Fallback
                    Log.d(TAG, "[OCR-C] Pass 1 yielded ${result1.fieldsFoundCount}/10 fields. Running Pass 2 (Conservative Preprocessing Fallback)...")
                    val preprocessedBitmap = OcrImagePreprocessor.preprocessBitmap(bitmap)
                    val pass2Image = InputImage.fromBitmap(preprocessedBitmap, 0)

                    recognizer.process(pass2Image)
                        .addOnSuccessListener { visionText2 ->
                            val rawText2 = visionText2.text
                            val blockCount2 = visionText2.textBlocks.size
                            val result2 = OcrReportParser.parseText(rawText2)

                            Log.d(TAG, "========== [OCR-C] ML KIT (PASS 2 - PREPROCESSED) ==========")
                            Log.d(TAG, "STATUS: SUCCESS")
                            Log.d(TAG, "CHARACTER COUNT: ${rawText2.length}")
                            Log.d(TAG, "BLOCK COUNT: $blockCount2")
                            Log.d(TAG, "FIELDS DETECTED: ${result2.fieldsFoundCount}/10")
                            Log.d(TAG, "=============================================================")

                            val finalResult = if (result2.fieldsFoundCount > result1.fieldsFoundCount) {
                                Log.d(TAG, "[OCR-C] Pass 2 Preprocessed extracted MORE fields (${result2.fieldsFoundCount} vs ${result1.fieldsFoundCount}). Using Pass 2.")
                                Log.d(TAG, "========== [OCR-D] RAW TEXT ==========\n$rawText2\n=======================================")
                                result2
                            } else {
                                Log.d(TAG, "[OCR-C] Pass 1 Raw Image extracted EQUAL OR MORE fields (${result1.fieldsFoundCount} vs ${result2.fieldsFoundCount}). Using Pass 1.")
                                Log.d(TAG, "========== [OCR-D] RAW TEXT ==========\n$rawText1\n=======================================")
                                result1
                            }

                            populateAssessmentFromOcr(finalResult)
                            showScreen(screenAssessment, navLblAssess)
                        }
                        .addOnFailureListener { e ->
                            Log.w(TAG, "[OCR-C] Pass 2 failed: ${e.message}. Falling back to Pass 1 result.", e)
                            Log.d(TAG, "========== [OCR-D] RAW TEXT ==========\n$rawText1\n=======================================")
                            populateAssessmentFromOcr(result1)
                            showScreen(screenAssessment, navLblAssess)
                        }
                }
                .addOnFailureListener { e ->
                    Log.e(TAG, "========== [OCR-C] ML KIT ==========")
                    Log.e(TAG, "STATUS: FAILED (${e.message})")
                    Log.e(TAG, "CHARACTER COUNT: 0")
                    Log.e(TAG, "BLOCK COUNT: 0")
                    Log.e(TAG, "=====================================")
                    val failedResult = OcrResult(
                        success = false,
                        rawText = "",
                        charCount = 0,
                        parsedData = OcrExtractedPatient(),
                        fieldsFoundCount = 0,
                        statusMessage = "ML Kit Text Recognition Failed: ${e.message}",
                        errorMsg = e.message
                    )
                    populateAssessmentFromOcr(failedResult)
                    showScreen(screenAssessment, navLblAssess)
                }
        } catch (e: Exception) {
            Log.e(TAG, "[OCR-C EXCEPTION] ${e.message}", e)
            val exceptionResult = OcrResult(
                success = false,
                rawText = "",
                charCount = 0,
                parsedData = OcrExtractedPatient(),
                fieldsFoundCount = 0,
                statusMessage = "OCR Processing Exception: ${e.message}",
                errorMsg = e.message
            )
            populateAssessmentFromOcr(exceptionResult)
            showScreen(screenAssessment, navLblAssess)
        }
    }

    private fun processOcrPdf(pdfUri: Uri) {
        Log.d(TAG, "========== [OCR-A] INPUT ==========")
        Log.d(TAG, "URI: $pdfUri")
        Log.d(TAG, "TYPE: application/pdf")
        Log.d(TAG, "SOURCE: PDF Document")
        Log.d(TAG, "===================================")

        executor.execute {
            val pages = PdfOcrExtractor.renderPdfToBitmaps(this, pdfUri, maxPages = 5)
            if (pages.isEmpty()) {
                Log.e(TAG, "[OCR-B] PDF rendering returned 0 page Bitmaps")
                mainHandler.post {
                    val pdfFailResult = OcrResult(
                        success = false,
                        rawText = "",
                        charCount = 0,
                        parsedData = OcrExtractedPatient(),
                        fieldsFoundCount = 0,
                        statusMessage = "PDF Decoding Failed — 0 pages rendered",
                        errorMsg = "Unable to render PDF pages"
                    )
                    populateAssessmentFromOcr(pdfFailResult)
                    showScreen(screenAssessment, navLblAssess)
                }
                return@execute
            }

            Log.d(TAG, "========== [OCR-B] BITMAP (PDF) ==========")
            Log.d(TAG, "PAGES RENDERED: ${pages.size}")
            Log.d(TAG, "PAGE 1 DIMENSIONS: ${pages[0].width}x${pages[0].height}")
            Log.d(TAG, "VALID: true")
            Log.d(TAG, "==========================================")

            mainHandler.post {
                Toast.makeText(this, "Processing PDF report pages on-device...", Toast.LENGTH_SHORT).show()
            }

            val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
            val pageTexts = Array(pages.size) { "" }

            // Strictly ordered sequential page processing
            fun processPage(index: Int) {
                if (index >= pages.size) {
                    val combinedText = pageTexts.mapIndexed { idx, txt -> "--- PAGE ${idx + 1} ---\n$txt" }.joinToString("\n\n")
                    Log.d(TAG, "========== [OCR-C] ML KIT (PDF COMPLETE) ==========")
                    Log.d(TAG, "STATUS: SUCCESS")
                    Log.d(TAG, "TOTAL PAGES PROCESSED: ${pages.size}")
                    Log.d(TAG, "TOTAL CHARACTER COUNT: ${combinedText.length}")
                    Log.d(TAG, "===================================================")
                    Log.d(TAG, "========== [OCR-D] RAW TEXT (PDF) ==========\n$combinedText\n==========================================")

                    val ocrResult = OcrReportParser.parseText(combinedText)
                    mainHandler.post {
                        populateAssessmentFromOcr(ocrResult)
                        showScreen(screenAssessment, navLblAssess)
                    }
                    return
                }

                val pageBitmap = pages[index]
                val image = InputImage.fromBitmap(pageBitmap, 0)
                recognizer.process(image)
                    .addOnSuccessListener { visionText ->
                        pageTexts[index] = visionText.text
                        processPage(index + 1)
                    }
                    .addOnFailureListener { e ->
                        pageTexts[index] = "[Page ${index + 1} OCR Failed: ${e.message}]"
                        processPage(index + 1)
                    }
            }

            processPage(0)
        }
    }

    private fun populateAssessmentFromOcr(ocrResult: OcrResult) {
        resetAssessmentForm()
        val parsed = ocrResult.parsedData

        val missingFields = mutableListOf<String>()
        if (parsed.age == null) missingFields.add("Age")
        if (parsed.gender == null) missingFields.add("Gender")
        if (parsed.height == null) missingFields.add("Height")
        if (parsed.weight == null) missingFields.add("Weight")
        if (parsed.apHi == null || parsed.apLo == null) missingFields.add("Blood Pressure")
        if (parsed.cholesterol == null) missingFields.add("Cholesterol")
        if (parsed.glucose == null) missingFields.add("Glucose")
        if (parsed.smoke == null) missingFields.add("Smoking Status")
        if (parsed.alco == null) missingFields.add("Alcohol Intake")
        if (parsed.active == null) missingFields.add("Physical Activity")

        Log.d(TAG, "========== [OCR-E] PARSED DATA ==========")
        Log.d(TAG, "Age: ${parsed.age ?: "Not detected"}")
        Log.d(TAG, "Gender: ${when(parsed.gender) { 1 -> "Female (1)"; 2 -> "Male (2)"; else -> "Not detected" }}")
        Log.d(TAG, "Height: ${parsed.height ?: "Not detected"}")
        Log.d(TAG, "Weight: ${parsed.weight ?: "Not detected"}")
        Log.d(TAG, "SBP: ${parsed.apHi ?: "Not detected"}")
        Log.d(TAG, "DBP: ${parsed.apLo ?: "Not detected"}")
        Log.d(TAG, "Cholesterol: ${when(parsed.cholesterol) { 1 -> "Level 1 (Normal)"; 2 -> "Level 2 (Above Normal)"; 3 -> "Level 3 (Well Above Normal)"; else -> "Not detected" }}")
        Log.d(TAG, "Glucose: ${when(parsed.glucose) { 1 -> "Level 1 (Normal)"; 2 -> "Level 2 (Above Normal)"; 3 -> "Level 3 (Well Above Normal)"; else -> "Not detected" }}")
        Log.d(TAG, "Smoking: ${when(parsed.smoke) { 0 -> "Non-smoker (0)"; 1 -> "Active smoker (1)"; else -> "Not detected / Ambiguous" }}")
        Log.d(TAG, "Alcohol: ${when(parsed.alco) { 0 -> "Non-drinker (0)"; 1 -> "Regular drinker (1)"; else -> "Not detected / Ambiguous" }}")
        Log.d(TAG, "Activity: ${when(parsed.active) { 0 -> "Inactive (0)"; 1 -> "Active (1)"; else -> "Not detected" }}")
        Log.d(TAG, "Detected fields: ${parsed.fieldsFoundCount()}/10")
        Log.d(TAG, "Missing: ${if (missingFields.isEmpty()) "None" else missingFields.joinToString(", ")}")
        Log.d(TAG, "Ambiguous: ${if (parsed.ambiguousFields.isEmpty()) "None" else parsed.ambiguousFields.joinToString(", ")}")
        Log.d(TAG, "=========================================")

        // Automatically populate extracted values into form fields
        parsed.age?.let {
            val ageStr = if (it % 1.0 == 0.0) it.toInt().toString() else String.format(Locale.US, "%.1f", it)
            etAge.setText(ageStr)
        }
        parsed.height?.let {
            val hStr = if (it % 1.0 == 0.0) it.toInt().toString() else String.format(Locale.US, "%.1f", it)
            etHeight.setText(hStr)
        }
        parsed.weight?.let {
            val wStr = if (it % 1.0 == 0.0) it.toInt().toString() else String.format(Locale.US, "%.1f", it)
            etWeight.setText(wStr)
        }
        parsed.apHi?.let { etApHi.setText(it.toString()) }
        parsed.apLo?.let { etApLo.setText(it.toString()) }

        // Explicit Model Value -> Spinner Position Mapping
        parsed.gender?.let { if (it in 1..2) spinnerGender.setSelection(it) }
        parsed.cholesterol?.let { if (it in 1..3) spinnerCholesterol.setSelection(it) }
        parsed.glucose?.let { if (it in 1..3) spinnerGlucose.setSelection(it) }
        parsed.smoke?.let { if (it in 0..1) spinnerSmoke.setSelection(it + 1) }
        parsed.alco?.let { if (it in 0..1) spinnerAlco.setSelection(it + 1) }
        parsed.active?.let { if (it in 0..1) spinnerActive.setSelection(it + 1) }

        Log.d(TAG, "========== [OCR-F] UI POPULATION ==========")
        Log.d(TAG, "Age: '${etAge.text}'")
        Log.d(TAG, "Gender: '${spinnerGender.selectedItem}' (pos=${spinnerGender.selectedItemPosition})")
        Log.d(TAG, "Height: '${etHeight.text}'")
        Log.d(TAG, "Weight: '${etWeight.text}'")
        Log.d(TAG, "SBP: '${etApHi.text}'")
        Log.d(TAG, "DBP: '${etApLo.text}'")
        Log.d(TAG, "Cholesterol: '${spinnerCholesterol.selectedItem}' (pos=${spinnerCholesterol.selectedItemPosition})")
        Log.d(TAG, "Glucose: '${spinnerGlucose.selectedItem}' (pos=${spinnerGlucose.selectedItemPosition})")
        Log.d(TAG, "Smoking: '${spinnerSmoke.selectedItem}' (pos=${spinnerSmoke.selectedItemPosition})")
        Log.d(TAG, "Alcohol: '${spinnerAlco.selectedItem}' (pos=${spinnerAlco.selectedItemPosition})")
        Log.d(TAG, "Activity: '${spinnerActive.selectedItem}' (pos=${spinnerActive.selectedItemPosition})")
        Log.d(TAG, "============================================")

        val alertText = StringBuilder("OCR DIAGNOSTIC STATUS REPORT\n")
        val fieldsFound = parsed.fieldsFoundCount()
        if (fieldsFound == 10) {
            alertText.append("✅ OCR Successful — 10/10 fields auto-populated.\n")
        } else if (fieldsFound > 0) {
            alertText.append("ℹ️ OCR Completed — ").append(fieldsFound).append("/10 fields auto-populated.\n")
        } else {
            alertText.append("⚠️ OCR Completed — 0/10 clinical fields detected. Please enter details manually.\n")
        }

        alertText.append("• Characters Detected: ").append(ocrResult.charCount).append("\n")

        if (missingFields.isNotEmpty()) {
            alertText.append("\n⚠️ Missing fields requiring manual entry: ").append(missingFields.joinToString(", "))
        }
        if (parsed.ambiguousFields.isNotEmpty()) {
            alertText.append("\n\n❓ Needs verification: ").append(parsed.ambiguousFields.joinToString(", "))
        }
        if (parsed.warningMsg != null) {
            alertText.append("\n\n⚠️ ").append(parsed.warningMsg)
        }
        tvOcrStatusAlert.text = alertText.toString()
    }

    private fun handleAnalyzePatient() {
        val age = etAge.text.toString().toDoubleOrNull()
        val height = etHeight.text.toString().toDoubleOrNull()
        val weight = etWeight.text.toString().toDoubleOrNull()
        val apHi = etApHi.text.toString().toIntOrNull()
        val apLo = etApLo.text.toString().toIntOrNull()

        val genderPos = spinnerGender.selectedItemPosition
        val cholPos = spinnerCholesterol.selectedItemPosition
        val glucPos = spinnerGlucose.selectedItemPosition
        val smokePos = spinnerSmoke.selectedItemPosition
        val alcoPos = spinnerAlco.selectedItemPosition
        val activePos = spinnerActive.selectedItemPosition

        // Strict Validation: NO FALLBACK TO DEMO DATA!
        val missing = mutableListOf<String>()
        if (age == null || age < 1.0 || age > 120.0) missing.add("Age (1-120 yrs)")
        if (genderPos == 0) missing.add("Gender")
        if (height == null || height < 50.0 || height > 250.0) missing.add("Height (50-250 cm)")
        if (weight == null || weight < 20.0 || weight > 300.0) missing.add("Weight (20-300 kg)")
        if (apHi == null || apHi < 60 || apHi > 260) missing.add("Systolic BP")
        if (apLo == null || apLo < 40 || apLo > 180) missing.add("Diastolic BP")
        if (apHi != null && apLo != null && apLo >= apHi) missing.add("Systolic > Diastolic BP")
        if (cholPos == 0) missing.add("Cholesterol Level")
        if (glucPos == 0) missing.add("Glucose Level")
        if (smokePos == 0) missing.add("Smoking Status")
        if (alcoPos == 0) missing.add("Alcohol Intake")
        if (activePos == 0) missing.add("Physical Activity")

        if (missing.isNotEmpty()) {
            Toast.makeText(
                this,
                "Please complete all required fields:\n${missing.joinToString(", ")}",
                Toast.LENGTH_LONG
            ).show()
            return
        }

        // Convert UI Spinner Selected Positions back to ML Model Encoded Values
        val gender = genderPos // Pos 1 ("Female") -> 1, Pos 2 ("Male") -> 2
        val cholesterol = cholPos // Pos 1 ("Normal") -> 1, Pos 2 ("Above") -> 2, Pos 3 ("Well Above") -> 3
        val glucose = glucPos // Pos 1 ("Normal") -> 1, Pos 2 ("Above") -> 2, Pos 3 ("Well Above") -> 3
        val smoke = smokePos - 1 // Pos 1 ("Non-Smoker") -> 0, Pos 2 ("Active Smoker") -> 1
        val alco = alcoPos - 1 // Pos 1 ("Non-Drinker") -> 0, Pos 2 ("Regular Drinker") -> 1
        val active = activePos - 1 // Pos 1 ("Physically Inactive") -> 0, Pos 2 ("Physically Active") -> 1

        Log.d(TAG, "========== [MODEL INPUT VERIFICATION] ==========")
        Log.d(TAG, "SUBMITTED FORM TO MODEL:")
        Log.d(TAG, "age=$age, gender=$gender, height=$height, weight=$weight")
        Log.d(TAG, "ap_hi=$apHi, ap_lo=$apLo, cholesterol=$cholesterol, gluc=$glucose")
        Log.d(TAG, "smoke=$smoke, alco=$alco, active=$active")
        Log.d(TAG, "===============================================")

        val isOnlineMode = spinnerMode.selectedItemPosition == 1

        if (isOnlineMode) {
            executeOnlineInference(age!!, gender, height!!, weight!!, apHi!!, apLo!!, cholesterol, glucose, smoke, alco, active)
        } else {
            executeOfflineInference(age!!, gender, height!!, weight!!, apHi!!, apLo!!, cholesterol, glucose, smoke, alco, active)
        }
    }

    private fun executeOfflineInference(
        age: Double, gender: Int, height: Double, weight: Double,
        apHi: Int, apLo: Int, cholesterol: Int, glucose: Int,
        smoke: Int, alco: Int, active: Int
    ) {
        try {
            val sJson = scalerJson ?: JSONObject(assets.open("scaler_params.json").bufferedReader().readText())
            val wJson = weightsJson ?: JSONObject(assets.open("global_model_weights.json").bufferedReader().readText())

            val means = sJson.getJSONArray("mean")
            val scales = sJson.getJSONArray("scale")

            val ageStd = (age - means.getDouble(0)) / scales.getDouble(0)
            val heightStd = (height - means.getDouble(1)) / scales.getDouble(1)
            val weightStd = (weight - means.getDouble(2)) / scales.getDouble(2)
            val apHiStd = (apHi - means.getDouble(3)) / scales.getDouble(3)
            val apLoStd = (apLo - means.getDouble(4)) / scales.getDouble(4)

            val input = doubleArrayOf(
                ageStd, gender.toDouble(), heightStd, weightStd, apHiStd, apLoStd,
                cholesterol.toDouble(), glucose.toDouble(), smoke.toDouble(), alco.toDouble(), active.toDouble()
            )

            // Layer 1: 11 -> 64 with ReLU
            val fc1W = wJson.getJSONArray("fc1_w")
            val fc1B = wJson.getJSONArray("fc1_b")
            val h1 = DoubleArray(64)
            for (i in 0 until 64) {
                var sum = fc1B.getDouble(i)
                val row = fc1W.getJSONArray(i)
                for (j in 0 until 11) {
                    sum += row.getDouble(j) * input[j]
                }
                h1[i] = max(0.0, sum)
            }

            // Layer 2: 64 -> 32 with ReLU
            val fc2W = wJson.getJSONArray("fc2_w")
            val fc2B = wJson.getJSONArray("fc2_b")
            val h2 = DoubleArray(32)
            for (i in 0 until 32) {
                var sum = fc2B.getDouble(i)
                val row = fc2W.getJSONArray(i)
                for (j in 0 until 64) {
                    sum += row.getDouble(j) * h1[j]
                }
                h2[i] = max(0.0, sum)
            }

            // Layer 3: 32 -> 1 raw logit
            val fc3W = wJson.getJSONArray("fc3_w")
            val fc3B = wJson.getJSONArray("fc3_b")
            var logit = fc3B.getDouble(0)
            val row3 = fc3W.getJSONArray(0)
            for (j in 0 until 32) {
                logit += row3.getDouble(j) * h2[j]
            }

            val prob = 1.0 / (1.0 + exp(-logit))
            val predClass = if (prob >= 0.5) 1 else 0

            renderResult(predClass, prob, apHi, apLo, height, weight, cholesterol, glucose, active, smoke, age)

        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Assessment could not be completed. Please try again.", Toast.LENGTH_LONG).show()
        }
    }

    private fun executeOnlineInference(
        age: Double, gender: Int, height: Double, weight: Double,
        apHi: Int, apLo: Int, cholesterol: Int, glucose: Int,
        smoke: Int, alco: Int, active: Int
    ) {
        val payload = JSONObject().apply {
            put("age", age)
            put("gender", gender)
            put("height", height)
            put("weight", weight)
            put("ap_hi", apHi)
            put("ap_lo", apLo)
            put("cholesterol", cholesterol)
            put("gluc", glucose)
            put("smoke", smoke)
            put("alco", alco)
            put("active", active)
        }

        executor.execute {
            try {
                val url = URL("http://10.0.2.2:8000/predict")
                val conn = url.openConnection() as HttpURLConnection
                conn.requestMethod = "POST"
                conn.setRequestProperty("Content-Type", "application/json; utf-8")
                conn.doOutput = true
                conn.connectTimeout = 3000
                conn.readTimeout = 3000

                conn.outputStream.use { os ->
                    val input = payload.toString().toByteArray(charset("utf-8"))
                    os.write(input, 0, input.size)
                }

                val code = conn.responseCode
                if (code == 200) {
                    val responseStr = conn.inputStream.bufferedReader().use { it.readText() }
                    val resObj = JSONObject(responseStr)
                    val predClass = resObj.getInt("predicted_class")
                    val prob = resObj.getDouble("probability")

                    mainHandler.post {
                        renderResult(predClass, prob, apHi, apLo, height, weight, cholesterol, glucose, active, smoke, age)
                    }
                } else {
                    mainHandler.post {
                        Toast.makeText(this@MainActivity, "Server unavailable. Falling back to Offline Mode.", Toast.LENGTH_SHORT).show()
                        executeOfflineInference(age, gender, height, weight, apHi, apLo, cholesterol, glucose, smoke, alco, active)
                    }
                }
            } catch (e: Exception) {
                mainHandler.post {
                    Toast.makeText(this@MainActivity, "Server unavailable. Using Offline Mode.", Toast.LENGTH_SHORT).show()
                    executeOfflineInference(age, gender, height, weight, apHi, apLo, cholesterol, glucose, smoke, alco, active)
                }
            }
        }
    }

    private fun renderResult(
        predClass: Int, prob: Double,
        apHi: Int, apLo: Int, height: Double, weight: Double,
        cholesterol: Int, glucose: Int, active: Int, smoke: Int, age: Double
    ) {
        lastComputedProb = prob
        lastComputedApHi = apHi
        lastComputedApLo = apLo
        lastComputedAge = age

        val pct = String.format(Locale.US, "%.2f%%", prob * 100)
        tvProbability.text = pct

        val activeBg = R.drawable.bg_scale_active
        val inactiveBg = R.drawable.bg_scale_inactive

        vScaleLow.setBackgroundResource(inactiveBg)
        vScaleMod.setBackgroundResource(inactiveBg)
        vScaleElev.setBackgroundResource(inactiveBg)
        vScaleHigh.setBackgroundResource(inactiveBg)

        tvLblLow.setTextColor(Color.parseColor("#64748B"))
        tvLblMod.setTextColor(Color.parseColor("#64748B"))
        tvLblElev.setTextColor(Color.parseColor("#64748B"))
        tvLblHigh.setTextColor(Color.parseColor("#64748B"))

        when {
            prob < 0.30 -> {
                lastComputedCategory = "Low Risk"
                tvRiskTag.text = "LOW POTENTIAL RISK"
                tvRiskTag.setTextColor(Color.parseColor("#065F46"))
                tvRiskTag.setBackgroundColor(Color.parseColor("#D1FAE5"))
                cardRiskPrimary.setBackgroundColor(Color.parseColor("#ECFDF5"))

                vScaleLow.setBackgroundResource(activeBg)
                tvLblLow.setTextColor(Color.parseColor("#0D9488"))
            }
            prob < 0.50 -> {
                lastComputedCategory = "Moderate Risk"
                tvRiskTag.text = "MODERATE POTENTIAL RISK"
                tvRiskTag.setTextColor(Color.parseColor("#92400E"))
                tvRiskTag.setBackgroundColor(Color.parseColor("#FEF3C7"))
                cardRiskPrimary.setBackgroundColor(Color.parseColor("#FFFBEB"))

                vScaleMod.setBackgroundResource(activeBg)
                tvLblMod.setTextColor(Color.parseColor("#0D9488"))
            }
            prob < 0.75 -> {
                lastComputedCategory = "Elevated Risk"
                tvRiskTag.text = "ELEVATED POTENTIAL RISK"
                tvRiskTag.setTextColor(Color.parseColor("#991B1B"))
                tvRiskTag.setBackgroundColor(Color.parseColor("#FEE2E2"))
                cardRiskPrimary.setBackgroundColor(Color.parseColor("#FEF2F2"))

                vScaleElev.setBackgroundResource(activeBg)
                tvLblElev.setTextColor(Color.parseColor("#0D9488"))
            }
            else -> {
                lastComputedCategory = "High Risk"
                tvRiskTag.text = "HIGH POTENTIAL RISK"
                tvRiskTag.setTextColor(Color.parseColor("#991B1B"))
                tvRiskTag.setBackgroundColor(Color.parseColor("#FEE2E2"))
                cardRiskPrimary.setBackgroundColor(Color.parseColor("#FEF2F2"))

                vScaleHigh.setBackgroundResource(activeBg)
                tvLblHigh.setTextColor(Color.parseColor("#0D9488"))
            }
        }

        tvResAge.text = String.format(Locale.US, "%.1f years", age)
        tvResBp.text = "$apHi/$apLo mmHg"

        val heightM = height / 100.0
        val bmi = if (heightM > 0) weight / (heightM * heightM) else 22.0
        tvResBmi.text = String.format(Locale.US, "%.1f kg/m²", bmi)

        tvResCholesterol.text = "Level $cholesterol"
        tvResGlucose.text = "Level $glucose"

        val factors = mutableListOf<String>()

        if (apHi >= 140 || apLo >= 90) {
            factors.add("• Elevated Blood Pressure ($apHi/$apLo mmHg)")
        } else if (apHi >= 130 || apLo >= 80) {
            factors.add("• Prehypertension BP range ($apHi/$apLo mmHg)")
        }

        if (cholesterol >= 2) {
            val cholDesc = if (cholesterol == 3) "Well Above Normal" else "Above Normal"
            factors.add("• Cholesterol Level ($cholDesc)")
        }

        if (glucose >= 2) {
            val glucDesc = if (glucose == 3) "Well Above Normal" else "Above Normal"
            factors.add("• Glucose Level ($glucDesc)")
        }

        if (active == 0) {
            factors.add("• Physical Inactivity (< 30 min daily activity)")
        }

        if (smoke == 1) {
            factors.add("• Active Smoking Status")
        }

        if (bmi >= 25.0) {
            factors.add(String.format(Locale.US, "• Elevated Body Mass Index (%.1f kg/m²)", bmi))
        }

        if (age >= 55.0) {
            factors.add(String.format(Locale.US, "• Age Baseline Factor (%.1f years)", age))
        }

        if (factors.isEmpty()) {
            factors.add("• Blood pressure within normal range ($apHi/$apLo mmHg)")
            factors.add("• Cholesterol and glucose levels normal")
            factors.add("• Active lifestyle reported")
        }

        tvContributingFactors.text = factors.joinToString("\n")
        showScreen(screenResult, navLblAssess)
    }

    private fun loadAndDisplayHistory() {
        val historyList = HistoryManager.getHistory(this)
        containerHistoryList.removeAllViews()

        if (historyList.isEmpty()) {
            tvHistoryEmpty.visibility = View.VISIBLE
        } else {
            tvHistoryEmpty.visibility = View.GONE
            for (rec in historyList) {
                val card = LinearLayout(this).apply {
                    orientation = LinearLayout.VERTICAL
                    setPadding(32, 24, 32, 24)
                    background = getDrawable(R.drawable.bg_card)
                    val params = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    )
                    params.setMargins(0, 0, 0, 16)
                    layoutParams = params
                }

                val headerLayout = LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL
                }

                val titleTv = TextView(this).apply {
                    text = rec.riskCategory.uppercase()
                    setTextSize(android.util.TypedValue.COMPLEX_UNIT_SP, 13f)
                    setTypeface(typeface, android.graphics.Typeface.BOLD)
                    setTextColor(Color.parseColor("#0D9488"))
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                }

                val probTv = TextView(this).apply {
                    text = String.format(Locale.US, "%.2f%%", rec.probability * 100)
                    setTextSize(android.util.TypedValue.COMPLEX_UNIT_SP, 15f)
                    setTypeface(typeface, android.graphics.Typeface.BOLD)
                    setTextColor(Color.parseColor("#0F172A"))
                }

                headerLayout.addView(titleTv)
                headerLayout.addView(probTv)

                val detailsTv = TextView(this).apply {
                    text = "BP: ${rec.bpStr}  |  Age: ${rec.ageStr}\nDate: ${rec.date}"
                    setTextSize(android.util.TypedValue.COMPLEX_UNIT_SP, 12f)
                    setTextColor(Color.parseColor("#64748B"))
                    setPadding(0, 8, 0, 0)
                }

                card.addView(headerLayout)
                card.addView(detailsTv)
                containerHistoryList.addView(card)
            }
        }
    }
}
