package com.federated.copilot

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import org.json.JSONObject
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.exp
import kotlin.math.max

class MainActivity : AppCompatActivity() {

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
    private var lastComputedApHi: Int = 140
    private var lastComputedApLo: Int = 90
    private var lastComputedAge: Double = 55.5

    // Offline Assets & Executor
    private var weightsJson: JSONObject? = null
    private var scalerJson: JSONObject? = null
    private val executor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    // Activity Result Launchers for Report Selection Options

    // 1. Take Photo
    private val takePhotoLauncher = registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap: Bitmap? ->
        if (bitmap != null) {
            val processed = OcrImagePreprocessor.preprocessBitmap(bitmap)
            processOcrBitmap(processed)
        }
    }

    // 2. Choose Image (JPG, PNG, WEBP, HEIC - Google Photos Compatible)
    private val selectImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) {
            processOcrImageUri(uri)
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
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        setupSpinners()
        loadLocalAssetsAsync()
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
            takePhotoLauncher.launch(null)
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

        spinnerGender.adapter = createCustomAdapter(arrayOf("Female", "Male"))
        spinnerCholesterol.adapter = createCustomAdapter(arrayOf("Normal (< 200 mg/dL)", "Above Normal (200-239 mg/dL)", "Well Above Normal (≥ 240 mg/dL)"))
        spinnerGlucose.adapter = createCustomAdapter(arrayOf("Normal (< 100 mg/dL)", "Above Normal (100-125 mg/dL)", "Well Above Normal (≥ 126 mg/dL)"))
        spinnerSmoke.adapter = createCustomAdapter(arrayOf("Non-Smoker", "Active Smoker"))
        spinnerAlco.adapter = createCustomAdapter(arrayOf("Non-Drinker", "Regular Drinker"))
        spinnerActive.adapter = createCustomAdapter(arrayOf("Physically Inactive", "Physically Active"))

        spinnerCholesterol.setSelection(1)
        spinnerActive.setSelection(1)
    }

    private fun resetAssessmentForm() {
        etAge.setText("55.5")
        etHeight.setText("165.0")
        etWeight.setText("70.0")
        etApHi.setText("140")
        etApLo.setText("90")
        spinnerGender.setSelection(0)
        spinnerCholesterol.setSelection(1)
        spinnerGlucose.setSelection(0)
        spinnerSmoke.setSelection(0)
        spinnerAlco.setSelection(0)
        spinnerActive.setSelection(1)
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

    private fun processOcrImageUri(uri: Uri) {
        try {
            val mime = contentResolver.getType(uri) ?: ""
            if (mime.contains("pdf", ignoreCase = true) || uri.toString().endsWith(".pdf", ignoreCase = true)) {
                processOcrPdf(uri)
                return
            }

            val bitmap: Bitmap? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri)) { decoder, _, _ ->
                    decoder.isMutableRequired = true
                }
            } else {
                contentResolver.openInputStream(uri).use { stream ->
                    BitmapFactory.decodeStream(stream)
                }
            }

            if (bitmap != null) {
                val processed = OcrImagePreprocessor.preprocessBitmap(bitmap)
                processOcrBitmap(processed)
            } else {
                Toast.makeText(this, "Could not load image. Please select another file.", Toast.LENGTH_SHORT).show()
                showScreen(screenAssessment, navLblAssess)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Error processing report image: ${e.message}", Toast.LENGTH_SHORT).show()
            showScreen(screenAssessment, navLblAssess)
        }
    }

    private fun processOcrBitmap(bitmap: Bitmap) {
        try {
            val image = InputImage.fromBitmap(bitmap, 0)
            val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

            Toast.makeText(this, "Processing report on-device...", Toast.LENGTH_SHORT).show()

            recognizer.process(image)
                .addOnSuccessListener { visionText ->
                    val rawText = visionText.text
                    if (rawText.isBlank()) {
                        Toast.makeText(this, "We couldn't read enough information from this report. Please enter information manually.", Toast.LENGTH_LONG).show()
                        showScreen(screenAssessment, navLblAssess)
                        return@addOnSuccessListener
                    }

                    val parsed = OcrReportParser.parseText(rawText)
                    populateAssessmentFromOcr(parsed)
                    showScreen(screenOcrReview, navLblAssess)
                }
                .addOnFailureListener { e ->
                    Toast.makeText(this, "OCR failed: ${e.message}. Please enter information manually.", Toast.LENGTH_LONG).show()
                    showScreen(screenAssessment, navLblAssess)
                }
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Error processing report: ${e.message}", Toast.LENGTH_SHORT).show()
            showScreen(screenAssessment, navLblAssess)
        }
    }

    private fun processOcrPdf(pdfUri: Uri) {
        executor.execute {
            val pages = PdfOcrExtractor.renderPdfToBitmaps(this, pdfUri, maxPages = 3)
            if (pages.isEmpty()) {
                mainHandler.post {
                    Toast.makeText(this, "Could not render PDF document. Please choose an image or enter manually.", Toast.LENGTH_LONG).show()
                    showScreen(screenAssessment, navLblAssess)
                }
                return@execute
            }

            val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
            val combinedText = StringBuilder()
            var processedCount = 0

            mainHandler.post {
                Toast.makeText(this, "Processing PDF report pages on-device...", Toast.LENGTH_SHORT).show()
            }

            for (pageBitmap in pages) {
                val preprocessed = OcrImagePreprocessor.preprocessBitmap(pageBitmap)
                val image = InputImage.fromBitmap(preprocessed, 0)
                recognizer.process(image)
                    .addOnSuccessListener { visionText ->
                        combinedText.append("\n").append(visionText.text)
                        processedCount++
                        if (processedCount == pages.size) {
                            val parsed = OcrReportParser.parseText(combinedText.toString())
                            populateAssessmentFromOcr(parsed)
                            showScreen(screenOcrReview, navLblAssess)
                        }
                    }
                    .addOnFailureListener {
                        processedCount++
                        if (processedCount == pages.size) {
                            val parsed = OcrReportParser.parseText(combinedText.toString())
                            populateAssessmentFromOcr(parsed)
                            showScreen(screenOcrReview, navLblAssess)
                        }
                    }
            }
        }
    }

    private fun populateAssessmentFromOcr(parsed: OcrExtractedPatient) {
        parsed.age?.let { etAge.setText(String.format(Locale.US, "%.1f", it)) }
        parsed.height?.let { etHeight.setText(String.format(Locale.US, "%.1f", it)) }
        parsed.weight?.let { etWeight.setText(String.format(Locale.US, "%.1f", it)) }
        parsed.apHi?.let { etApHi.setText(it.toString()) }
        parsed.apLo?.let { etApLo.setText(it.toString()) }

        parsed.gender?.let { if (it in 1..2) spinnerGender.setSelection(it - 1) }
        parsed.cholesterol?.let { if (it in 1..3) spinnerCholesterol.setSelection(it - 1) }
        parsed.glucose?.let { if (it in 1..3) spinnerGlucose.setSelection(it - 1) }
        parsed.smoke?.let { if (it in 0..1) spinnerSmoke.setSelection(it) }
        parsed.alco?.let { if (it in 0..1) spinnerAlco.setSelection(it) }
        parsed.active?.let { if (it in 0..1) spinnerActive.setSelection(it) }

        val alertText = StringBuilder("OCR Extraction Complete — Your report is processed on this device.")
        if (parsed.warningMsg != null) {
            alertText.append("\n\n⚠️ ").append(parsed.warningMsg)
        }
        tvOcrStatusAlert.text = alertText.toString()
    }

    private fun handleAnalyzePatient() {
        val age = etAge.text.toString().toDoubleOrNull() ?: 55.5
        val height = etHeight.text.toString().toDoubleOrNull() ?: 165.0
        val weight = etWeight.text.toString().toDoubleOrNull() ?: 70.0
        val apHi = etApHi.text.toString().toIntOrNull() ?: 140
        val apLo = etApLo.text.toString().toIntOrNull() ?: 90

        if (apLo >= apHi) {
            Toast.makeText(this, "Systolic BP must be strictly greater than Diastolic BP", Toast.LENGTH_SHORT).show()
            return
        }

        val gender = spinnerGender.selectedItemPosition + 1
        val cholesterol = spinnerCholesterol.selectedItemPosition + 1
        val glucose = spinnerGlucose.selectedItemPosition + 1
        val smoke = spinnerSmoke.selectedItemPosition
        val alco = spinnerAlco.selectedItemPosition
        val active = spinnerActive.selectedItemPosition

        val isOnlineMode = spinnerMode.selectedItemPosition == 1

        if (isOnlineMode) {
            executeOnlineInference(age, gender, height, weight, apHi, apLo, cholesterol, glucose, smoke, alco, active)
        } else {
            executeOfflineInference(age, gender, height, weight, apHi, apLo, cholesterol, glucose, smoke, alco, active)
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
