package com.federated.copilot

import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONObject
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import kotlin.math.exp
import kotlin.math.max

class MainActivity : AppCompatActivity() {

    // Views
    private lateinit var screenHome: View
    private lateinit var screenAssessment: View
    private lateinit var screenResult: View

    private lateinit var spinnerMode: Spinner
    private lateinit var tvHeaderStatus: TextView
    private lateinit var tvModeDescription: TextView

    // Input Views
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
    private lateinit var tvClinicalReasoning: TextView
    private lateinit var tvResBp: TextView
    private lateinit var tvResBmi: TextView
    private lateinit var tvResCholesterol: TextView
    private lateinit var tvResActivity: TextView
    private lateinit var tvResSmoking: TextView

    // Offline Assets
    private var weightsJson: JSONObject? = null
    private var scalerJson: JSONObject? = null
    private val executor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        setupSpinners()
        loadLocalAssetsAsync()
    }

    private fun initViews() {
        screenHome = findViewById(R.id.screen_home)
        screenAssessment = findViewById(R.id.screen_assessment)
        screenResult = findViewById(R.id.screen_result)

        tvHeaderStatus = findViewById(R.id.tv_header_status)
        tvModeDescription = findViewById(R.id.tv_mode_description)
        spinnerMode = findViewById(R.id.spinner_inference_mode)

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
        tvClinicalReasoning = findViewById(R.id.tv_clinical_reasoning)

        tvResBp = findViewById(R.id.tv_res_bp)
        tvResBmi = findViewById(R.id.tv_res_bmi)
        tvResCholesterol = findViewById(R.id.tv_res_cholesterol)
        tvResActivity = findViewById(R.id.tv_res_activity)
        tvResSmoking = findViewById(R.id.tv_res_smoking)

        findViewById<Button>(R.id.btn_goto_assessment).setOnClickListener {
            showScreen(screenAssessment)
        }

        findViewById<TextView>(R.id.btn_cancel_assessment).setOnClickListener {
            showScreen(screenHome)
        }

        findViewById<Button>(R.id.btn_analyze_patient).setOnClickListener {
            handleAnalyzePatient()
        }

        findViewById<Button>(R.id.btn_new_assessment).setOnClickListener {
            showScreen(screenAssessment)
        }
    }

    private fun createCustomAdapter(items: Array<String>): ArrayAdapter<String> {
        return ArrayAdapter(this, R.layout.spinner_item, items).apply {
            setDropDownViewResource(R.layout.spinner_dropdown_item)
        }
    }

    private fun setupSpinners() {
        val modeOptions = arrayOf("Offline Mode (Local)", "Online Mode (FastAPI)")
        spinnerMode.adapter = createCustomAdapter(modeOptions)

        spinnerMode.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (position == 1) {
                    tvHeaderStatus.text = "Online Mode"
                    tvHeaderStatus.setBackgroundColor(Color.parseColor("#10B981"))
                    tvModeDescription.text = "Connected to central FastAPI server for remote inference verification and federated parameter synchronization."
                } else {
                    tvHeaderStatus.text = "Offline Mode (Local)"
                    tvHeaderStatus.setBackgroundColor(Color.parseColor("#2563EB"))
                    tvModeDescription.text = "Local engine active. Standalone neural network computes risk predictions on-device with 0ms network latency."
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

        // Set default selections matching standard test sample
        spinnerCholesterol.setSelection(1) // Above Normal
        spinnerActive.setSelection(1) // Physically Active
    }

    private fun showScreen(target: View) {
        screenHome.visibility = View.GONE
        screenAssessment.visibility = View.GONE
        screenResult.visibility = View.GONE
        target.visibility = View.VISIBLE
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

            // Standardize continuous features (age, height, weight, ap_hi, ap_lo)
            val ageStd = (age - means.getDouble(0)) / scales.getDouble(0)
            val heightStd = (height - means.getDouble(1)) / scales.getDouble(1)
            val weightStd = (weight - means.getDouble(2)) / scales.getDouble(2)
            val apHiStd = (apHi - means.getDouble(3)) / scales.getDouble(3)
            val apLoStd = (apLo - means.getDouble(4)) / scales.getDouble(4)

            // Input array (11 features in canonical order)
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
            Toast.makeText(this, "Offline Engine Error: ${e.message}", Toast.LENGTH_LONG).show()
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
                        Toast.makeText(this@MainActivity, "Server Error ($code). Falling back to Offline Mode.", Toast.LENGTH_SHORT).show()
                        executeOfflineInference(age, gender, height, weight, apHi, apLo, cholesterol, glucose, smoke, alco, active)
                    }
                }
            } catch (e: Exception) {
                mainHandler.post {
                    Toast.makeText(this@MainActivity, "Online Server Unreachable. Using Offline Local Engine.", Toast.LENGTH_SHORT).show()
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
        if (predClass == 1) {
            tvRiskTag.text = "ELEVATED POTENTIAL RISK"
            tvRiskTag.setTextColor(Color.parseColor("#991B1B"))
            tvRiskTag.setBackgroundColor(Color.parseColor("#FEE2E2"))
            cardRiskPrimary.setBackgroundColor(Color.parseColor("#FEF2F2"))
        } else {
            tvRiskTag.text = "LOWER POTENTIAL RISK"
            tvRiskTag.setTextColor(Color.parseColor("#065F46"))
            tvRiskTag.setBackgroundColor(Color.parseColor("#D1FAE5"))
            cardRiskPrimary.setBackgroundColor(Color.parseColor("#ECFDF5"))
        }

        val pct = String.format("%.2f%%", prob * 100)
        tvProbability.text = pct

        tvResBp.text = "$apHi/$apLo mmHg"
        
        val heightM = height / 100.0
        val bmi = if (heightM > 0) weight / (heightM * heightM) else 22.0
        tvResBmi.text = String.format("%.1f kg/m²", bmi)

        val cholMap = arrayOf("Normal (< 200 mg/dL)", "Above Normal (200-239 mg/dL)", "Well Above Normal (≥ 240 mg/dL)")
        tvResCholesterol.text = cholMap.getOrElse(cholesterol - 1) { "Normal" }

        tvResActivity.text = if (active == 1) "Active (≥ 30 min/day)" else "Inactive"
        tvResSmoking.text = if (smoke == 1) "Active Smoker" else "Non-Smoker"

        // Build Contributing Factors & Clinical Reasoning
        val factors = mutableListOf<String>()
        val aspects = mutableListOf<String>()

        if (apHi >= 140 || apLo >= 90) {
            factors.add("• Elevated Blood Pressure: $apHi/$apLo mmHg (Hypertension range)")
            aspects.add("blood pressure")
        } else if (apHi >= 130 || apLo >= 80) {
            factors.add("• Prehypertension Blood Pressure: $apHi/$apLo mmHg")
            aspects.add("borderline blood pressure")
        }

        if (cholesterol >= 2) {
            factors.add("• Above-normal serum cholesterol (Level $cholesterol)")
            aspects.add("cholesterol level")
        }

        if (glucose >= 2) {
            factors.add("• Elevated fasting blood glucose (Level $glucose)")
            aspects.add("blood glucose")
        }

        if (active == 0) {
            factors.add("• Physical inactivity reported (< 30 min/day)")
            aspects.add("physical inactivity")
        }

        if (smoke == 1) {
            factors.add("• Active tobacco smoker status")
            aspects.add("smoking status")
        }

        if (bmi >= 25.0) {
            factors.add(String.format("• Elevated Body Mass Index: %.1f kg/m²", bmi))
            aspects.add("body mass index")
        }

        if (age >= 55.0) {
            factors.add(String.format("• Age demographic factor: %.1f years", age))
            aspects.add("age factor")
        }

        if (factors.isEmpty()) {
            factors.add("• Optimal baseline blood pressure reading")
            factors.add("• Normal lipid and glucose profile")
            factors.add("• Active lifestyle factors")
        }

        tvContributingFactors.text = factors.joinToString("\n")

        if (predClass == 1 && aspects.isNotEmpty()) {
            val aspectStr = aspects.distinct().joinToString(", ")
            tvClinicalReasoning.text = "The assessment is influenced primarily by the patient's $aspectStr. Decision support indicates statistical correlation with potential cardiovascular risk patterns derived from multicenter federated hospital cohorts."
        } else {
            tvClinicalReasoning.text = "The assessment indicates optimal alignment across primary vascular metrics (blood pressure, lipid profile, and activity levels), resulting in a lower potential cardiovascular risk estimation."
        }

        showScreen(screenResult)
    }
}
