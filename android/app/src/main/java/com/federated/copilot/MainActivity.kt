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

    private fun setupSpinners() {
        val modeOptions = arrayOf("Offline Mode (Local)", "Online Mode (FastAPI)")
        val modeAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, modeOptions)
        spinnerMode.adapter = modeAdapter

        spinnerMode.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (position == 1) {
                    tvHeaderStatus.text = "Online Mode"
                    tvHeaderStatus.setBackgroundColor(Color.parseColor("#10B981"))
                } else {
                    tvHeaderStatus.text = "Offline Mode"
                    tvHeaderStatus.setBackgroundColor(Color.parseColor("#2563EB"))
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        spinnerGender.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, arrayOf("Female", "Male"))
        spinnerCholesterol.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, arrayOf("Normal", "Above Normal", "Well Above Normal"))
        spinnerGlucose.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, arrayOf("Normal", "Above Normal", "Well Above Normal"))
        spinnerSmoke.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, arrayOf("Non-Smoker", "Smoker"))
        spinnerAlco.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, arrayOf("Non-Drinker", "Drinker"))
        spinnerActive.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, arrayOf("Physically Inactive", "Physically Active"))

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

            renderResult(predClass, prob, apHi, apLo, height, weight, cholesterol, active, smoke)

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
                // Try Android Emulator host mapping first, fallback to localhost
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
                        renderResult(predClass, prob, apHi, apLo, height, weight, cholesterol, active, smoke)
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
        cholesterol: Int, active: Int, smoke: Int
    ) {
        if (predClass == 1) {
            tvRiskTag.text = "ELEVATED RISK"
            tvRiskTag.setTextColor(Color.parseColor("#991B1B"))
            tvRiskTag.setBackgroundColor(Color.parseColor("#FEE2E2"))
            cardRiskPrimary.setBackgroundColor(Color.parseColor("#FEF2F2"))
        } else {
            tvRiskTag.text = "LOWER RISK"
            tvRiskTag.setTextColor(Color.parseColor("#065F46"))
            tvRiskTag.setBackgroundColor(Color.parseColor("#D1FAE5"))
            cardRiskPrimary.setBackgroundColor(Color.parseColor("#ECFDF5"))
        }

        val pct = String.format("%.2f%%", prob * 100)
        tvProbability.text = pct

        tvResBp.text = "$apHi/$apLo mmHg"
        
        val heightM = height / 100.0
        val bmi = if (heightM > 0) String.format("%.1f kg/m²", weight / (heightM * heightM)) else "N/A"
        tvResBmi.text = bmi

        val cholMap = arrayOf("Normal", "Above Normal", "Well Above Normal")
        tvResCholesterol.text = cholMap.getOrElse(cholesterol - 1) { "Normal" }

        tvResActivity.text = if (active == 1) "Active" else "Inactive"
        tvResSmoking.text = if (smoke == 1) "Smoker" else "Non-Smoker"

        showScreen(screenResult)
    }
}
