package com.federated.copilot

import android.util.Log
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

data class OcrExtractedPatient(
    val age: Double? = null,
    val gender: Int? = null, // 1 = Female, 2 = Male
    val height: Double? = null,
    val weight: Double? = null,
    val apHi: Int? = null,
    val apLo: Int? = null,
    val cholesterol: Int? = null, // 1 = Normal (<200), 2 = Above Normal (200-239), 3 = Well Above Normal (>=240)
    val glucose: Int? = null, // 1 = Normal (<100), 2 = Above Normal (100-125), 3 = Well Above Normal (>=126)
    val smoke: Int? = null, // 0 = Non-smoker, 1 = Active smoker
    val alco: Int? = null, // 0 = Non-drinker, 1 = Regular drinker
    val active: Int? = null, // 0 = Inactive, 1 = Active
    val warningMsg: String? = null,
    val ambiguousFields: List<String> = emptyList()
) {
    fun fieldsFoundCount(): Int {
        var count = 0
        if (age != null) count++
        if (gender != null) count++
        if (height != null) count++
        if (weight != null) count++
        if (apHi != null && apLo != null) count++
        if (cholesterol != null) count++
        if (glucose != null) count++
        if (smoke != null) count++
        if (alco != null) count++
        if (active != null) count++
        return count
    }
}

data class OcrResult(
    val success: Boolean,
    val rawText: String,
    val charCount: Int,
    val parsedData: OcrExtractedPatient,
    val fieldsFoundCount: Int,
    val statusMessage: String,
    val errorMsg: String? = null
)

object OcrReportParser {

    private const val TAG = "CardioSense_OCR"

    private fun safeLog(msg: String) {
        try {
            Log.d(TAG, msg)
        } catch (_: Throwable) {
            println("[$TAG] $msg")
        }
    }

    fun parseText(rawText: String): OcrResult {
        safeLog("==================== RAW OCR TEXT ====================")
        safeLog(rawText)
        safeLog("======================================================")


        if (rawText.isBlank()) {
            return OcrResult(
                success = true,
                rawText = "",
                charCount = 0,
                parsedData = OcrExtractedPatient(),
                fieldsFoundCount = 0,
                statusMessage = "OCR Succeeded (0 characters detected). Document appears empty or unreadable.",
                errorMsg = "No text detected by OCR"
            )
        }

        val normalizedText = normalizeOcrText(rawText)
        val lines = normalizedText.split("\n").map { it.trim() }.filter { it.isNotEmpty() }
        val warnings = mutableListOf<String>()
        val ambiguousFields = mutableListOf<String>()

        var age: Double? = null
        var calculatedAgeFromDob: Double? = null
        var gender: Int? = null
        var height: Double? = null
        var weight: Double? = null
        var apHi: Int? = null
        var apLo: Int? = null
        var cholesterol: Int? = null
        var glucose: Int? = null
        var smoke: Int? = null
        var alco: Int? = null
        var active: Int? = null

        // --------------------------------------------------------------------
        // PASS 1: INLINE & KEY-VALUE PATTERN MATCHING
        // --------------------------------------------------------------------

        // 1. Blood Pressure
        // Format: 145/92, 145 / 92, 145-92, 145 \ 92
        val fullBpMatch = Regex("""(?:bp|b\.p\.|blood\s*pressure|sbp\/dbp|pressure|reading|vitals?)[\s:=\n-]*(\d{2,3})\s*[\/\\:-]\s*(\d{2,3})\b""").find(normalizedText)
        if (fullBpMatch != null) {
            val hi = fullBpMatch.groupValues[1].toIntOrNull()
            val lo = fullBpMatch.groupValues[2].toIntOrNull()
            if (hi != null && hi in 60..260 && lo != null && lo in 40..180 && hi > lo) {
                apHi = hi
                apLo = lo
            }
        }
        if (apHi == null) {
            val sbpMatch = Regex("""(?:systolic\s*bp(?:\s*\(sbp\))?|systolic\s*blood\s*pressure|sbp|sys\s*bp|systolic)[\s:=\n-]*(\d{2,3})\b""").find(normalizedText)
            if (sbpMatch != null) {
                val hi = sbpMatch.groupValues[1].toIntOrNull()
                if (hi != null && hi in 60..260) apHi = hi
            }
        }
        if (apLo == null) {
            val dbpMatch = Regex("""(?:diastolic\s*bp(?:\s*\(dbp\))?|diastolic\s*blood\s*pressure|dbp|dia\s*bp|diastolic)[\s:=\n-]*(\d{2,3})\b""").find(normalizedText)
            if (dbpMatch != null) {
                val lo = dbpMatch.groupValues[1].toIntOrNull()
                if (lo != null && lo in 40..180) apLo = lo
            }
        }

        // 2. Glucose
        // Numeric (mg/dL) or Explicit level/categorical (Level 1..3, Normal, Above Normal, Well Above Normal)
        val glucCatMatch = Regex("""(?:fasting\s*blood\s*glucose|fasting\s*glucose|fasting\s*blood\s*sugar|fasting\s*sugar|random\s*blood\s*sugar|random\s*sugar|postprandial\s*blood\s*sugar|postprandial\s*sugar|blood\s*glucose|blood\s*sugar|serum\s*glucose|fbs|fbg|rbs|ppbs|glucose|sugar)[\s:=\n-]*(?:\([a-z\/\s]*\))?[\s:=\n-]*(well\s*above\s*normal|above\s*normal|normal|level\s*[123])\b""").find(normalizedText)
        if (glucCatMatch != null) {
            val catStr = glucCatMatch.groupValues[1]
            glucose = when {
                catStr.contains("well above") || catStr.contains("3") -> 3
                catStr.contains("above") || catStr.contains("2") -> 2
                catStr.contains("normal") || catStr.contains("1") -> 1
                else -> null
            }
        } else {
            val fullGlucMatch = Regex("""(?:fasting\s*blood\s*glucose|fasting\s*glucose|fasting\s*blood\s*sugar|fasting\s*sugar|random\s*blood\s*sugar|random\s*sugar|postprandial\s*blood\s*sugar|postprandial\s*sugar|blood\s*glucose|blood\s*sugar|serum\s*glucose|fbs|fbg|rbs|ppbs|glucose|sugar)[\s:=\n-]*(?:\([a-z\/\s]*\))?[\s:=\n-]*(?:level\s*)?(\d{1,3})\b""").find(normalizedText)
            if (fullGlucMatch != null) {
                val valNum = fullGlucMatch.groupValues[1].toIntOrNull() ?: 0
                glucose = when {
                    valNum in 1..3 -> valNum
                    valNum >= 126 -> 3
                    valNum >= 100 -> 2
                    valNum > 0 -> 1
                    else -> null
                }
            }
        }


        // 3. Cholesterol
        val cholCatMatch = Regex("""(?:total\s*cholesterol|serum\s*cholesterol|s\.\s*cholesterol|total\s*chol|cholesterol|tc)[\s:=\n-]*(?:\([a-z\/\s]*\))?[\s:=\n-]*(well\s*above\s*normal|above\s*normal|normal|level\s*[123])\b""").find(normalizedText)
        if (cholCatMatch != null) {
            val catStr = cholCatMatch.groupValues[1]
            cholesterol = when {
                catStr.contains("well above") || catStr.contains("3") -> 3
                catStr.contains("above") || catStr.contains("2") -> 2
                catStr.contains("normal") || catStr.contains("1") -> 1
                else -> null
            }
        } else {
            val fullCholMatch = Regex("""(?:total\s*cholesterol|serum\s*cholesterol|s\.\s*cholesterol|total\s*chol|cholesterol|tc)[\s:=\n-]*(?:\([a-z\/\s]*\))?[\s:=\n-]*(?:level\s*)?(\d{1,3})\b""").find(normalizedText)
            if (fullCholMatch != null) {
                val valNum = fullCholMatch.groupValues[1].toIntOrNull() ?: 0
                cholesterol = when {
                    valNum in 1..3 -> valNum
                    valNum >= 240 -> 3
                    valNum >= 200 -> 2
                    valNum > 0 -> 1
                    else -> null
                }
            }
        }

        // 4. Height
        val fullHftMatch = Regex("""(?:height|ht|ht\.|stature)[\s:=\n-]*(?:\([a-z\/\s]*\))?[\s:=\n-]*(\d)\s*(?:ft|feet|')\s*(\d{1,2})?\s*(?:in|inches|")?""").find(normalizedText)
        if (fullHftMatch != null) {
            val feet = fullHftMatch.groupValues[1].toDoubleOrNull() ?: 0.0
            val inches = fullHftMatch.groupValues[2].toDoubleOrNull() ?: 0.0
            val totalCm = (feet * 30.48) + (inches * 2.54)
            if (totalCm in 50.0..250.0) height = totalCm
        } else {
            val fullHmMatch = Regex("""(?:height|ht|ht\.|stature)[\s:=\n-]*(?:\([a-z\/\s]*\))?[\s:=\n-]*(\d(?:\.\d{1,2})?)\s*m\b""").find(normalizedText)
            if (fullHmMatch != null) {
                val mVal = fullHmMatch.groupValues[1].toDoubleOrNull()
                if (mVal != null && mVal in 0.8..2.5) height = mVal * 100.0
            } else {
                val fullHcmMatch = Regex("""(?:height|ht|ht\.|stature)[\s:=\n-]*(?:\([a-z\/\s]*\))?[\s:=\n-]*(\d{2,3}(?:\.\d)?)\s*(?:cm)?\b""").find(normalizedText)
                if (fullHcmMatch != null) {
                    val valNum = fullHcmMatch.groupValues[1].toDoubleOrNull()
                    if (valNum != null && valNum in 50.0..250.0) height = valNum
                }
            }
        }

        // 5. Weight
        val fullWlbsMatch = Regex("""(?:weight|wt|wt\.|body\s*weight)[\s:=\n-]*(?:\([a-z\/\s]*\))?[\s:=\n-]*(\d{2,3}(?:\.\d)?)\s*(?:lbs|lb|pounds)\b""").find(normalizedText)
        if (fullWlbsMatch != null) {
            val lbsVal = fullWlbsMatch.groupValues[1].toDoubleOrNull()
            if (lbsVal != null) weight = lbsVal * 0.45359237
        } else {
            val fullWkgMatch = Regex("""(?:weight|wt|wt\.|body\s*weight)[\s:=\n-]*(?:\([a-z\/\s]*\))?[\s:=\n-]*(\d{2,3}(?:\.\d)?)\s*(?:kg|kilograms)?\b""").find(normalizedText)
            if (fullWkgMatch != null) {
                val valNum = fullWkgMatch.groupValues[1].toDoubleOrNull()
                if (valNum != null && valNum in 20.0..300.0) weight = valNum
            }
        }

        // 6. Direct Age
        val fullAgeMatch = Regex("""\b(?:patient\s*age|age)[\s:=\n-]*(?:\([a-z\/\s]*\))?[\s:=\n-]*(\d{1,3}(?:\.\d)?)\s*(?:yrs|years|y)?\b""").find(normalizedText)
        if (fullAgeMatch != null) {
            val ageVal = fullAgeMatch.groupValues[1].toDoubleOrNull()
            if (ageVal != null && ageVal in 1.0..120.0) age = ageVal
        }

        // Shorthand Age/Gender (e.g. 34/F, 34yo female, 45y male)
        if (age == null || gender == null) {
            val shorthandMatch = Regex("""\b(\d{1,2})\s*(?:yo|y\/o|years?\s*old|y)?\s*[\/,-]?\s*(female|male|f|m)\b""").find(normalizedText)
            if (shorthandMatch != null) {
                if (age == null) {
                    val aVal = shorthandMatch.groupValues[1].toDoubleOrNull()
                    if (aVal != null && aVal in 1.0..120.0) age = aVal
                }
                if (gender == null) {
                    val gStr = shorthandMatch.groupValues[2]
                    gender = if (gStr == "female" || gStr == "f") 1 else 2
                }
            }
        }

        // 7. Gender
        if (gender == null) {
            val fullSexMatch = Regex("""(?:gender|sex|biological\s*sex)[\s:=\n-]*(female|male|f|m)\b""").find(normalizedText)
            if (fullSexMatch != null) {
                val sexStr = fullSexMatch.groupValues[1]
                gender = if (sexStr == "female" || sexStr == "f") 1 else 2
            }
        }

        // 8. Smoking
        val fullSmokeMatch = Regex("""(?:smoking\s*status|smoking\s*history|smoking\s*habits|smoking|smoker|tobacco\s*use|tobacco|cigarette[s]?|nicotine)[\s:=\n-]*(?:\([a-z\/\s]*\))?[\s:=\n-]*(never\s*smoker|non-smoker|nonsmoker|does\s*not\s*smoke|denies\s*smoking|no\s*smoking|former\s*smoker|ex-smoker|past\s*smoker|quit\s*smoking|current\s*smoker|current\s*smoking|smokes|tobacco\s*user|active\s*smoker|active|yes|no|none|nil)\b""").find(normalizedText)

        if (fullSmokeMatch != null) {
            val sStr = fullSmokeMatch.groupValues[1]
            when {
                sStr.contains("former") || sStr.contains("ex-") || sStr.contains("past") || sStr.contains("quit") -> {
                    warnings.add("Smoking status is ambiguous ('$sStr') — please verify.")
                    ambiguousFields.add("Smoking Status")
                    smoke = null
                }
                sStr.contains("current") || sStr.contains("smokes") || sStr.contains("active") || sStr.contains("yes") -> {
                    smoke = 1
                }
                sStr.contains("non") || sStr.contains("never") || sStr.contains("does not") || sStr.contains("denies") || sStr.contains("no") || sStr.contains("none") || sStr.contains("nil") -> {
                    smoke = 0
                }
            }
        }

        // 9. Alcohol
        val fullAlcoMatch = Regex("""(?:alcohol\s*use|alcohol\s*intake|alcohol\s*consumption|alcohol|drinking)[\s:=\n-]*(?:\([a-z\/\s]*\))?[\s:=\n-]*(non-drinker|does\s*not\s*drink|denies\s*alcohol|no\s*alcohol|never|nil|none|occasionally|social\s*drinking|social|rarely|rare|former|current\s*alcohol\s*use|regular\s*drinker|regular|drinks|yes|no)\b""").find(normalizedText)
        if (fullAlcoMatch != null) {
            val aStr = fullAlcoMatch.groupValues[1]
            when {
                aStr.contains("occasion") || aStr.contains("social") || aStr.contains("rare") || aStr.contains("former") -> {
                    warnings.add("Alcohol intake is ambiguous ('$aStr') — please verify.")
                    ambiguousFields.add("Alcohol Intake")
                    alco = null
                }
                aStr.contains("current") || aStr.contains("regular") || aStr.contains("drinks") || aStr.contains("yes") -> {
                    alco = 1
                }
                aStr.contains("non") || aStr.contains("never") || aStr.contains("does not") || aStr.contains("denies") || aStr.contains("no") || aStr.contains("none") || aStr.contains("nil") -> {
                    alco = 0
                }
            }
        }

        // 10. Physical Activity
        val fullActiveMatch = Regex("""(?:physical\s*activity|physically\s*active|exercise\s*level|exercise|activity\s*level|daily\s*exercise|sedentary|lifestyle|active)[\s:=\n-]*(?:\([a-z\/\s]*\))?[\s:=\n-]*(physically\s*active|regular\s*exercise|exercises\s*regularly|active|sedentary|inactive|no\s*regular\s*exercise|less\s*than\s*30\s*minutes|yes|no)\b""").find(normalizedText)
        if (fullActiveMatch != null) {
            val actStr = fullActiveMatch.groupValues[1]
            when {
                actStr.contains("active") || actStr.contains("regular") || actStr.contains("yes") -> {
                    active = 1
                }
                actStr.contains("sedentary") || actStr.contains("inactive") || actStr.contains("no") || actStr.contains("less than") -> {
                    active = 0
                }
            }
        }

        // --------------------------------------------------------------------
        // PASS 2: DOB AGE CALCULATION
        // --------------------------------------------------------------------
        for (line in lines) {
            val isReferenceLine = line.contains("reference range") || line.contains("ref. interval") || line.contains("ref range")

            if (calculatedAgeFromDob == null && !isReferenceLine) {
                val isNonPatientDate = line.contains("report date") || line.contains("collection date") ||
                        line.contains("admission date") || line.contains("registration date") ||
                        line.contains("specimen date") || line.contains("date of printing") || line.contains("printed")

                if (!isNonPatientDate) {
                    val dobMatch = Regex("""(?:dob|date\s*of\s*birth|birth\s*date|d\.o\.b\.|born)[\s:=\n-]*(\d{1,4}[\/\.-]\d{1,2}[\/\.-]\d{1,4}|\d{1,2}\s+[a-z]{3,9}\s+\d{4}|[a-z]{3,9}\s+\d{1,2},?\s+\d{4})""").find(line)
                    if (dobMatch != null) {
                        val dobStr = dobMatch.groupValues[1]
                        calculatedAgeFromDob = calculateAgeFromDobString(dobStr)
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // PASS 3: DISCONNECTED / 2-COLUMN TABLE SEARCH & GLOBAL FALLBACK
        // --------------------------------------------------------------------

        // 1. Blood Pressure search
        if (apHi == null || apLo == null) {
            val globalBp = Regex("""\b(\d{2,3})\s*[\/\\:-]\s*(\d{2,3})\s*(?:mmhg)?\b""").find(normalizedText)
            if (globalBp != null) {
                val hi = globalBp.groupValues[1].toIntOrNull()
                val lo = globalBp.groupValues[2].toIntOrNull()
                if (hi != null && hi in 60..260 && lo != null && lo in 40..180 && hi > lo) {
                    apHi = hi
                    apLo = lo
                }
            }
        }

        // 2. Height search
        if (height == null) {
            val globalH = Regex("""\b(\d{2,3}(?:\.\d)?)\s*cm\b""").find(normalizedText)
            if (globalH != null) {
                val valNum = globalH.groupValues[1].toDoubleOrNull()
                if (valNum != null && valNum in 50.0..250.0) height = valNum
            }
        }

        // 3. Weight search
        if (weight == null) {
            val globalW = Regex("""\b(\d{2,3}(?:\.\d)?)\s*kg\b""").find(normalizedText)
            if (globalW != null) {
                val valNum = globalW.groupValues[1].toDoubleOrNull()
                if (valNum != null && valNum in 20.0..300.0) weight = valNum
            }
        }

        // 4. Age search
        if (age == null) {
            val globalAge = Regex("""\b(\d{1,3}(?:\.\d)?)\s*(?:years|yrs)\b""").find(normalizedText)
            if (globalAge != null) {
                val valNum = globalAge.groupValues[1].toDoubleOrNull()
                if (valNum != null && valNum in 1.0..120.0) age = valNum
            }
        }

        // 5. Gender search
        if (gender == null) {
            val globalSex = Regex("""\b(female|male)\b""").find(normalizedText)
            if (globalSex != null) {
                val sexStr = globalSex.groupValues[1]
                gender = if (sexStr == "female") 1 else 2
            }
        }

        // 6. Cholesterol Level search
        if (cholesterol == null) {
            val cholLevelMatch = Regex("""cholesterol[\s\S]{0,150}?\blevel\s*([123])\b""").find(normalizedText)
            if (cholLevelMatch != null) {
                cholesterol = cholLevelMatch.groupValues[1].toIntOrNull()
            } else {
                val cholNumMatch = Regex("""cholesterol[\s\S]{0,150}?\b(\d{2,3})\s*(?:mg\/dl)?\b""").find(normalizedText)
                if (cholNumMatch != null) {
                    val valNum = cholNumMatch.groupValues[1].toIntOrNull() ?: 0
                    cholesterol = when {
                        valNum >= 240 -> 3
                        valNum >= 200 -> 2
                        valNum > 0 -> 1
                        else -> null
                    }
                }
            }
        }

        // 7. Glucose Level search
        if (glucose == null) {
            val glucLevelMatch = Regex("""glucose[\s\S]{0,150}?\blevel\s*([123])\b""").find(normalizedText)
            if (glucLevelMatch != null) {
                glucose = glucLevelMatch.groupValues[1].toIntOrNull()
            } else {
                val glucNumMatch = Regex("""glucose[\s\S]{0,150}?\b(\d{2,3})\s*(?:mg\/dl)?\b""").find(normalizedText)
                if (glucNumMatch != null) {
                    val valNum = glucNumMatch.groupValues[1].toIntOrNull() ?: 0
                    glucose = when {
                        valNum >= 126 -> 3
                        valNum >= 100 -> 2
                        valNum > 0 -> 1
                        else -> null
                    }
                }
            }
        }

        // 8. Lifestyle search
        if (smoke == null && !ambiguousFields.contains("Smoking Status")) {
            val smokeMatch = Regex("""smoking[\s\S]{0,120}?\b(yes|no|active|never|non-smoker|nonsmoker|nil|none)\b""").find(normalizedText)
            if (smokeMatch != null) {
                val sStr = smokeMatch.groupValues[1]
                smoke = if (sStr.contains("yes") || sStr.contains("active")) 1 else 0
            }
        }

        if (alco == null && !ambiguousFields.contains("Alcohol Intake")) {
            val alcoMatch = Regex("""alcohol[\s\S]{0,120}?\b(yes|no|regular|never|non-drinker|nil|none)\b""").find(normalizedText)
            if (alcoMatch != null) {
                val aStr = alcoMatch.groupValues[1]
                alco = if (aStr.contains("yes") || aStr.contains("regular")) 1 else 0
            }
        }

        if (active == null) {
            val activeMatch = Regex("""(?:physical\s*activity|exercise)[\s\S]{0,120}?\b(yes|no|active|regular|sedentary|inactive)\b""").find(normalizedText)
            if (activeMatch != null) {
                val actStr = activeMatch.groupValues[1]
                active = if (actStr.contains("yes") || actStr.contains("active") || actStr.contains("regular")) 1 else 0
            }
        }

        // Pass 3B: Disconnected 2-Column Table Resolution
        val discRes = parseDisconnectedTableLifestyle(normalizedText, smoke, alco, active, cholesterol, glucose, ambiguousFields)
        smoke = discRes.smoke
        alco = discRes.alco
        active = discRes.active
        cholesterol = discRes.chol
        glucose = discRes.gluc



        // Apply Age from DOB if direct Age is missing
        if (age == null && calculatedAgeFromDob != null) {
            age = calculatedAgeFromDob
        } else if (age != null && calculatedAgeFromDob != null) {
            if (Math.abs(age - calculatedAgeFromDob) > 1.5) {
                warnings.add("Age ($age yrs) and DOB ($calculatedAgeFromDob yrs) appear inconsistent — please verify.")
            }
        }

        val combinedWarning = if (warnings.isNotEmpty()) warnings.distinct().joinToString("\n") else null

        val parsedData = OcrExtractedPatient(
            age = age,
            gender = gender,
            height = height,
            weight = weight,
            apHi = apHi,
            apLo = apLo,
            cholesterol = cholesterol,
            glucose = glucose,
            smoke = smoke,
            alco = alco,
            active = active,
            warningMsg = combinedWarning,
            ambiguousFields = ambiguousFields.distinct()
        )

        val foundCount = parsedData.fieldsFoundCount()
        val charLen = rawText.length

        val statusMsg = when {
            foundCount == 0 -> "OCR Succeeded ($charLen chars detected), but 0 medical fields matched."
            else -> "OCR Succeeded ($charLen chars detected) — $foundCount/10 medical fields auto-populated."
        }

        safeLog("==================== PARSED PATIENT DATA ====================")
        safeLog("age=${parsedData.age}, gender=${parsedData.gender}, height=${parsedData.height}, weight=${parsedData.weight}")
        safeLog("apHi=${parsedData.apHi}, apLo=${parsedData.apLo}, cholesterol=${parsedData.cholesterol}, glucose=${parsedData.glucose}")
        safeLog("smoke=${parsedData.smoke}, alco=${parsedData.alco}, active=${parsedData.active}")
        safeLog("fieldsFoundCount=$foundCount/10")
        safeLog("=============================================================")


        return OcrResult(
            success = true,
            rawText = rawText,
            charCount = charLen,
            parsedData = parsedData,
            fieldsFoundCount = foundCount,
            statusMessage = statusMsg
        )
    }

    private fun normalizeOcrText(text: String): String {
        return text.lowercase(Locale.US)
            .replace("syst0lic", "systolic")
            .replace("gluc0se", "glucose")
            .replace("cholesteroi", "cholesterol")
            .replace("sm0king", "smoking")
            .replace("alc0hol", "alcohol")
            .replace("physicai", "physical")
            .replace("diast0lic", "diastolic")
    }

    private fun calculateAgeFromDobString(dobStr: String): Double? {
        val formats = arrayOf(
            SimpleDateFormat("dd/MM/yyyy", Locale.US),
            SimpleDateFormat("dd-MM-yyyy", Locale.US),
            SimpleDateFormat("dd.MM.yyyy", Locale.US),
            SimpleDateFormat("yyyy-MM-dd", Locale.US),
            SimpleDateFormat("yyyy/MM/dd", Locale.US),
            SimpleDateFormat("MM/dd/yyyy", Locale.US),
            SimpleDateFormat("dd MMMM yyyy", Locale.US),
            SimpleDateFormat("dd MMM yyyy", Locale.US),
            SimpleDateFormat("MMMM dd, yyyy", Locale.US),
            SimpleDateFormat("MMM dd, yyyy", Locale.US)
        )

        for (fmt in formats) {
            fmt.isLenient = false
            try {
                val birthDate: Date? = fmt.parse(dobStr)
                if (birthDate != null) {
                    val dobCal = Calendar.getInstance().apply { time = birthDate }
                    val nowCal = Calendar.getInstance()
                    var ageYears = nowCal.get(Calendar.YEAR) - dobCal.get(Calendar.YEAR)
                    if (nowCal.get(Calendar.DAY_OF_YEAR) < dobCal.get(Calendar.DAY_OF_YEAR)) {
                        ageYears--
                    }
                    if (ageYears in 1..120) {
                        return ageYears.toDouble()
                    }
                }
            } catch (_: Exception) {}
        }
        return null
    }

    private fun parseDisconnectedTableLifestyle(
        text: String,
        currentSmoke: Int?,
        currentAlco: Int?,
        currentActive: Int?,
        currentChol: Int?,
        currentGluc: Int?,
        ambiguousFields: List<String>
    ): DisconnectedTableResult {
        var smoke = currentSmoke
        var alco = currentAlco
        var active = currentActive
        var chol = currentChol
        var gluc = currentGluc

        val lines = text.split("\n").map { it.trim() }.filter { it.isNotEmpty() }

        // 2-Column Table Categorical Levels (Cholesterol & Glucose)
        val cholLineIdx = lines.indexOfFirst { it.contains("cholesterol") }
        val glucLineIdx = lines.indexOfFirst { it.contains("glucose") || it.contains("sugar") }

        if (cholLineIdx != -1 && glucLineIdx != -1 && cholLineIdx < glucLineIdx) {
            val levelValues = mutableListOf<Int>()
            val levelRegex = Regex("""^level\s*([123])$""", RegexOption.IGNORE_CASE)
            for (i in glucLineIdx + 1 until lines.size) {
                val l = lines[i].lowercase(Locale.US)
                val m = levelRegex.find(l)
                if (m != null) {
                    val lvl = m.groupValues[1].toIntOrNull()
                    if (lvl != null) levelValues.add(lvl)
                }
            }
            if (levelValues.size >= 2) {
                chol = levelValues[0]
                gluc = levelValues[1]
            }
        }

        // 2-Column Table Lifestyle Fields (Smoking, Alcohol, Physical Activity)
        val smokeLineIdx = lines.indexOfFirst { it.contains("smoking") || it.contains("smoker") || it.contains("tobacco") }
        val alcoLineIdx = lines.indexOfFirst { it.contains("alcohol") || it.contains("drinking") }
        val activeLineIdx = lines.indexOfFirst { it.contains("physical activity") || it.contains("exercise") || it.contains("lifestyle") }

        if (smokeLineIdx != -1 && alcoLineIdx != -1 && activeLineIdx != -1 &&
            smokeLineIdx < alcoLineIdx && alcoLineIdx < activeLineIdx) {

            val candidateValues = mutableListOf<String>()
            val lifestyleValueRegex = Regex("""^(yes|no|active|inactive|sedentary|regular|never|non-smoker|non-drinker|nil|none)$""", RegexOption.IGNORE_CASE)

            for (i in activeLineIdx + 1 until lines.size) {
                val l = lines[i].lowercase(Locale.US)
                if (lifestyleValueRegex.matches(l) || l == "yes" || l == "no" || l.contains("smoker") || l.contains("drinker") || l.contains("exercise")) {
                    candidateValues.add(l)
                }
            }

            if (candidateValues.size >= 3) {
                val smokeVal = candidateValues[candidateValues.size - 3]
                val alcoVal = candidateValues[candidateValues.size - 2]
                val activeVal = candidateValues[candidateValues.size - 1]

                if (!ambiguousFields.contains("Smoking Status")) {
                    smoke = if (smokeVal.contains("yes") || smokeVal.contains("active") || smokeVal.contains("smoker")) 1 else 0
                }
                if (!ambiguousFields.contains("Alcohol Intake")) {
                    alco = if (alcoVal.contains("yes") || alcoVal.contains("regular") || alcoVal.contains("drinker")) 1 else 0
                }
                active = if (activeVal.contains("yes") || activeVal.contains("active") || activeVal.contains("regular")) 1 else 0

            }
        }

        return DisconnectedTableResult(smoke, alco, active, chol, gluc)
    }
}

private data class DisconnectedTableResult(
    val smoke: Int?,
    val alco: Int?,
    val active: Int?,
    val chol: Int?,
    val gluc: Int?
)


