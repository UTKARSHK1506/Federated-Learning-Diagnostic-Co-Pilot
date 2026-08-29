package com.federated.copilot

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
    val cholesterol: Int? = null, // 1 = Normal, 2 = Above Normal, 3 = Well Above Normal
    val glucose: Int? = null, // 1 = Normal, 2 = Above Normal, 3 = Well Above Normal
    val smoke: Int? = null, // 0 = Non-smoker, 1 = Active smoker
    val alco: Int? = null, // 0 = Non-drinker, 1 = Regular drinker
    val active: Int? = null, // 0 = Inactive, 1 = Active
    val warningMsg: String? = null
)

object OcrReportParser {

    fun parseText(rawText: String): OcrExtractedPatient {
        val text = rawText.lowercase()
        val warnings = mutableListOf<String>()

        // 1. Blood Pressure Parsing (SBP & DBP)
        var apHi: Int? = null
        var apLo: Int? = null

        // Context-aware pair match: "BP 140/90" or "BP: 140/90 mmHg" or "140/90"
        val bpMatch = Regex("""(?:bp|blood\s*pressure|sbp/dbp)\D*(\d{2,3})\s*[\/:-]\s*(\d{2,3})""").find(text)
        if (bpMatch != null) {
            apHi = bpMatch.groupValues[1].toIntOrNull()
            apLo = bpMatch.groupValues[2].toIntOrNull()
        } else {
            // Separate SBP & DBP matches
            val sbpMatch = Regex("""(?:systolic\s*bp|systolic\s*blood\s*pressure|sbp|sys\s*bp|systolic)\D*(\d{2,3})""").find(text)
            if (sbpMatch != null) apHi = sbpMatch.groupValues[1].toIntOrNull()

            val dbpMatch = Regex("""(?:diastolic\s*bp|diastolic\s*blood\s*pressure|dbp|dia\s*bp|diastolic)\D*(\d{2,3})""").find(text)
            if (dbpMatch != null) apLo = dbpMatch.groupValues[2].toIntOrNull()
        }

        // Sanity check BP ranges
        if (apHi != null && (apHi < 60 || apHi > 260)) apHi = null
        if (apLo != null && (apLo < 40 || apLo > 180)) apLo = null

        // 2. Glucose / Blood Sugar
        var glucose: Int? = null
        val glucNumMatch = Regex("""(?:blood\s*sugar|blood\s*glucose|fasting\s*glucose|fasting\s*blood\s*sugar|rbs|fbs|ppbs|glucose|sugar|gluc)\D*(\d{2,3})""").find(text)
        if (glucNumMatch != null) {
            val valNum = glucNumMatch.groupValues[1].toIntOrNull() ?: 0
            glucose = when {
                valNum >= 126 -> 3
                valNum >= 100 -> 2
                valNum > 0 -> 1
                else -> null
            }
        }
        if (glucose == null) {
            val glucLvlMatch = Regex("""(?:glucose|blood\s*sugar|sugar|gluc)\D*level\s*([123])""").find(text)
            if (glucLvlMatch != null) {
                glucose = glucLvlMatch.groupValues[1].toIntOrNull()
            }
        }

        // 3. Cholesterol & Lipid Profile Handling
        var cholesterol: Int? = null
        val cholNumMatch = Regex("""(?:total\s*cholesterol|serum\s*cholesterol|tc|cholesterol|chol)\D*(\d{2,3})""").find(text)
        if (cholNumMatch != null) {
            val valNum = cholNumMatch.groupValues[1].toIntOrNull() ?: 0
            cholesterol = when {
                valNum >= 240 -> 3
                valNum >= 200 -> 2
                valNum > 0 -> 1
                else -> null
            }
        }
        if (cholesterol == null) {
            val cholLvlMatch = Regex("""(?:cholesterol|chol)\D*level\s*([123])""").find(text)
            if (cholLvlMatch != null) {
                cholesterol = cholLvlMatch.groupValues[1].toIntOrNull()
            }
        }

        // Check if only LDL/HDL/Triglycerides present without total cholesterol
        if (cholesterol == null && (text.contains("ldl") || text.contains("hdl") || text.contains("triglycerides"))) {
            warnings.add("Cholesterol level could not be mapped automatically — please review.")
        }

        // 4. DOB & Age Parsing
        var age: Double? = null
        var calculatedAgeFromDob: Double? = null

        // Try DOB match: "dob: 12/05/1971" or "date of birth: 1971-05-12"
        val dobMatch = Regex("""(?:dob|date\s*of\s*birth|birth\s*date|d\.o\.b\.|born)\D*(\d{1,4}[\/\.-]\d{1,2}[\/\.-]\d{1,4})""").find(text)
        if (dobMatch != null) {
            val dobStr = dobMatch.groupValues[1]
            calculatedAgeFromDob = calculateAgeFromDobString(dobStr)
            if (calculatedAgeFromDob == null) {
                warnings.add("Date format unclear — please verify.")
            }
        }

        // Try direct Age match
        val ageMatch = Regex("""(?:patient\s*age|age)\D*(\d{1,3}(?:\.\d)?)\s*(?:yrs|years|y)?\b""").find(text)
        if (ageMatch != null) {
            age = ageMatch.groupValues[1].toDoubleOrNull()
        }

        // Apply Age from DOB if Age is missing
        if (age == null && calculatedAgeFromDob != null) {
            age = calculatedAgeFromDob
        } else if (age != null && calculatedAgeFromDob != null) {
            if (Math.abs(age - calculatedAgeFromDob) > 1.5) {
                warnings.add("Age and DOB appear inconsistent — please verify.")
            }
        }

        // 5. Height (cm, m, in -> cm)
        var height: Double? = null
        val hmMatch = Regex("""(?:height|ht|body\s*height|stature)\D*(\d(?:\.\d{1,2})?)\s*m\b""").find(text)
        if (hmMatch != null) {
            val mVal = hmMatch.groupValues[1].toDoubleOrNull()
            if (mVal != null && mVal in 0.8..2.5) {
                height = mVal * 100.0
            }
        }
        if (height == null) {
            val hcmMatch = Regex("""(?:height|ht|body\s*height|stature)\D*(\d{2,3}(?:\.\d)?)\s*(?:cm)?""").find(text)
            if (hcmMatch != null) {
                val valNum = hcmMatch.groupValues[1].toDoubleOrNull()
                if (valNum != null && valNum in 50.0..250.0) {
                    height = valNum
                }
            }
        }

        // 6. Weight (kg, lbs -> kg)
        var weight: Double? = null
        val wlbsMatch = Regex("""(?:weight|wt|body\s*weight)\D*(\d{2,3}(?:\.\d)?)\s*(?:lbs|lb)\b""").find(text)
        if (wlbsMatch != null) {
            val lbsVal = wlbsMatch.groupValues[1].toDoubleOrNull()
            if (lbsVal != null) {
                weight = lbsVal * 0.45359237
            }
        }
        if (weight == null) {
            val wkgMatch = Regex("""(?:weight|wt|body\s*weight)\D*(\d{2,3}(?:\.\d)?)\s*(?:kg)?""").find(text)
            if (wkgMatch != null) {
                val valNum = wkgMatch.groupValues[1].toDoubleOrNull()
                if (valNum != null && valNum in 20.0..300.0) {
                    weight = valNum
                }
            }
        }

        // 7. Gender / Sex
        var gender: Int? = null
        val sexMatch = Regex("""(?:gender|sex)\D*(female|male|f|m)\b""").find(text)
        if (sexMatch != null) {
            val sexStr = sexMatch.groupValues[1]
            gender = if (sexStr == "female" || sexStr == "f") 1 else 2
        }

        // 8. Smoking Status
        var smoke: Int? = null
        val smokeMatch = Regex("""(?:smoking\s*status|smoking|smoker|tobacco)\D*(non-smoker|never|former|current|active|yes|no)\b""").find(text)
        if (smokeMatch != null) {
            val sStr = smokeMatch.groupValues[1]
            smoke = if (sStr == "active" || sStr == "current" || sStr == "yes") 1 else 0
        }

        // 9. Alcohol Status
        var alco: Int? = null
        val alcoMatch = Regex("""(?:alcohol\s*intake|alcohol\s*use|alcohol|drinking)\D*(non-drinker|regular|yes|no)\b""").find(text)
        if (alcoMatch != null) {
            val aStr = alcoMatch.groupValues[1]
            alco = if (aStr == "regular" || aStr == "yes") 1 else 0
        }

        // 10. Physical Activity
        var active: Int? = null
        val activeMatch = Regex("""(?:physical\s*activity|exercise|activity\s*level|active|sedentary)\D*(active|inactive|sedentary|yes|no)\b""").find(text)
        if (activeMatch != null) {
            val actStr = activeMatch.groupValues[1]
            active = if (actStr == "active" || actStr == "yes") 1 else 0
        }

        val combinedWarning = if (warnings.isNotEmpty()) warnings.joinToString("\n") else null

        return OcrExtractedPatient(
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
            warningMsg = combinedWarning
        )
    }

    private fun calculateAgeFromDobString(dobStr: String): Double? {
        val formats = arrayOf(
            SimpleDateFormat("dd/MM/yyyy", Locale.US),
            SimpleDateFormat("dd-MM-yyyy", Locale.US),
            SimpleDateFormat("yyyy-MM-dd", Locale.US),
            SimpleDateFormat("MMM dd, yyyy", Locale.US),
            SimpleDateFormat("dd MMM yyyy", Locale.US)
        )

        for (fmt in formats) {
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
            } catch (e: Exception) {
                // Continue trying formats
            }
        }
        return null
    }
}
