package com.federated.copilot

import org.junit.Assert.*
import org.junit.Test

class OcrReportParserTest {

    private fun logTestResult(testName: String, passed: Boolean, expected: String, actual: String) {
        println("==================================================================")
        println("TEST: $testName")
        println("RESULT: ${if (passed) "PASS ✅" else "FAIL ❌"}")
        println("EXPECTED: $expected")
        println("ACTUAL:   $actual")
        println("==================================================================")
    }

    @Test
    fun test01_NormalInlineMedicalReport() {
        val text = """
            Patient Medical Report
            Age: 34 years
            Gender: Female
            Height: 172 cm
            Weight: 68 kg
            Blood Pressure: 118/76 mmHg
            Total Cholesterol: 182 mg/dL
            Glucose: 95 mg/dL
            Smoking Status: No
            Alcohol Consumption: No
            Physical Activity: Yes
        """.trimIndent()

        val result = OcrReportParser.parseText(text)
        val data = result.parsedData

        val passed = data.age == 34.0 &&
                data.gender == 1 &&
                data.height == 172.0 &&
                data.weight == 68.0 &&
                data.apHi == 118 &&
                data.apLo == 76 &&
                data.cholesterol == 1 &&
                data.glucose == 1 &&
                data.smoke == 0 &&
                data.alco == 0 &&
                data.active == 1

        val expected = "10/10 fields (age=34, gender=1, height=172, weight=68, SBP=118, DBP=76, chol=1, gluc=1, smoke=0, alco=0, active=1)"
        val actual = "${data.fieldsFoundCount()}/10 fields ($data)"

        logTestResult("TEST 1 - Normal Inline Medical Report", passed, expected, actual)
        assertTrue(passed)
    }

    @Test
    fun test02_DisconnectedTwoColumnTableOcrOutput() {
        val text = """
            Patient Information
            Age
            Gender
            Height
            Weight
            Blood Pressure
            Cholesterol
            Glucose
            Smoking
            Alcohol
            Physical Activity

            55 years
            Male
            165 cm
            70 kg
            140 / 90 mmHg
            Level 2
            Level 1
            No
            No
            Yes
        """.trimIndent()

        val result = OcrReportParser.parseText(text)
        val data = result.parsedData

        val passed = data.age == 55.0 &&
                data.gender == 2 &&
                data.height == 165.0 &&
                data.weight == 70.0 &&
                data.apHi == 140 &&
                data.apLo == 90 &&
                data.cholesterol == 2 &&
                data.glucose == 1 &&
                data.smoke == 0 &&
                data.alco == 0 &&
                data.active == 1

        val expected = "10/10 fields (age=55, gender=2, height=165, weight=70, SBP=140, DBP=90, chol=2, gluc=1, smoke=0, alco=0, active=1)"
        val actual = "${data.fieldsFoundCount()}/10 fields ($data)"

        logTestResult("TEST 2 - Disconnected 2-Column Table OCR", passed, expected, actual)
        if (!passed) {
            println("TEST 2 FAILED DATA DETAILS: $data")
        }
        assertTrue(passed)
    }

    @Test
    fun test03_BloodPressureSlashFormat() {
        val text = "Patient Vitals: Blood Pressure 140 / 90 mmHg"
        val result = OcrReportParser.parseText(text)
        val data = result.parsedData

        val passed = data.apHi == 140 && data.apLo == 90
        logTestResult("TEST 3 - Blood Pressure 140 / 90 mmHg", passed, "apHi=140, apLo=90", "apHi=${data.apHi}, apLo=${data.apLo}")
        assertTrue(passed)
    }

    @Test
    fun test04_DisconnectedSystolicAndDiastolic() {
        val text = """
            Vitals Reading:
            Systolic: 140 mmHg
            Diastolic: 90 mmHg
        """.trimIndent()

        val result = OcrReportParser.parseText(text)
        val data = result.parsedData

        val passed = data.apHi == 140 && data.apLo == 90
        logTestResult("TEST 4 - Systolic 140 / Diastolic 90", passed, "apHi=140, apLo=90", "apHi=${data.apHi}, apLo=${data.apLo}")
        assertTrue(passed)
    }

    @Test
    fun test05_NumericCholesterol() {
        val text = "Total Cholesterol: 182 mg/dL"
        val result = OcrReportParser.parseText(text)
        val data = result.parsedData

        val passed = data.cholesterol == 1
        logTestResult("TEST 5 - Cholesterol 182 mg/dL", passed, "cholesterol=1 (Normal)", "cholesterol=${data.cholesterol}")
        assertTrue(passed)
    }

    @Test
    fun test06_CategoricalCholesterolLevel2() {
        val text = "Serum Cholesterol: Level 2"
        val result = OcrReportParser.parseText(text)
        val data = result.parsedData

        val passed = data.cholesterol == 2
        logTestResult("TEST 6 - Cholesterol Level 2", passed, "cholesterol=2 (Above Normal)", "cholesterol=${data.cholesterol}")
        assertTrue(passed)
    }

    @Test
    fun test07_NumericGlucoseHigh() {
        val text = "Fasting Blood Glucose: 126 mg/dL"
        val result = OcrReportParser.parseText(text)
        val data = result.parsedData

        val passed = data.glucose == 3
        logTestResult("TEST 7 - Glucose 126 mg/dL", passed, "glucose=3 (Well Above Normal)", "glucose=${data.glucose}")
        assertTrue(passed)
    }

    @Test
    fun test08_SmokingCurrentSmoker() {
        val text = "Smoking Status: Current Smoker"
        val result = OcrReportParser.parseText(text)
        val data = result.parsedData

        val passed = data.smoke == 1
        logTestResult("TEST 8 - Smoking: Current Smoker", passed, "smoke=1 (Active Smoker)", "smoke=${data.smoke}")
        assertTrue(passed)
    }

    @Test
    fun test09_SmokingFormerSmokerAmbiguous() {
        val text = "Smoking History: Former Smoker (Quit 5 years ago)"
        val result = OcrReportParser.parseText(text)
        val data = result.parsedData

        val passed = data.smoke == null && data.ambiguousFields.contains("Smoking Status")
        logTestResult("TEST 9 - Smoking: Former Smoker", passed, "smoke=null (AMBIGUOUS)", "smoke=${data.smoke}, ambiguous=${data.ambiguousFields}")
        assertTrue(passed)
    }

    @Test
    fun test10_DobCalculationAge() {
        val text = """
            Patient Name: John Doe
            Date of Birth: 15/06/1971
            Report Date: 30/08/2026
        """.trimIndent()

        val result = OcrReportParser.parseText(text)
        val data = result.parsedData

        val passed = data.age != null && Math.abs(data.age!! - 55.0) <= 1.0
        logTestResult("TEST 10 - DOB 15/06/1971 -> Age ~55", passed, "age=55.0", "age=${data.age}")
        assertTrue(passed)
    }

    @Test
    fun test11_OcrTyposNormalization() {
        val text = """
            syst0lic: 130
            diast0lic: 85
            gluc0se: 110
            cholesteroi: 210
            sm0king: no
            alc0hol: no
            physicai activity: yes
        """.trimIndent()

        val result = OcrReportParser.parseText(text)
        val data = result.parsedData

        val passed = data.apHi == 130 &&
                data.apLo == 85 &&
                data.glucose == 2 &&
                data.cholesterol == 2 &&
                data.smoke == 0 &&
                data.alco == 0 &&
                data.active == 1

        logTestResult("TEST 11 - OCR Typos Normalization", passed, "All typo fields recognized correctly", "data=$data")
        assertTrue(passed)
    }

    @Test
    fun test12_UnrelatedNumbersDoNotFalseMatch() {
        val text = """
            CARDIO CLINIC REPORT #984712
            Doctor Reg: 44102
            Patient ID: 8871625
            Phone: 9876543210
            Ref Range: 60-100 mg/dL
            Date: 2026-08-30
        """.trimIndent()

        val result = OcrReportParser.parseText(text)
        val data = result.parsedData

        // Ensure Patient ID, Phone, Report No, Doctor Reg were not mapped to Age, Height, Weight, BP
        val passed = data.age == null && data.height == null && data.weight == null && data.apHi == null
        logTestResult("TEST 12 - Unrelated Numbers Prevention", passed, "No false matches for ID/Phone/Reg numbers", "data=$data")
        assertTrue(passed)
    }

    @Test
    fun test13_ReportWithMissingFields() {
        val text = """
            Patient Age: 50
            Gender: Male
            Blood Pressure: 130/80 mmHg
        """.trimIndent()

        val result = OcrReportParser.parseText(text)
        val data = result.parsedData

        val passed = data.age == 50.0 && data.gender == 2 && data.apHi == 130 && data.apLo == 80 && data.height == null && data.weight == null
        logTestResult("TEST 13 - Partial Medical Report", passed, "Fields found=3/10, height/weight=null", "fieldsFound=${data.fieldsFoundCount()}/10")
        assertTrue(passed)
    }

    @Test
    fun test14_HighResolutionSimulatedInput() {
        val text = "HEIGHT: 180 CM\nWEIGHT: 85 KG\nBP: 120 / 80 MMHG"
        val result = OcrReportParser.parseText(text)
        val data = result.parsedData

        val passed = data.height == 180.0 && data.weight == 85.0 && data.apHi == 120 && data.apLo == 80
        logTestResult("TEST 14 - High Res Simulated Input", passed, "Height 180, Weight 85, BP 120/80", "data=$data")
        assertTrue(passed)
    }

    @Test
    fun test15_MultiPagePdfTextParsing() {
        val text = """
            --- PAGE 1 ---
            PATIENT INFORMATION
            Age: 42 years
            Sex: Female
            Height: 168 cm
            Weight: 62 kg

            --- PAGE 2 ---
            CLINICAL LABORATORY RESULTS
            Blood Pressure: 125/82 mmHg
            Total Cholesterol: 190 mg/dL
            Blood Glucose: 92 mg/dL

            --- PAGE 3 ---
            LIFESTYLE ASSESSMENT
            Smoking: Non-Smoker
            Alcohol: Non-Drinker
            Physical Exercise: Active
        """.trimIndent()

        val result = OcrReportParser.parseText(text)
        val data = result.parsedData

        val passed = data.fieldsFoundCount() == 10
        logTestResult("TEST 15 - Multi-Page PDF Combined Parsing", passed, "10/10 fields populated across pages", "fieldsFound=${data.fieldsFoundCount()}/10")
        assertTrue(passed)
    }

    @Test
    fun test16_ContentUriSimulationText() {
        val text = "Patient Age: 29 years\nGender: Female\nHeight: 160 cm\nWeight: 55 kg"
        val result = OcrReportParser.parseText(text)
        val data = result.parsedData

        val passed = data.age == 29.0 && data.gender == 1 && data.height == 160.0 && data.weight == 55.0
        logTestResult("TEST 16 - Photo Picker / Content URI Text", passed, "Fields extracted cleanly", "data=$data")
        assertTrue(passed)
    }

    @Test
    fun test17_CameraCapturedDocumentText() {
        val text = """
            CLINICAL ASSESSMENT FORM
            Age: 48
            Sex: Male
            Height (cm): 175
            Weight (kg): 78
            Blood Pressure: 135/88
            Total Chol (mg/dL): 215
            Fasting Sugar (mg/dL): 108
            Smoking: No
            Alcohol: No
            Active: Yes
        """.trimIndent()

        val result = OcrReportParser.parseText(text)
        val data = result.parsedData

        val passed = data.age == 48.0 &&
                data.gender == 2 &&
                data.height == 175.0 &&
                data.weight == 78.0 &&
                data.apHi == 135 &&
                data.apLo == 88 &&
                data.cholesterol == 2 &&
                data.glucose == 2 &&
                data.smoke == 0 &&
                data.alco == 0 &&
                data.active == 1

        logTestResult("TEST 17 - Camera Captured Document Text", passed, "10/10 fields matched", "fieldsFound=${data.fieldsFoundCount()}/10")
        if (!passed) {
            println("TEST 17 FAILED DATA DETAILS: $data")
        }
        assertTrue(passed)
    }

    @Test
    fun test18_UserUploadedPdfReport() {
        val text = """
            PATIENT HEALTH REPORT
            Synthetic / Dummy Patient Record
            PATIENT INFORMATION
            Patient ID P001
            Age 34 years
            Gender Male
            Height 172 cm
            Weight 68 kg
            CLINICAL MEASUREMENTS
            Blood Pressure 118 / 76 mmHg
            Systolic BP 118 mmHg
            Diastolic BP 76 mmHg
            Cholesterol 182 mg/dL
            Glucose Level 2
            LIFESTYLE INFORMATION
            Alcohol No
            Smoker No
            Physically Active Yes
            NOTE
            This report contains entirely synthetic data created for testing, demonstration, or model-development purposes. It does not represent a real patient.
        """.trimIndent()

        val result = OcrReportParser.parseText(text)
        val data = result.parsedData

        val passed = data.age == 34.0 &&
                data.gender == 2 &&
                data.height == 172.0 &&
                data.weight == 68.0 &&
                data.apHi == 118 &&
                data.apLo == 76 &&
                data.cholesterol == 1 &&
                data.glucose == 2 &&
                data.smoke == 0 &&
                data.alco == 0 &&
                data.active == 1

        logTestResult("TEST 18 - User Uploaded Synthetic PDF Report", passed, "10/10 fields matched (age=34, active=1)", "fieldsFound=${data.fieldsFoundCount()}/10 ($data)")
        if (!passed) {
            println("TEST 18 FAILED DATA DETAILS: $data")
        }
        assertTrue(passed)
    }
}

