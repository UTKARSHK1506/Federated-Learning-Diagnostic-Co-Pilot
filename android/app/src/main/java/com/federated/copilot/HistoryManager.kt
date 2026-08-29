package com.federated.copilot

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class AssessmentRecord(
    val id: String,
    val date: String,
    val probability: Double,
    val riskCategory: String,
    val bpStr: String,
    val ageStr: String
)

object HistoryManager {
    private const val PREF_NAME = "cardiosense_history"
    private const val KEY_RECORDS = "records_json"

    fun saveAssessment(context: Context, probability: Double, riskCategory: String, apHi: Int, apLo: Int, age: Double) {
        val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        val existingStr = prefs.getString(KEY_RECORDS, "[]") ?: "[]"
        val array = JSONArray(existingStr)

        val dateFormat = SimpleDateFormat("MMM dd, yyyy - HH:mm", Locale.getDefault())
        val dateStr = dateFormat.format(Date())

        val recordObj = JSONObject().apply {
            put("id", System.currentTimeMillis().toString())
            put("date", dateStr)
            put("probability", probability)
            put("riskCategory", riskCategory)
            put("bpStr", "$apHi/$apLo mmHg")
            put("ageStr", String.format("%.1f yrs", age))
        }

        val newArray = JSONArray()
        newArray.put(recordObj)
        for (i in 0 until array.length()) {
            if (i < 20) { // Limit to 20 recent records
                newArray.put(array.getJSONObject(i))
            }
        }

        prefs.edit().putString(KEY_RECORDS, newArray.toString()).apply()
    }

    fun getHistory(context: Context): List<AssessmentRecord> {
        val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        val existingStr = prefs.getString(KEY_RECORDS, "[]") ?: "[]"
        val array = JSONArray(existingStr)
        val list = mutableListOf<AssessmentRecord>()

        for (i in 0 until array.length()) {
            val obj = array.getJSONObject(i)
            list.add(
                AssessmentRecord(
                    id = obj.optString("id"),
                    date = obj.optString("date"),
                    probability = obj.optDouble("probability", 0.0),
                    riskCategory = obj.optString("riskCategory", "Unknown"),
                    bpStr = obj.optString("bpStr", "--"),
                    ageStr = obj.optString("ageStr", "--")
                )
            )
        }
        return list
    }
}
