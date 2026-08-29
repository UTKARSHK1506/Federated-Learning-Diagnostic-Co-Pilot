package com.federated.copilot.api

data class PatientPredictRequest(
    val age: Double,
    val gender: Int,
    val height: Double,
    val weight: Double,
    val ap_hi: Int,
    val ap_lo: Int,
    val cholesterol: Int,
    val gluc: Int,
    val smoke: Int,
    val alco: Int,
    val active: Int
)

data class PredictionResponse(
    val predicted_class: Int,
    val probability: Double,
    val model_type: String
)

data class ModelInfoResponse(
    val model_architecture: String,
    val federated_rounds: Int,
    val participating_clinics: List<String>,
    val model_status: String
)
