package com.truthlens.app

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

class TruthLensModel(
    private val context: Context
) {

    companion object {
        private const val TAG = "TruthLensModel"

        private const val MODEL_NAME = "truthlens_model.tflite"
        private const val LABELS_NAME = "labels.txt"

        private const val IMAGE_SIZE = 224
        private const val NUM_CLASSES = 4
    }

    // Class labels
    private val labels: List<String> = loadLabels()

    // LiteRT model
    private val model: CompiledModel =
        CompiledModel.create(
            context.assets,
            MODEL_NAME,
            CompiledModel.Options(Accelerator.CPU)
        )

    // Explicit types are important here
    private val inputBuffers: List<TensorBuffer> =
        model.createInputBuffers()

    private val outputBuffers: List<TensorBuffer> =
        model.createOutputBuffers()

    init {
        require(labels.size == NUM_CLASSES) {
            "Expected $NUM_CLASSES labels, but found ${labels.size}"
        }

        Log.d(TAG, "TruthLens model loaded")
        Log.d(TAG, "Labels: $labels")
    }

    /**
     * Classify a bitmap image.
     */
    fun classify(bitmap: Bitmap): Prediction {

        // Resize image to 224 x 224
        val resizedBitmap = Bitmap.createScaledBitmap(
            bitmap,
            IMAGE_SIZE,
            IMAGE_SIZE,
            true
        )

        // Convert Bitmap to Float32 input
        val inputData: FloatArray =
            bitmapToFloatArray(resizedBitmap)

        // Send input to LiteRT
        inputBuffers[0].writeFloat(inputData)

        // Run model
        model.run(
            inputBuffers,
            outputBuffers
        )

        // Read model output
        val output: FloatArray =
            outputBuffers[0].readFloat()

        require(output.size == NUM_CLASSES) {
            "Expected $NUM_CLASSES outputs, but received ${output.size}"
        }

        // Find highest probability
        var bestIndex: Int = 0
        var bestProbability: Float = output[0]

        for (i in 1 until output.size) {
            if (output[i] > bestProbability) {
                bestProbability = output[i]
                bestIndex = i
            }
        }

        val predictedLabel: String =
            labels[bestIndex]

        val confidence: Float =
            bestProbability.coerceIn(0f, 1f)

        // Log all model probabilities
        Log.d(TAG, "==============================")
        Log.d(TAG, "TruthLens Prediction")
        Log.d(TAG, "AI_Generated: ${output[0]}")
        Log.d(TAG, "Deepfake: ${output[1]}")
        Log.d(TAG, "Manipulated: ${output[2]}")
        Log.d(TAG, "Real: ${output[3]}")
        Log.d(TAG, "------------------------------")
        Log.d(TAG, "Prediction: $predictedLabel")
        Log.d(TAG, "Confidence: ${confidence * 100f}%")
        Log.d(TAG, "==============================")

        return Prediction(
            label = predictedLabel,
            confidence = confidence,
            probabilities = output.toList()
        )
    }

    /**
     * Convert Bitmap into Float32 RGB array.
     *
     * Model input:
     * [1, 224, 224, 3]
     */
    private fun bitmapToFloatArray(
        bitmap: Bitmap
    ): FloatArray {

        val pixels = IntArray(
            IMAGE_SIZE * IMAGE_SIZE
        )

        bitmap.getPixels(
            pixels,
            0,
            IMAGE_SIZE,
            0,
            0,
            IMAGE_SIZE,
            IMAGE_SIZE
        )

        val input = FloatArray(
            IMAGE_SIZE * IMAGE_SIZE * 3
        )

        var index: Int = 0

        for (pixel in pixels) {

            val red: Int =
                (pixel shr 16) and 0xFF

            val green: Int =
                (pixel shr 8) and 0xFF

            val blue: Int =
                pixel and 0xFF

            // EfficientNetB0 expects float pixel values in [0,255].
            input[index++] = red.toFloat()
            input[index++] = green.toFloat()
            input[index++] = blue.toFloat()
        }

        return input
    }

    /**
     * Load labels from assets/labels.txt
     */
    private fun loadLabels(): List<String> {

        return context.assets
            .open(LABELS_NAME)
            .bufferedReader()
            .useLines { lines ->
                lines
                    .map { line -> line.trim() }
                    .filter { line -> line.isNotEmpty() }
                    .toList()
            }
    }

    /**
     * Release LiteRT resources.
     */
    fun close() {

        inputBuffers.forEach { buffer ->
            buffer.close()
        }

        outputBuffers.forEach { buffer ->
            buffer.close()
        }

        model.close()

        Log.d(TAG, "TruthLens model closed")
    }

    /**
     * Result returned by the model.
     */
    data class Prediction(
        val label: String,
        val confidence: Float,
        val probabilities: List<Float>
    )
}