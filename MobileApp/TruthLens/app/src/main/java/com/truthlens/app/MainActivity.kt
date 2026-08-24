package com.truthlens.app

import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.truthlens.app.ui.theme.TruthLensTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {

    private lateinit var truthLensModel: TruthLensModel

    private var selectedBitmap by mutableStateOf<android.graphics.Bitmap?>(null)

    private var prediction by mutableStateOf<TruthLensModel.Prediction?>(null)

    private var isAnalyzing by mutableStateOf(false)

    private val imagePicker =
        registerForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri ->

            if (uri != null) {

                lifecycleScope.launch {

                    val bitmap = withContext(Dispatchers.IO) {

                        contentResolver
                            .openInputStream(uri)
                            ?.use { inputStream ->
                                BitmapFactory.decodeStream(inputStream)
                            }
                    }

                    selectedBitmap = bitmap

                    // Clear previous result
                    prediction = null
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)

        enableEdgeToEdge()

        // Load TruthLens AI model
        truthLensModel = TruthLensModel(this)

        setContent {

            TruthLensTheme {

                Scaffold(
                    modifier = Modifier.fillMaxSize()
                ) { innerPadding ->

                    TruthLensScreen(
                        modifier = Modifier.padding(innerPadding),

                        bitmap = selectedBitmap,

                        prediction = prediction,

                        isAnalyzing = isAnalyzing,

                        onSelectImage = {

                            imagePicker.launch("image/*")
                        },

                        onAnalyze = {

                            val bitmap = selectedBitmap

                            if (bitmap != null) {

                                isAnalyzing = true

                                lifecycleScope.launch {

                                    val result =
                                        withContext(Dispatchers.Default) {

                                            truthLensModel.classify(
                                                bitmap
                                            )
                                        }

                                    prediction = result

                                    isAnalyzing = false
                                }
                            }
                        }
                    )
                }
            }
        }
    }

    override fun onDestroy() {

        truthLensModel.close()

        super.onDestroy()
    }
}


@androidx.compose.runtime.Composable
fun TruthLensScreen(
    modifier: Modifier = Modifier,
    bitmap: android.graphics.Bitmap?,
    prediction: TruthLensModel.Prediction?,
    isAnalyzing: Boolean,
    onSelectImage: () -> Unit,
    onAnalyze: () -> Unit
) {

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(24.dp),

        horizontalAlignment = Alignment.CenterHorizontally,

        verticalArrangement = Arrangement.Center
    ) {

        Text(
            text = "TruthLens",
            style = MaterialTheme.typography.headlineLarge
        )

        Spacer(
            modifier = Modifier.height(8.dp)
        )

        Text(
            text = "Digital Media Authenticity Verification",
            style = MaterialTheme.typography.bodyMedium
        )

        Spacer(
            modifier = Modifier.height(24.dp)
        )

        // Selected image
        if (bitmap != null) {

            Image(
                bitmap = bitmap.asImageBitmap(),

                contentDescription = "Selected image",

                modifier = Modifier
                    .size(280.dp),

                contentScale = ContentScale.Fit
            )

            Spacer(
                modifier = Modifier.height(20.dp)
            )
        }

        // Select image button
        Button(
            onClick = onSelectImage,

            modifier = Modifier
                .fillMaxWidth()
        ) {

            Text(
                text = "Select Image"
            )
        }

        Spacer(
            modifier = Modifier.height(12.dp)
        )

        // Analyze button
        Button(
            onClick = onAnalyze,

            enabled = bitmap != null && !isAnalyzing,

            modifier = Modifier
                .fillMaxWidth()
        ) {

            if (isAnalyzing) {

                CircularProgressIndicator(
                    modifier = Modifier.size(20.dp)
                )

            } else {

                Text(
                    text = "Analyze Image"
                )
            }
        }

        Spacer(
            modifier = Modifier.height(24.dp)
        )

        // Result
        if (prediction != null) {

            Text(
                text = "Analysis Result",
                style = MaterialTheme.typography.titleLarge
            )

            Spacer(
                modifier = Modifier.height(12.dp)
            )

            Text(
                text = prediction.label,
                style = MaterialTheme.typography.headlineMedium
            )

            Spacer(
                modifier = Modifier.height(8.dp)
            )

            Text(
                text = "Confidence: ${
                    String.format(
                        "%.2f",
                        prediction.confidence * 100f
                    )
                }%"
            )

            Spacer(
                modifier = Modifier.height(16.dp)
            )

            Text(
                text = "AI Generated: ${
                    String.format(
                        "%.2f",
                        prediction.probabilities[0] * 100f
                    )
                }%"
            )

            Text(
                text = "Deepfake: ${
                    String.format(
                        "%.2f",
                        prediction.probabilities[1] * 100f
                    )
                }%"
            )

            Text(
                text = "Manipulated: ${
                    String.format(
                        "%.2f",
                        prediction.probabilities[2] * 100f
                    )
                }%"
            )

            Text(
                text = "Real: ${
                    String.format(
                        "%.2f",
                        prediction.probabilities[3] * 100f
                    )
                }%"
            )
        }
    }
}