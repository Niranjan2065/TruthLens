package com.truthlens.app

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Bundle
import android.provider.Settings

import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding

import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll

import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale

import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

import androidx.core.content.ContextCompat

import androidx.lifecycle.lifecycleScope

import com.truthlens.app.ui.theme.TruthLensTheme

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext


class MainActivity : ComponentActivity() {

    /*
     * ============================================================
     * MODEL
     * ============================================================
     */

    private lateinit var truthLensModel:
            TruthLensModel


    /*
     * ============================================================
     * MEDIA PROJECTION
     * ============================================================
     */

    private lateinit var mediaProjectionManager:
            MediaProjectionManager


    /*
     * ============================================================
     * UI STATE
     * ============================================================
     */

    private var selectedBitmap by
    mutableStateOf<Bitmap?>(null)


    private var prediction by
    mutableStateOf<
            TruthLensModel.Prediction?
            >(null)


    private var isAnalyzing by
    mutableStateOf(false)


    private var isCapturing by
    mutableStateOf(false)


    private var captureMessage by
    mutableStateOf<String?>(null)


    /*
     * ============================================================
     * FLOATING BUTTON STATE
     * ============================================================
     *
     * Step 1 of the floating-button feature: just track whether the
     * overlay service is running and show a status message. Tapping
     * the floating button itself is not yet wired to the capture
     * flow -- that comes in the next step.
     */

    private var isFloatingButtonActive by
    mutableStateOf(false)


    private var floatingButtonMessage by
    mutableStateOf<String?>(null)


    /*
     * ============================================================
     * IMAGE PICKER
     * ============================================================
     */

    private val imagePicker =
        registerForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri ->

            if (uri != null) {

                lifecycleScope.launch {

                    val bitmap =
                        withContext(
                            Dispatchers.IO
                        ) {

                            contentResolver
                                .openInputStream(
                                    uri
                                )
                                ?.use { inputStream ->

                                    BitmapFactory
                                        .decodeStream(
                                            inputStream
                                        )
                                }
                        }


                    selectedBitmap =
                        bitmap

                    prediction =
                        null

                    captureMessage =
                        null
                }
            }
        }


    /*
     * ============================================================
     * SCREEN CAPTURE PERMISSION
     * ============================================================
     */

    private val screenCaptureLauncher =
        registerForActivityResult(
            ActivityResultContracts
                .StartActivityForResult()
        ) { result ->

            if (
                result.resultCode ==
                RESULT_OK &&
                result.data != null
            ) {

                /*
                 * Capture has started.
                 */

                isCapturing =
                    true

                prediction =
                    null

                captureMessage =
                    "Capture started. Switch to the media you want to verify."


                /*
                 * Build service Intent.
                 */

                val serviceIntent =
                    Intent(
                        this,
                        ScreenCaptureService::class.java
                    ).apply {

                        action =
                            ScreenCaptureService
                                .ACTION_START_CAPTURE


                        putExtra(
                            ScreenCaptureService
                                .EXTRA_RESULT_CODE,
                            result.resultCode
                        )


                        putExtra(
                            ScreenCaptureService
                                .EXTRA_RESULT_DATA,
                            result.data
                        )
                    }


                /*
                 * Start foreground service.
                 */

                ContextCompat
                    .startForegroundService(
                        this,
                        serviceIntent
                    )

            } else {

                /*
                 * User cancelled permission.
                 */

                isCapturing =
                    false

                captureMessage =
                    "Screen capture permission was cancelled."
            }
        }


    /*
     * ============================================================
     * OVERLAY ("DRAW OVER OTHER APPS") PERMISSION
     * ============================================================
     *
     * SYSTEM_ALERT_WINDOW is a "special" permission: it isn't granted
     * through the normal runtime permission dialog. Instead we send
     * the user to a system Settings screen and re-check
     * Settings.canDrawOverlays() when they come back.
     */

    private val overlayPermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts
                .StartActivityForResult()
        ) {

            if (
                Settings.canDrawOverlays(this)
            ) {

                startFloatingButtonService()

            } else {

                floatingButtonMessage =
                    "Overlay permission was not granted. TruthLens needs " +
                            "\"Display over other apps\" to show the floating button."
            }
        }


    /*
     * ============================================================
     * FLOATING SESSION: SCREEN CAPTURE PERMISSION
     * ============================================================
     *
     * Requested ONCE when the floating button is enabled (after the
     * overlay permission is confirmed). The granted token is handed
     * to FloatingButtonService, which keeps it alive for the whole
     * floating-button session -- taps never show this dialog again.
     *
     * Deliberately separate from screenCaptureLauncher, which backs
     * the existing one-shot "Capture Screen" button and is left
     * completely untouched.
     */

    private val floatingCaptureAuthLauncher =
        registerForActivityResult(
            ActivityResultContracts
                .StartActivityForResult()
        ) { result ->

            if (
                result.resultCode == RESULT_OK &&
                result.data != null
            ) {

                val serviceIntent =
                    Intent(
                        this,
                        FloatingButtonService::class.java
                    ).apply {

                        action =
                            FloatingButtonService
                                .ACTION_START_SESSION

                        putExtra(
                            FloatingButtonService
                                .EXTRA_PROJECTION_RESULT_CODE,
                            result.resultCode
                        )

                        putExtra(
                            FloatingButtonService
                                .EXTRA_PROJECTION_RESULT_DATA,
                            result.data
                        )
                    }

                ContextCompat.startForegroundService(
                    this,
                    serviceIntent
                )

                isFloatingButtonActive = true

                floatingButtonMessage =
                    "Floating button enabled. Switch to another app to test it."

            } else {

                floatingButtonMessage =
                    "Screen capture permission was cancelled. " +
                            "The floating button needs it to work."
            }
        }


    /*
     * ============================================================
     * SCREEN CAPTURE RESULT RECEIVER
     * ============================================================
     */

    private val screenCaptureReceiver =
        object : BroadcastReceiver() {

            override fun onReceive(
                context: Context?,
                intent: Intent?
            ) {

                if (
                    intent?.action !=
                    ScreenCaptureService
                        .ACTION_CAPTURE_COMPLETE
                ) {

                    return
                }


                /*
                 * Check for an error.
                 */

                val error =
                    intent.getStringExtra(
                        ScreenCaptureService
                            .EXTRA_ERROR
                    )


                if (
                    error != null
                ) {

                    isCapturing =
                        false

                    captureMessage =
                        "Capture failed: $error"

                    return
                }


                /*
                 * Get captured image path.
                 */

                val imagePath =
                    intent.getStringExtra(
                        ScreenCaptureService
                            .EXTRA_IMAGE_PATH
                    )


                if (
                    imagePath == null
                ) {

                    isCapturing =
                        false

                    captureMessage =
                        "Capture completed, but no image was returned."

                    return
                }


                /*
                 * Was this capture triggered by the floating button?
                 * If so, run the model automatically once the bitmap
                 * is decoded. Absent for the manual "Capture Screen"
                 * button, so its behavior is unchanged.
                 */

                val autoAnalyze =
                    intent.getBooleanExtra(
                        FloatingButtonService
                            .EXTRA_AUTO_ANALYZE,
                        false
                    )


                /*
                 * Decode captured image.
                 */

                lifecycleScope.launch {

                    val bitmap =
                        withContext(
                            Dispatchers.IO
                        ) {

                            BitmapFactory
                                .decodeFile(
                                    imagePath
                                )
                        }


                    if (
                        bitmap != null
                    ) {

                        selectedBitmap =
                            bitmap

                        prediction =
                            null

                        captureMessage =
                            if (autoAnalyze) {
                                "Captured from the floating button. Analyzing..."
                            } else {
                                "Screen captured successfully."
                            }


                        if (autoAnalyze) {
                            analyzeSelectedImage()
                        }

                    } else {

                        captureMessage =
                            "Captured image could not be loaded."
                    }


                    isCapturing =
                        false
                }
            }
        }


    /*
     * ============================================================
     * ACTIVITY CREATED
     * ============================================================
     */

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {

        super.onCreate(
            savedInstanceState
        )


        enableEdgeToEdge()


        /*
         * Initialize TruthLens model.
         */

        truthLensModel =
            TruthLensModel(this)


        /*
         * Initialize MediaProjection.
         */

        mediaProjectionManager =
            getSystemService(
                Context.MEDIA_PROJECTION_SERVICE
            ) as MediaProjectionManager


        /*
         * Register internal receiver.
         *
         * NOT_EXPORTED is important because the receiver is
         * only used by this application.
         */

        val filter =
            IntentFilter(
                ScreenCaptureService
                    .ACTION_CAPTURE_COMPLETE
            )


        ContextCompat.registerReceiver(
            this,
            screenCaptureReceiver,
            filter,
            ContextCompat
                .RECEIVER_NOT_EXPORTED
        )


        /*
         * Compose UI.
         */

        setContent {

            TruthLensTheme {

                TruthLensScreen(

                    bitmap =
                        selectedBitmap,

                    prediction =
                        prediction,

                    isAnalyzing =
                        isAnalyzing,

                    isCapturing =
                        isCapturing,

                    captureMessage =
                        captureMessage,

                    isFloatingButtonActive =
                        isFloatingButtonActive,

                    floatingButtonMessage =
                        floatingButtonMessage,


                    onSelectImage = {

                        imagePicker.launch(
                            "image/*"
                        )
                    },


                    onCaptureScreen = {

                        requestScreenCapture()
                    },


                    onAnalyze = {

                        analyzeSelectedImage()
                    },

                    onToggleFloatingButton = {

                        onToggleFloatingButton()
                    }
                )
            }
        }
    }


    /*
     * ============================================================
     * REQUEST SCREEN CAPTURE
     * ============================================================
     */

    private fun requestScreenCapture() {

        /*
         * Clear old result.
         */

        prediction =
            null

        captureMessage =
            null


        /*
         * Ask Android for screen-capture permission.
         */

        val captureIntent =
            mediaProjectionManager
                .createScreenCaptureIntent()


        screenCaptureLauncher.launch(
            captureIntent
        )
    }


    /*
     * ============================================================
     * FLOATING BUTTON: TOGGLE
     * ============================================================
     */

    private fun onToggleFloatingButton() {

        if (
            isFloatingButtonActive
        ) {

            stopFloatingButtonService()

        } else {

            requestOverlayPermissionAndStart()
        }
    }


    /*
     * ============================================================
     * FLOATING BUTTON: REQUEST PERMISSION (IF NEEDED) THEN START
     * ============================================================
     */

    private fun requestOverlayPermissionAndStart() {

        if (
            Settings.canDrawOverlays(this)
        ) {

            startFloatingButtonService()
            return
        }


        floatingButtonMessage =
            "Please allow TruthLens to display over other apps, " +
                    "then come back here."

        val intent =
            Intent(
                Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse(
                    "package:$packageName"
                )
            )

        overlayPermissionLauncher.launch(
            intent
        )
    }


    /*
     * ============================================================
     * FLOATING BUTTON: REQUEST CAPTURE PERMISSION THEN START
     * ============================================================
     *
     * This is the ONE place the floating button's screen-capture
     * permission is requested. FloatingButtonService reuses the
     * resulting token for every tap, so taps themselves never show
     * this dialog again.
     */

    private fun startFloatingButtonService() {

        floatingButtonMessage =
            "Grant screen capture permission to finish enabling " +
                    "the floating button."

        val captureIntent =
            mediaProjectionManager
                .createScreenCaptureIntent()

        floatingCaptureAuthLauncher.launch(
            captureIntent
        )
    }


    /*
     * ============================================================
     * FLOATING BUTTON: STOP SERVICE
     * ============================================================
     */

    private fun stopFloatingButtonService() {

        val serviceIntent =
            Intent(
                this,
                FloatingButtonService::class.java
            ).apply {

                action =
                    FloatingButtonService
                        .ACTION_STOP_OVERLAY
            }


        /*
         * The service is already running (and foreground), so this
         * delivers the stop action to its onStartCommand() rather
         * than creating a second instance.
         */

        startService(
            serviceIntent
        )

        isFloatingButtonActive =
            false

        floatingButtonMessage =
            "Floating button disabled."
    }


    /*
     * ============================================================
     * ANALYZE SELECTED IMAGE
     * ============================================================
     */

    private fun analyzeSelectedImage() {

        val bitmap =
            selectedBitmap


        if (
            bitmap == null
        ) {

            captureMessage =
                "Please select or capture an image first."

            return
        }


        /*
         * Start loading state.
         */

        isAnalyzing =
            true


        prediction =
            null


        lifecycleScope.launch {

            try {

                val result =
                    withContext(
                        Dispatchers.Default
                    ) {

                        truthLensModel
                            .classify(
                                bitmap
                            )
                    }


                prediction =
                    result

            } catch (
                e: Exception
            ) {

                captureMessage =
                    "Analysis failed: ${
                        e.message
                            ?: "Unknown error"
                    }"

            } finally {

                isAnalyzing =
                    false
            }
        }
    }


    /*
     * ============================================================
     * ACTIVITY DESTROYED
     * ============================================================
     */

    override fun onDestroy() {

        try {

            unregisterReceiver(
                screenCaptureReceiver
            )

        } catch (
            _: Exception
        ) {
        }


        try {

            truthLensModel.close()

        } catch (
            _: Exception
        ) {
        }


        super.onDestroy()
    }
}


/*
 * ================================================================
 * TRUTHLENS MAIN SCREEN
 * ================================================================
 */

@androidx.compose.runtime.Composable
fun TruthLensScreen(

    bitmap: Bitmap?,

    prediction:
    TruthLensModel.Prediction?,

    isAnalyzing: Boolean,

    isCapturing: Boolean,

    captureMessage: String?,

    isFloatingButtonActive: Boolean,

    floatingButtonMessage: String?,

    onSelectImage: () -> Unit,

    onCaptureScreen: () -> Unit,

    onAnalyze: () -> Unit,

    onToggleFloatingButton: () -> Unit
) {

    val backgroundColor =
        Color(0xFFFFF9F5)


    val scrollState =
        rememberScrollState()


    Column(

        modifier =
            Modifier
                .fillMaxSize()
                .background(
                    backgroundColor
                )
                .verticalScroll(
                    scrollState
                )
                .padding(
                    horizontal = 24.dp,
                    vertical = 28.dp
                ),

        horizontalAlignment =
            Alignment.CenterHorizontally,

        verticalArrangement =
            Arrangement.Top
    ) {


        Spacer(
            modifier =
                Modifier.height(28.dp)
        )


        /*
         * ========================================================
         * TITLE
         * ========================================================
         */

        Text(

            text =
                "TruthLens",

            style =
                MaterialTheme.typography
                    .headlineLarge,

            fontSize =
                42.sp
        )


        Spacer(
            modifier =
                Modifier.height(8.dp)
        )


        Text(

            text =
                "Digital Media Authenticity Verification",

            fontSize =
                19.sp
        )


        Spacer(
            modifier =
                Modifier.height(32.dp)
        )


        /*
         * ========================================================
         * IMAGE PREVIEW
         * ========================================================
         */

        if (
            bitmap != null
        ) {

            Card(

                modifier =
                    Modifier
                        .fillMaxWidth()
                        .padding(
                            horizontal = 4.dp
                        ),

                shape =
                    RoundedCornerShape(
                        16.dp
                    ),

                colors =
                    CardDefaults
                        .cardColors(
                            containerColor =
                                Color(0xFFF5DFC8)
                        ),

                elevation =
                    CardDefaults
                        .cardElevation(
                            defaultElevation =
                                4.dp
                        )
            ) {

                Image(

                    bitmap =
                        bitmap
                            .asImageBitmap(),

                    contentDescription =
                        "Selected or captured image",

                    modifier =
                        Modifier
                            .fillMaxWidth()
                            .height(
                                400.dp
                            ),

                    contentScale =
                        ContentScale.Fit
                )
            }


            Spacer(
                modifier =
                    Modifier.height(28.dp)
            )
        }


        /*
         * ========================================================
         * SELECT IMAGE
         * ========================================================
         */

        Button(

            onClick =
                onSelectImage,

            enabled =
                !isCapturing &&
                        !isAnalyzing,

            modifier =
                Modifier
                    .fillMaxWidth()
                    .height(60.dp),

            shape =
                RoundedCornerShape(
                    30.dp
                ),

            colors =
                ButtonDefaults
                    .buttonColors(
                        containerColor =
                            Color(
                                0xFF9A5F00
                            )
                    )
        ) {

            Text(

                text =
                    "Select Image",

                fontSize =
                    18.sp
            )
        }


        Spacer(
            modifier =
                Modifier.height(14.dp)
        )


        /*
         * ========================================================
         * CAPTURE SCREEN
         * ========================================================
         */

        Button(

            onClick =
                onCaptureScreen,

            enabled =
                !isCapturing &&
                        !isAnalyzing,

            modifier =
                Modifier
                    .fillMaxWidth()
                    .height(60.dp),

            shape =
                RoundedCornerShape(
                    30.dp
                ),

            colors =
                ButtonDefaults
                    .buttonColors(
                        containerColor =
                            Color(
                                0xFF9A5F00
                            )
                    )
        ) {

            if (
                isCapturing
            ) {

                CircularProgressIndicator(

                    modifier =
                        Modifier.height(
                            24.dp
                        ),

                    color =
                        Color.White,

                    strokeWidth =
                        3.dp
                )

            } else {

                Text(

                    text =
                        "Capture Screen",

                    fontSize =
                        18.sp
                )
            }
        }


        /*
         * ========================================================
         * CAPTURE STATUS
         * ========================================================
         */

        if (
            captureMessage != null
        ) {

            Spacer(
                modifier =
                    Modifier.height(18.dp)
            )


            Text(

                text =
                    captureMessage,

                fontSize =
                    16.sp,

                modifier =
                    Modifier.fillMaxWidth()
            )
        }


        Spacer(
            modifier =
                Modifier.height(18.dp)
        )


        /*
         * ========================================================
         * FLOATING BUTTON TOGGLE
         * ========================================================
         */

        Button(

            onClick =
                onToggleFloatingButton,

            enabled =
                !isCapturing &&
                        !isAnalyzing,

            modifier =
                Modifier
                    .fillMaxWidth()
                    .height(60.dp),

            shape =
                RoundedCornerShape(
                    30.dp
                ),

            colors =
                ButtonDefaults
                    .buttonColors(
                        containerColor =
                            if (isFloatingButtonActive) {
                                Color(0xFFB00020)
                            } else {
                                Color(0xFF9A5F00)
                            }
                    )
        ) {

            Text(

                text =
                    if (isFloatingButtonActive) {
                        "Disable Floating Button"
                    } else {
                        "Enable Floating Button"
                    },

                fontSize =
                    18.sp
            )
        }


        if (
            floatingButtonMessage != null
        ) {

            Spacer(
                modifier =
                    Modifier.height(14.dp)
            )


            Text(

                text =
                    floatingButtonMessage,

                fontSize =
                    16.sp,

                modifier =
                    Modifier.fillMaxWidth()
            )
        }


        Spacer(
            modifier =
                Modifier.height(24.dp)
        )


        /*
         * ========================================================
         * ANALYZE BUTTON
         * ========================================================
         */

        Button(

            onClick =
                onAnalyze,

            enabled =
                bitmap != null &&
                        !isCapturing &&
                        !isAnalyzing,

            modifier =
                Modifier
                    .fillMaxWidth()
                    .height(60.dp),

            shape =
                RoundedCornerShape(
                    30.dp
                ),

            colors =
                ButtonDefaults
                    .buttonColors(
                        containerColor =
                            Color(
                                0xFF9A5F00
                            )
                    )
        ) {

            if (
                isAnalyzing
            ) {

                CircularProgressIndicator(

                    modifier =
                        Modifier.height(
                            24.dp
                        ),

                    color =
                        Color.White,

                    strokeWidth =
                        3.dp
                )

            } else {

                Text(

                    text =
                        "Analyze Image",

                    fontSize =
                        18.sp
                )
            }
        }


        /*
         * ========================================================
         * ANALYSIS RESULT
         * ========================================================
         */

        if (
            prediction != null
        ) {

            Spacer(
                modifier =
                    Modifier.height(36.dp)
            )


            Text(

                text =
                    "Analysis Result",

                fontSize =
                    32.sp
            )


            Spacer(
                modifier =
                    Modifier.height(18.dp)
            )


            Card(

                modifier =
                    Modifier.fillMaxWidth(),

                shape =
                    RoundedCornerShape(
                        18.dp
                    ),

                colors =
                    CardDefaults
                        .cardColors(
                            containerColor =
                                Color.White
                        ),

                elevation =
                    CardDefaults
                        .cardElevation(
                            defaultElevation =
                                4.dp
                        )
            ) {

                Column(

                    modifier =
                        Modifier
                            .fillMaxWidth()
                            .padding(
                                24.dp
                            ),

                    horizontalAlignment =
                        Alignment.CenterHorizontally
                ) {


                    /*
                     * Predicted class.
                     */

                    Text(

                        text =
                            formatLabel(
                                prediction.label
                            ),

                        fontSize =
                            32.sp,

                        fontWeight =
                            FontWeight.Bold
                    )


                    Spacer(
                        modifier =
                            Modifier.height(
                                12.dp
                            )
                    )


                    /*
                     * Confidence.
                     */

                    Text(

                        text =
                            "Confidence: ${
                                String.format(
                                    "%.2f",
                                    prediction
                                        .confidence *
                                            100f
                                )
                            }%",

                        fontSize =
                            20.sp
                    )


                    Spacer(
                        modifier =
                            Modifier.height(
                                24.dp
                            )
                    )


                    Text(

                        text =
                            "Class Probabilities",

                        fontSize =
                            21.sp,

                        fontWeight =
                            FontWeight.Bold
                    )


                    Spacer(
                        modifier =
                            Modifier.height(
                                18.dp
                            )
                    )


                    ProbabilityRow(
                        "AI Generated",
                        getProbability(
                            prediction,
                            0
                        )
                    )


                    Spacer(
                        modifier =
                            Modifier.height(
                                12.dp
                            )
                    )


                    ProbabilityRow(
                        "Deepfake",
                        getProbability(
                            prediction,
                            1
                        )
                    )


                    Spacer(
                        modifier =
                            Modifier.height(
                                12.dp
                            )
                    )


                    ProbabilityRow(
                        "Manipulated",
                        getProbability(
                            prediction,
                            2
                        )
                    )


                    Spacer(
                        modifier =
                            Modifier.height(
                                12.dp
                            )
                    )


                    ProbabilityRow(
                        "Real",
                        getProbability(
                            prediction,
                            3
                        )
                    )
                }
            }


            Spacer(
                modifier =
                    Modifier.height(
                        24.dp
                    )
            )


            Text(

                text =
                    getResultMessage(
                        prediction.label
                    ),

                fontSize =
                    16.sp,

                modifier =
                    Modifier.fillMaxWidth()
            )


            Spacer(
                modifier =
                    Modifier.height(
                        30.dp
                    )
            )
        }
    }
}


/*
 * ================================================================
 * SAFE PROBABILITY ACCESS
 * ================================================================
 */

fun getProbability(
    prediction:
    TruthLensModel.Prediction,
    index: Int
): Float {

    return if (
        index >= 0 &&
        index <
        prediction.probabilities.size
    ) {

        prediction.probabilities[index]

    } else {

        0f
    }
}


/*
 * ================================================================
 * PROBABILITY ROW
 * ================================================================
 */

@androidx.compose.runtime.Composable
fun ProbabilityRow(

    label: String,

    probability: Float
) {

    Text(

        text =
            "$label: ${
                String.format(
                    "%.2f",
                    probability * 100f
                )
            }%",

        fontSize =
            17.sp
    )
}


/*
 * ================================================================
 * FRIENDLY LABEL
 * ================================================================
 */

fun formatLabel(
    label: String
): String {

    return when (label) {

        "AI_Generated" ->
            "AI Generated"

        "Deepfake" ->
            "Deepfake"

        "Manipulated" ->
            "Manipulated"

        "Real" ->
            "Real"

        else ->
            label.replace(
                "_",
                " "
            )
    }
}


/*
 * ================================================================
 * RESULT EXPLANATION
 * ================================================================
 */

fun getResultMessage(
    label: String
): String {

    return when (label) {

        "AI_Generated" ->
            "The model predicts that this image is likely to be AI-generated."

        "Deepfake" ->
            "The model predicts that this image shows characteristics associated with a deepfake."

        "Manipulated" ->
            "The model predicts that this image may have been digitally manipulated."

        "Real" ->
            "The model predicts that this image is likely to be authentic."

        else ->
            "The model has completed the authenticity analysis."
    }
}