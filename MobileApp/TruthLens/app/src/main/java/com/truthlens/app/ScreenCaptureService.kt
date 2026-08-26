package com.truthlens.app

import android.app.Activity
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import android.view.WindowManager

import androidx.core.app.NotificationCompat


import java.io.File
import java.io.FileOutputStream


class ScreenCaptureService : Service() {

    companion object {

        private const val TAG =
            "TruthLensCapture"

        /*
         * ========================================================
         * ACTIONS
         * ========================================================
         */

        const val ACTION_START_CAPTURE =
            "com.truthlens.app.START_CAPTURE"

        const val ACTION_CAPTURE_COMPLETE =
            "com.truthlens.app.CAPTURE_COMPLETE"


        /*
         * ========================================================
         * EXTRAS
         * ========================================================
         */

        const val EXTRA_RESULT_CODE =
            "result_code"

        const val EXTRA_RESULT_DATA =
            "result_data"

        const val EXTRA_IMAGE_PATH =
            "image_path"

        const val EXTRA_ERROR =
            "capture_error"


        /*
         * ========================================================
         * NOTIFICATION
         * ========================================================
         */

        private const val CHANNEL_ID =
            "truthlens_screen_capture"

        private const val NOTIFICATION_ID =
            1001


        /*
         * ========================================================
         * CAPTURE DELAY
         * ========================================================
         *
         * This gives the user time to leave TruthLens and open
         * the media that needs to be verified.
         */

        private const val CAPTURE_DELAY_MS =
            2500L
    }


    /*
     * ============================================================
     * ANDROID SCREEN CAPTURE OBJECTS
     * ============================================================
     */

    private var mediaProjection:
            MediaProjection? = null

    private var virtualDisplay:
            VirtualDisplay? = null

    private var imageReader:
            ImageReader? = null


    /*
     * ============================================================
     * HANDLER
     * ============================================================
     */

    private val handler =
        Handler(Looper.getMainLooper())


    /*
     * ============================================================
     * SCREEN INFORMATION
     * ============================================================
     */

    private var screenWidth =
        0

    private var screenHeight =
        0

    private var screenDensity =
        0


    /*
     * ============================================================
     * CAPTURE STATE
     * ============================================================
     */

    private var captureArmed =
        false

    private var imageCaptured =
        false

    private var isCleaningUp =
        false


    /*
     * ============================================================
     * MEDIA PROJECTION CALLBACK
     * ============================================================
     */

    private val projectionCallback =
        object : MediaProjection.Callback() {

            override fun onStop() {

                Log.d(
                    TAG,
                    "MediaProjection stopped"
                )

                cleanup()

                super.onStop()
            }
        }


    /*
     * ============================================================
     * SERVICE CREATED
     * ============================================================
     */

    override fun onCreate() {
        super.onCreate()

        createNotificationChannel()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                createNotification(),
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
            )
        } else {
            startForeground(
                NOTIFICATION_ID,
                createNotification()
            )
        }
    }

    /*
     * ============================================================
     * SERVICE STARTED
     * ============================================================
     */

    override fun onStartCommand(
        intent: Intent?,
        flags: Int,
        startId: Int
    ): Int {

        Log.d(
            TAG,
            "ScreenCaptureService started"
        )


        if (intent == null) {

            Log.e(
                TAG,
                "Service intent is null"
            )

            sendFailure(
                "Invalid screen capture request"
            )

            stopSelf()

            return START_NOT_STICKY
        }


        /*
         * Make sure this is the expected action.
         */

        if (
            intent.action !=
            ACTION_START_CAPTURE
        ) {

            Log.e(
                TAG,
                "Unexpected service action: ${intent.action}"
            )

            sendFailure(
                "Invalid capture action"
            )

            stopSelf()

            return START_NOT_STICKY
        }


        /*
         * Get MediaProjection permission result.
         */

        /*
         * IMPORTANT: Activity.RESULT_OK is itself -1, so we can't use
         * -1 as the "extra missing" sentinel below -- doing so made
         * every *successful* grant (resultCode == RESULT_OK == -1)
         * look identical to a missing extra, and the request was
         * rejected as "Invalid screen capture permission" every time.
         * Int.MIN_VALUE can never collide with a real Activity result
         * code, so it's safe to use as the "not present" default.
         */

        val resultCode =
            intent.getIntExtra(
                EXTRA_RESULT_CODE,
                Int.MIN_VALUE
            )


        /*
         * Get permission Intent.
         */

        val resultData: Intent? =
            if (
                Build.VERSION.SDK_INT >=
                Build.VERSION_CODES.TIRAMISU
            ) {

                intent.getParcelableExtra(
                    EXTRA_RESULT_DATA,
                    Intent::class.java
                )

            } else {

                @Suppress("DEPRECATION")

                intent.getParcelableExtra(
                    EXTRA_RESULT_DATA
                )
            }


        /*
         * Validate permission information.
         */

        if (
            resultCode != Activity.RESULT_OK ||
            resultData == null
        ) {

            Log.e(
                TAG,
                "Invalid MediaProjection permission data " +
                        "(resultCode=$resultCode, resultData=$resultData)"
            )

            sendFailure(
                "Invalid screen capture permission"
            )

            stopSelf()

            return START_NOT_STICKY
        }


        /*
         * Start screen capture.
         */

        startScreenCapture(
            resultCode,
            resultData
        )


        return START_NOT_STICKY
    }


    /*
     * ============================================================
     * START SCREEN CAPTURE
     * ============================================================
     */

    private fun startScreenCapture(
        resultCode: Int,
        resultData: Intent
    ) {

        try {

            /*
             * Prevent duplicate capture.
             */

            if (
                mediaProjection != null
            ) {

                Log.w(
                    TAG,
                    "Capture already running"
                )

                return
            }


            /*
             * Get display metrics.
             */

            val windowManager =
                getSystemService(
                    Context.WINDOW_SERVICE
                ) as WindowManager


            if (
                Build.VERSION.SDK_INT >=
                Build.VERSION_CODES.R
            ) {

                val bounds =
                    windowManager
                        .currentWindowMetrics
                        .bounds

                screenWidth =
                    bounds.width()

                screenHeight =
                    bounds.height()

            } else {

                @Suppress(
                    "DEPRECATION"
                )

                val metrics =
                    android.util.DisplayMetrics()

                @Suppress(
                    "DEPRECATION"
                )

                windowManager
                    .defaultDisplay
                    .getRealMetrics(
                        metrics
                    )

                screenWidth =
                    metrics.widthPixels

                screenHeight =
                    metrics.heightPixels
            }


            screenDensity =
                resources
                    .displayMetrics
                    .densityDpi


            Log.d(
                TAG,
                "Screen size: " +
                        "${screenWidth}x${screenHeight}"
            )


            /*
             * Get MediaProjection manager.
             */

            val projectionManager =
                getSystemService(
                    Context.MEDIA_PROJECTION_SERVICE
                ) as MediaProjectionManager


            /*
             * Create MediaProjection.
             */

            mediaProjection =
                projectionManager
                    .getMediaProjection(
                        resultCode,
                        resultData
                    )


            if (
                mediaProjection == null
            ) {

                sendFailure(
                    "Unable to create MediaProjection"
                )

                stopSelf()

                return
            }


            /*
             * Register callback.
             */

            mediaProjection?.registerCallback(
                projectionCallback,
                handler
            )


            /*
             * Create ImageReader.
             */

            createImageReader()


            /*
             * Create virtual display.
             */

            createVirtualDisplay()


            /*
             * Wait before allowing capture.
             *
             * This is important because the first frames may
             * still show the TruthLens application itself.
             */

            captureArmed = false

            handler.postDelayed({

                if (
                    !imageCaptured &&
                    !isCleaningUp
                ) {

                    captureArmed = true

                    Log.d(
                        TAG,
                        "Capture armed"
                    )
                }

            }, CAPTURE_DELAY_MS)


        } catch (e: Exception) {

            Log.e(
                TAG,
                "Failed to start screen capture",
                e
            )

            sendFailure(
                e.message
                    ?: "Screen capture failed"
            )

            cleanup()
        }
    }


    /*
     * ============================================================
     * CREATE IMAGE READER
     * ============================================================
     */

    private fun createImageReader() {

        imageReader =
            ImageReader.newInstance(
                screenWidth,
                screenHeight,
                PixelFormat.RGBA_8888,
                2
            )


        imageReader?.setOnImageAvailableListener(
            { reader ->

                /*
                 * Ignore frames until the delay is finished.
                 */

                if (!captureArmed) {

                    try {

                        reader
                            .acquireLatestImage()
                            ?.close()

                    } catch (
                        _: Exception
                    ) {
                    }

                    return@setOnImageAvailableListener
                }


                /*
                 * Only capture one image.
                 */

                if (imageCaptured) {
                    return@setOnImageAvailableListener
                }


                val image: Image? = try {

                    reader.acquireLatestImage()

                } catch (e: Exception) {

                    Log.e(
                        TAG,
                        "Unable to acquire image",
                        e
                    )

                    null
                }


                if (image == null) {
                    return@setOnImageAvailableListener
                }


                try {

                    imageCaptured = true

                    Log.d(
                        TAG,
                        "Screen image received"
                    )


                    val imagePath =
                        saveImage(image)


                    if (
                        imagePath != null
                    ) {

                        sendCaptureComplete(
                            imagePath
                        )

                    } else {

                        sendFailure(
                            "Unable to save captured image"
                        )
                    }


                } catch (e: Exception) {

                    Log.e(
                        TAG,
                        "Failed to process image",
                        e
                    )

                    sendFailure(
                        e.message
                            ?: "Failed to process image"
                    )

                } finally {

                    image.close()

                    cleanup()
                }

            },
            handler
        )
    }


    /*
     * ============================================================
     * CREATE VIRTUAL DISPLAY
     * ============================================================
     */

    private fun createVirtualDisplay() {

        val surface =
            imageReader?.surface


        if (surface == null) {

            sendFailure(
                "ImageReader surface is unavailable"
            )

            cleanup()

            return
        }


        virtualDisplay =
            mediaProjection?.createVirtualDisplay(

                "TruthLensScreenCapture",

                screenWidth,

                screenHeight,

                screenDensity,

                DisplayManager
                    .VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,

                surface,

                null,

                handler
            )


        if (
            virtualDisplay == null
        ) {

            sendFailure(
                "Unable to create virtual display"
            )

            cleanup()

            return
        }


        Log.d(
            TAG,
            "Virtual display created"
        )
    }


    /*
     * ============================================================
     * SAVE IMAGE
     * ============================================================
     */

    private fun saveImage(
        image: Image
    ): String? {

        try {

            val planes =
                image.planes


            if (
                planes.isEmpty()
            ) {

                return null
            }


            val plane =
                planes[0]


            val buffer =
                plane.buffer


            val pixelStride =
                plane.pixelStride

            val rowStride =
                plane.rowStride


            val rowPadding =
                rowStride -
                        pixelStride *
                        image.width


            val bitmapWidth =
                image.width +
                        rowPadding /
                        pixelStride


            /*
             * Create bitmap large enough to contain
             * the row padding.
             */

            val bitmap =
                Bitmap.createBitmap(
                    bitmapWidth,
                    image.height,
                    Bitmap.Config.ARGB_8888
                )


            buffer.rewind()

            bitmap.copyPixelsFromBuffer(
                buffer
            )


            /*
             * Crop away row padding.
             */

            val croppedBitmap =
                if (
                    bitmapWidth !=
                    image.width
                ) {

                    Bitmap.createBitmap(
                        bitmap,
                        0,
                        0,
                        image.width,
                        image.height
                    )

                } else {

                    bitmap
                }


            /*
             * Save inside app-specific external files.
             */

            val directory =
                File(
                    getExternalFilesDir(
                        null
                    ),
                    "captures"
                )


            if (
                !directory.exists()
            ) {

                directory.mkdirs()
            }


            val file =
                File(
                    directory,
                    "truthlens_capture_" +
                            System.currentTimeMillis() +
                            ".jpg"
                )


            FileOutputStream(
                file
            ).use { outputStream ->

                croppedBitmap.compress(
                    Bitmap.CompressFormat.JPEG,
                    95,
                    outputStream
                )
            }


            /*
             * Release temporary bitmaps.
             */

            if (
                croppedBitmap !== bitmap
            ) {

                croppedBitmap.recycle()
            }

            bitmap.recycle()


            Log.d(
                TAG,
                "Image saved: ${file.absolutePath}"
            )


            return file.absolutePath

        } catch (e: Exception) {

            Log.e(
                TAG,
                "Image save failed",
                e
            )

            return null
        }
    }


    /*
     * ============================================================
     * SEND SUCCESS BROADCAST
     * ============================================================
     */

    private fun sendCaptureComplete(
        imagePath: String
    ) {

        Log.d(
            TAG,
            "Sending captured image to MainActivity"
        )


        val intent =
            Intent(
                ACTION_CAPTURE_COMPLETE
            ).apply {

                setPackage(
                    packageName
                )

                putExtra(
                    EXTRA_IMAGE_PATH,
                    imagePath
                )
            }


        sendBroadcast(
            intent
        )
    }


    /*
     * ============================================================
     * SEND FAILURE BROADCAST
     * ============================================================
     */

    private fun sendFailure(
        message: String
    ) {

        Log.e(
            TAG,
            message
        )


        val intent =
            Intent(
                ACTION_CAPTURE_COMPLETE
            ).apply {

                setPackage(
                    packageName
                )

                putExtra(
                    EXTRA_ERROR,
                    message
                )
            }


        sendBroadcast(
            intent
        )
    }


    /*
     * ============================================================
     * CLEANUP
     * ============================================================
     */

    private fun cleanup() {

        if (isCleaningUp) {
            return
        }


        isCleaningUp = true


        Log.d(
            TAG,
            "Cleaning up screen capture"
        )


        /*
         * Cancel pending capture delay.
         */

        handler.removeCallbacksAndMessages(
            null
        )


        /*
         * Release virtual display.
         */

        try {

            virtualDisplay?.release()

        } catch (
            _: Exception
        ) {
        }

        virtualDisplay = null


        /*
         * Close ImageReader.
         */

        try {

            imageReader?.close()

        } catch (
            _: Exception
        ) {
        }

        imageReader = null


        /*
         * Unregister MediaProjection callback.
         */

        try {

            mediaProjection
                ?.unregisterCallback(
                    projectionCallback
                )

        } catch (
            _: Exception
        ) {
        }


        /*
         * Stop MediaProjection.
         */

        try {

            mediaProjection?.stop()

        } catch (
            _: Exception
        ) {
        }

        mediaProjection = null


        /*
         * Stop service.
         */

        stopSelf()
    }


    /*
     * ============================================================
     * NOTIFICATION CHANNEL
     * ============================================================
     */

    private fun createNotificationChannel() {

        if (
            Build.VERSION.SDK_INT >=
            Build.VERSION_CODES.O
        ) {

            val channel =
                NotificationChannel(
                    CHANNEL_ID,
                    "TruthLens Screen Capture",
                    NotificationManager
                        .IMPORTANCE_LOW
                )


            channel.description =
                "Used while TruthLens captures the screen."


            val manager =
                getSystemService(
                    NotificationManager::class.java
                )


            manager.createNotificationChannel(
                channel
            )
        }
    }


    /*
     * ============================================================
     * NOTIFICATION
     * ============================================================
     */

    private fun createNotification():
            Notification {

        return NotificationCompat
            .Builder(
                this,
                CHANNEL_ID
            )
            .setContentTitle(
                "TruthLens"
            )
            .setContentText(
                "Preparing screen capture..."
            )
            .setSmallIcon(
                android.R.drawable.ic_menu_camera
            )
            .setOngoing(true)
            .setPriority(
                NotificationCompat
                    .PRIORITY_LOW
            )
            .build()
    }


    /*
     * ============================================================
     * BINDING
     * ============================================================
     */

    override fun onBind(
        intent: Intent?
    ): IBinder? {

        return null
    }


    /*
     * ============================================================
     * SERVICE DESTROYED
     * ============================================================
     */

    override fun onDestroy() {

        Log.d(
            TAG,
            "ScreenCaptureService destroyed"
        )


        cleanup()


        super.onDestroy()
    }
}