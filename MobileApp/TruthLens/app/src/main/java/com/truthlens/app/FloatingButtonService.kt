package com.truthlens.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.drawable.GradientDrawable
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
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.TextView

import androidx.core.app.NotificationCompat

import java.io.File
import java.io.FileOutputStream

import kotlin.math.abs


/*
 * ====================================================================
 * FLOATING BUTTON SERVICE
 * ====================================================================
 *
 * STEP 2 of the floating-button feature.
 *
 * Responsible for:
 *   - showing a small draggable "TL" button over other apps
 *   - holding a MediaProjection session that is granted ONCE (when
 *     the button is enabled) and reused for every tap, so tapping
 *     the button never shows a permission dialog and never has to
 *     bring TruthLens's own UI in front of whatever the user is
 *     trying to verify
 *   - on each tap, saving the most recently captured frame and
 *     broadcasting it using ScreenCaptureService's own
 *     ACTION_CAPTURE_COMPLETE / EXTRA_IMAGE_PATH / EXTRA_ERROR
 *     constants, so MainActivity's existing receiver (unchanged)
 *     picks it up exactly like it does for the manual "Capture
 *     Screen" button. An extra flag (EXTRA_AUTO_ANALYZE) tells that
 *     receiver to run the model automatically for this capture.
 *
 * IMPORTANT DESIGN NOTE:
 * A single MediaProjection permission grant is a full *session* --
 * it can back many captures. We keep the MediaProjection,
 * ImageReader, and VirtualDisplay alive for as long as the floating
 * button is enabled, instead of tearing them down after one frame
 * (which is what ScreenCaptureService correctly does for its
 * one-shot "Capture Screen" button). This is what makes tap-to-
 * capture instant and non-disruptive.
 *
 * ScreenCaptureService itself is completely untouched by this file.
 */

class FloatingButtonService : Service() {

    companion object {

        private const val TAG =
            "TruthLensFloatingBtn"


        /*
         * ========================================================
         * ACTIONS
         * ========================================================
         */

        const val ACTION_START_SESSION =
            "com.truthlens.app.FLOATING_START_SESSION"

        const val ACTION_STOP_OVERLAY =
            "com.truthlens.app.STOP_OVERLAY"

        const val ACTION_FLOATING_BUTTON_TAPPED =
            "com.truthlens.app.FLOATING_BUTTON_TAPPED"


        /*
         * ========================================================
         * EXTRAS
         * ========================================================
         *
         * Kept separate from ScreenCaptureService's own
         * EXTRA_RESULT_CODE / EXTRA_RESULT_DATA so the two capture
         * paths can never be confused with each other.
         */

        const val EXTRA_PROJECTION_RESULT_CODE =
            "floating_projection_result_code"

        const val EXTRA_PROJECTION_RESULT_DATA =
            "floating_projection_result_data"

        /*
         * Read by MainActivity's screenCaptureReceiver to decide
         * whether to automatically analyze this particular capture.
         * Absent (or false) for anything ScreenCaptureService sends,
         * so the manual "Capture Screen" button's behavior is
         * completely unaffected.
         */
        const val EXTRA_AUTO_ANALYZE =
            "auto_analyze"


        /*
         * ========================================================
         * NOTIFICATION
         * ========================================================
         */

        private const val CHANNEL_ID =
            "truthlens_floating_button"

        private const val NOTIFICATION_ID =
            1002


        /*
         * ========================================================
         * DRAG VS. TAP
         * ========================================================
         *
         * If the finger moves more than this many pixels between
         * ACTION_DOWN and ACTION_UP, treat it as a drag, not a tap.
         */

        private const val CLICK_DRAG_TOLERANCE_PX =
            12


        /*
         * ========================================================
         * CAPTURE TIMING
         * ========================================================
         *
         * How long to wait, after hiding the floating button, before
         * grabbing the cached frame. Needs to be long enough for the
         * compositor to redraw without the button AND for our
         * ImageReader listener to receive and decode that new frame.
         */

        private const val HIDE_BUTTON_CAPTURE_DELAY_MS =
            250L
    }


    /*
     * ============================================================
     * OVERLAY OBJECTS
     * ============================================================
     */

    private var windowManager:
            WindowManager? = null

    private var floatingView:
            View? = null

    private var layoutParams:
            WindowManager.LayoutParams? = null


    /*
     * ============================================================
     * PERSISTENT CAPTURE SESSION OBJECTS
     * ============================================================
     */

    private var mediaProjection:
            MediaProjection? = null

    private var virtualDisplay:
            VirtualDisplay? = null

    private var imageReader:
            ImageReader? = null

    private var isCaptureSessionActive =
        false

    private var screenWidth = 0
    private var screenHeight = 0
    private var screenDensity = 0


    /*
     * ============================================================
     * LATEST FRAME CACHE
     * ============================================================
     *
     * The ImageReader listener continuously replaces this with the
     * most recent frame it decoded, so a tap can grab an
     * up-to-date screenshot instantly instead of waiting on a new
     * frame to arrive. All access happens on `handler` (the main
     * thread), so no extra synchronization is needed.
     */

    private var latestFrameBitmap:
            Bitmap? = null


    /*
     * ============================================================
     * HANDLER
     * ============================================================
     */

    private val handler =
        Handler(Looper.getMainLooper())


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
                    "MediaProjection stopped -- ending floating capture session"
                )

                /*
                 * The projection can stop on its own (system revoke,
                 * user dismissed the screen-recording indicator,
                 * etc.). Without it the floating button can no
                 * longer capture, so shut everything down.
                 */

                stopSelf()

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

        if (
            Build.VERSION.SDK_INT >=
            Build.VERSION_CODES.UPSIDE_DOWN_CAKE
        ) {

            startForeground(
                NOTIFICATION_ID,
                createNotification(),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
            )

        } else {

            startForeground(
                NOTIFICATION_ID,
                createNotification()
            )
        }

        addFloatingButton()
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

        when (intent?.action) {

            ACTION_STOP_OVERLAY -> {

                Log.d(
                    TAG,
                    "Stop requested"
                )

                stopSelf()
            }

            ACTION_START_SESSION -> {

                val resultCode =
                    intent.getIntExtra(
                        EXTRA_PROJECTION_RESULT_CODE,
                        Int.MIN_VALUE
                    )

                val resultData: Intent? =
                    if (
                        Build.VERSION.SDK_INT >=
                        Build.VERSION_CODES.TIRAMISU
                    ) {

                        intent.getParcelableExtra(
                            EXTRA_PROJECTION_RESULT_DATA,
                            Intent::class.java
                        )

                    } else {

                        @Suppress("DEPRECATION")
                        intent.getParcelableExtra(
                            EXTRA_PROJECTION_RESULT_DATA
                        )
                    }

                if (
                    resultCode != android.app.Activity.RESULT_OK ||
                    resultData == null
                ) {

                    Log.e(
                        TAG,
                        "Invalid MediaProjection permission for floating session"
                    )

                    stopSelf()

                } else {

                    startCaptureSession(
                        resultCode,
                        resultData
                    )
                }
            }
        }

        /*
         * START_STICKY: if Android kills this service under memory
         * pressure, try to restart it. Note that a restart WITHOUT
         * a fresh ACTION_START_SESSION intent will bring the overlay
         * back but not the capture session -- MainActivity re-grants
         * that the next time the user enables the floating button.
         */

        return START_STICKY
    }


    /*
     * ============================================================
     * START PERSISTENT CAPTURE SESSION
     * ============================================================
     */

    private fun startCaptureSession(
        resultCode: Int,
        resultData: Intent
    ) {

        if (isCaptureSessionActive) {

            Log.w(
                TAG,
                "Capture session already active"
            )

            return
        }

        try {

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

                screenWidth = bounds.width()
                screenHeight = bounds.height()

            } else {

                @Suppress("DEPRECATION")
                val metrics =
                    android.util.DisplayMetrics()

                @Suppress("DEPRECATION")
                windowManager
                    .defaultDisplay
                    .getRealMetrics(metrics)

                screenWidth = metrics.widthPixels
                screenHeight = metrics.heightPixels
            }

            screenDensity =
                resources.displayMetrics.densityDpi

            val projectionManager =
                getSystemService(
                    Context.MEDIA_PROJECTION_SERVICE
                ) as MediaProjectionManager

            mediaProjection =
                projectionManager.getMediaProjection(
                    resultCode,
                    resultData
                )

            if (mediaProjection == null) {

                Log.e(
                    TAG,
                    "Unable to create MediaProjection for floating session"
                )

                stopSelf()
                return
            }

            mediaProjection?.registerCallback(
                projectionCallback,
                handler
            )

            imageReader =
                ImageReader.newInstance(
                    screenWidth,
                    screenHeight,
                    PixelFormat.RGBA_8888,
                    2
                )

            imageReader?.setOnImageAvailableListener(
                { reader ->
                    cacheLatestFrame(reader)
                },
                handler
            )

            virtualDisplay =
                mediaProjection?.createVirtualDisplay(
                    "TruthLensFloatingCapture",
                    screenWidth,
                    screenHeight,
                    screenDensity,
                    DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                    imageReader?.surface,
                    null,
                    handler
                )

            if (virtualDisplay == null) {

                Log.e(
                    TAG,
                    "Unable to create virtual display for floating session"
                )

                stopSelf()
                return
            }

            isCaptureSessionActive = true

            Log.d(
                TAG,
                "Floating capture session started"
            )

        } catch (e: Exception) {

            Log.e(
                TAG,
                "Failed to start floating capture session",
                e
            )

            stopSelf()
        }
    }


    /*
     * ============================================================
     * CACHE THE LATEST FRAME
     * ============================================================
     *
     * Runs every time the mirrored display produces a new frame.
     * We immediately decode it to a Bitmap and close the Image (an
     * Image must not be held onto), so a tap can use the cached
     * Bitmap right away without waiting on the capture pipeline.
     */

    private fun cacheLatestFrame(reader: ImageReader) {

        val image: Image? =
            try {

                reader.acquireLatestImage()

            } catch (e: Exception) {

                Log.e(
                    TAG,
                    "Unable to acquire floating frame",
                    e
                )

                null
            }

        if (image == null) {
            return
        }

        try {

            val bitmap =
                imageToBitmap(image)

            latestFrameBitmap?.recycle()

            latestFrameBitmap = bitmap

        } catch (e: Exception) {

            Log.e(
                TAG,
                "Failed to decode floating frame",
                e
            )

        } finally {

            image.close()
        }
    }


    /*
     * ============================================================
     * CONVERT AN Image TO A CROPPED Bitmap
     * ============================================================
     */

    private fun imageToBitmap(
        image: Image
    ): Bitmap {

        val plane = image.planes[0]

        val buffer = plane.buffer

        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride

        val rowPadding =
            rowStride - pixelStride * image.width

        val bitmapWidth =
            image.width + rowPadding / pixelStride

        val rawBitmap =
            Bitmap.createBitmap(
                bitmapWidth,
                image.height,
                Bitmap.Config.ARGB_8888
            )

        buffer.rewind()

        rawBitmap.copyPixelsFromBuffer(buffer)

        return if (bitmapWidth != image.width) {

            val cropped =
                Bitmap.createBitmap(
                    rawBitmap,
                    0,
                    0,
                    image.width,
                    image.height
                )

            rawBitmap.recycle()

            cropped

        } else {

            rawBitmap
        }
    }


    /*
     * ============================================================
     * BUILD & ATTACH THE OVERLAY VIEW
     * ============================================================
     */

    private fun addFloatingButton() {

        windowManager =
            getSystemService(
                Context.WINDOW_SERVICE
            ) as WindowManager


        val button =
            TextView(this).apply {

                text = "TL"

                setTextColor(
                    Color.WHITE
                )

                textSize = 16f

                gravity =
                    Gravity.CENTER

                background =
                    GradientDrawable().apply {

                        shape =
                            GradientDrawable.OVAL

                        setColor(
                            0xFF9A5F00.toInt()
                        )
                    }
            }


        val sizePx =
            (56 * resources.displayMetrics.density).toInt()


        val overlayType =
            if (
                Build.VERSION.SDK_INT >=
                Build.VERSION_CODES.O
            ) {

                WindowManager.LayoutParams
                    .TYPE_APPLICATION_OVERLAY

            } else {

                @Suppress("DEPRECATION")
                WindowManager.LayoutParams
                    .TYPE_PHONE
            }


        val params =
            WindowManager.LayoutParams(

                sizePx,
                sizePx,

                overlayType,

                WindowManager.LayoutParams
                    .FLAG_NOT_FOCUSABLE or
                        WindowManager.LayoutParams
                            .FLAG_LAYOUT_NO_LIMITS,

                PixelFormat.TRANSLUCENT

            ).apply {

                gravity =
                    Gravity.TOP or Gravity.START

                x = 0
                y = 300
            }


        layoutParams = params


        /*
         * ========================================================
         * DRAG-TO-MOVE / TAP-TO-ACT TOUCH HANDLING
         * ========================================================
         */

        var initialX = 0
        var initialY = 0
        var initialTouchX = 0f
        var initialTouchY = 0f
        var isDragging = false

        button.setOnTouchListener { view, event ->

            when (event.action) {

                MotionEvent.ACTION_DOWN -> {

                    initialX = params.x
                    initialY = params.y

                    initialTouchX = event.rawX
                    initialTouchY = event.rawY

                    isDragging = false

                    true
                }

                MotionEvent.ACTION_MOVE -> {

                    val dx =
                        (event.rawX - initialTouchX).toInt()

                    val dy =
                        (event.rawY - initialTouchY).toInt()

                    if (
                        abs(dx) > CLICK_DRAG_TOLERANCE_PX ||
                        abs(dy) > CLICK_DRAG_TOLERANCE_PX
                    ) {

                        isDragging = true
                    }

                    params.x = initialX + dx
                    params.y = initialY + dy

                    try {

                        windowManager?.updateViewLayout(
                            view,
                            params
                        )

                    } catch (e: Exception) {

                        Log.e(
                            TAG,
                            "Failed to update overlay position",
                            e
                        )
                    }

                    true
                }

                MotionEvent.ACTION_UP -> {

                    if (!isDragging) {

                        onFloatingButtonTapped()
                    }

                    true
                }

                else -> false
            }
        }

        floatingView = button

        try {

            windowManager?.addView(
                button,
                params
            )

            Log.d(
                TAG,
                "Floating button added"
            )

        } catch (e: Exception) {

            /*
             * Most commonly happens if the overlay permission was
             * revoked after this service started.
             */

            Log.e(
                TAG,
                "Failed to add floating button",
                e
            )

            stopSelf()
        }
    }


    /*
     * ============================================================
     * REMOVE OVERLAY VIEW
     * ============================================================
     */

    private fun removeFloatingButton() {

        try {

            floatingView?.let {
                windowManager?.removeView(it)
            }

        } catch (e: Exception) {

            Log.e(
                TAG,
                "Failed to remove floating button",
                e
            )
        }

        floatingView = null
        layoutParams = null
        windowManager = null
    }


    /*
     * ============================================================
     * BUTTON TAPPED
     * ============================================================
     *
     * Uses the already-cached latest frame -- no new permission
     * dialog, no bringing any Activity to the front, no visible
     * disruption to whatever app the user is currently looking at.
     */

    private fun onFloatingButtonTapped() {

        Log.d(
            TAG,
            "Floating button tapped"
        )

        // Kept for anyone else listening (e.g. for debugging/testing).
        sendBroadcast(
            Intent(ACTION_FLOATING_BUTTON_TAPPED).apply {
                setPackage(packageName)
            }
        )

        if (!isCaptureSessionActive) {

            sendFailure(
                "Floating capture session is not active. " +
                        "Try disabling and re-enabling the floating button."
            )

            return
        }

        /*
         * IMPORTANT: the floating button is itself an overlay window,
         * so MediaProjection's mirrored VirtualDisplay captures it
         * too -- without this, the "TL" badge ends up baked into
         * every captured (and analyzed) image.
         *
         * Fix: hide the button, give the compositor + our capture
         * pipeline a brief moment to produce a fresh frame without
         * it, then grab that frame and restore the button.
         */

        floatingView?.visibility =
            View.INVISIBLE

        handler.postDelayed(
            {
                captureCleanFrame()
            },
            HIDE_BUTTON_CAPTURE_DELAY_MS
        )
    }


    /*
     * ============================================================
     * CAPTURE THE FRAME ONCE THE BUTTON IS HIDDEN
     * ============================================================
     */

    private fun captureCleanFrame() {

        val bitmap = latestFrameBitmap

        // Restore the button immediately -- we already have (or are
        // about to save) the frame, so there's no reason to keep it
        // hidden any longer than necessary.
        floatingView?.visibility =
            View.VISIBLE

        if (bitmap == null) {

            sendFailure(
                "No screen frame captured yet. Try tapping again in a moment."
            )

            return
        }

        val imagePath =
            saveBitmapToFile(bitmap)

        if (imagePath != null) {

            sendCaptureComplete(imagePath)

        } else {

            sendFailure(
                "Unable to save captured image."
            )
        }
    }


    /*
     * ============================================================
     * SAVE BITMAP TO A FILE
     * ============================================================
     */

    private fun saveBitmapToFile(
        bitmap: Bitmap
    ): String? {

        return try {

            val directory =
                File(
                    getExternalFilesDir(null),
                    "captures"
                )

            if (!directory.exists()) {
                directory.mkdirs()
            }

            val file =
                File(
                    directory,
                    "truthlens_floating_" +
                            System.currentTimeMillis() +
                            ".jpg"
                )

            FileOutputStream(file).use { outputStream ->

                bitmap.compress(
                    Bitmap.CompressFormat.JPEG,
                    95,
                    outputStream
                )
            }

            Log.d(
                TAG,
                "Floating capture saved: ${file.absolutePath}"
            )

            file.absolutePath

        } catch (e: Exception) {

            Log.e(
                TAG,
                "Floating capture save failed",
                e
            )

            null
        }
    }


    /*
     * ============================================================
     * SEND SUCCESS BROADCAST
     * ============================================================
     *
     * Reuses ScreenCaptureService's own action/extra so
     * MainActivity's existing receiver handles this without any
     * changes to how it processes a successful capture.
     */

    private fun sendCaptureComplete(
        imagePath: String
    ) {

        val intent =
            Intent(
                ScreenCaptureService.ACTION_CAPTURE_COMPLETE
            ).apply {

                setPackage(packageName)

                putExtra(
                    ScreenCaptureService.EXTRA_IMAGE_PATH,
                    imagePath
                )

                putExtra(
                    EXTRA_AUTO_ANALYZE,
                    true
                )
            }

        sendBroadcast(intent)
    }


    /*
     * ============================================================
     * SEND FAILURE BROADCAST
     * ============================================================
     */

    private fun sendFailure(message: String) {

        Log.e(TAG, message)

        val intent =
            Intent(
                ScreenCaptureService.ACTION_CAPTURE_COMPLETE
            ).apply {

                setPackage(packageName)

                putExtra(
                    ScreenCaptureService.EXTRA_ERROR,
                    message
                )
            }

        sendBroadcast(intent)
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
                    "TruthLens Floating Button",
                    NotificationManager
                        .IMPORTANCE_LOW
                )

            channel.description =
                "Shown while the TruthLens floating button is active."

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

        val stopIntent =
            Intent(
                this,
                FloatingButtonService::class.java
            ).apply {

                action = ACTION_STOP_OVERLAY
            }

        val stopPendingIntent =
            PendingIntent.getService(
                this,
                0,
                stopIntent,
                PendingIntent.FLAG_IMMUTABLE or
                        PendingIntent.FLAG_UPDATE_CURRENT
            )

        val openAppIntent =
            Intent(
                this,
                MainActivity::class.java
            ).apply {

                flags =
                    Intent.FLAG_ACTIVITY_NEW_TASK or
                            Intent.FLAG_ACTIVITY_SINGLE_TOP
            }

        val openAppPendingIntent =
            PendingIntent.getActivity(
                this,
                0,
                openAppIntent,
                PendingIntent.FLAG_IMMUTABLE or
                        PendingIntent.FLAG_UPDATE_CURRENT
            )

        return NotificationCompat
            .Builder(this, CHANNEL_ID)
            .setContentTitle("TruthLens")
            .setContentText(
                "Floating button is active. Tap here to view results."
            )
            .setSmallIcon(
                android.R.drawable.ic_menu_camera
            )
            .setOngoing(true)
            .setPriority(
                NotificationCompat.PRIORITY_LOW
            )
            .setContentIntent(openAppPendingIntent)
            .addAction(
                android.R.drawable.ic_menu_close_clear_cancel,
                "Stop",
                stopPendingIntent
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
            "FloatingButtonService destroyed"
        )

        removeFloatingButton()

        try {
            virtualDisplay?.release()
        } catch (_: Exception) {
        }
        virtualDisplay = null

        try {
            imageReader?.close()
        } catch (_: Exception) {
        }
        imageReader = null

        try {
            mediaProjection?.unregisterCallback(projectionCallback)
        } catch (_: Exception) {
        }

        try {
            mediaProjection?.stop()
        } catch (_: Exception) {
        }
        mediaProjection = null

        latestFrameBitmap?.recycle()
        latestFrameBitmap = null

        isCaptureSessionActive = false

        super.onDestroy()
    }
}