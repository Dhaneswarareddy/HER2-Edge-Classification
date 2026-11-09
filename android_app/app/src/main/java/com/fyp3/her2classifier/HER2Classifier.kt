package com.fyp3.her2classifier

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

data class ClassificationResult(
    val predictedClassIndex: Int,
    val predictedLabel: String,
    val clinicalStatus: String,
    val clinicalDetail: String,
    val probabilities: FloatArray,
    val inferenceTimeMs: Long,
    val totalTimeMs: Long,
    val threadCount: Int,
    val modelName: String
)

class HER2Classifier(
    private val context: Context,
    private var modelFileName: String = "models/primary_hires_fp16.tflite",
    private var threadCount: Int = 4
) {
    private var interpreter: Interpreter? = null
    private val inputSize = 448
    private val pixelBytes = 4 // float32 = 4 bytes
    private val channelCount = 3
    private val classCount = 4

    val classLabels = arrayOf("Score 0", "Score 1+", "Score 2+", "Score 3+")

    init {
        initInterpreter()
    }

    @Synchronized
    fun setModel(newModelFileName: String, newThreadCount: Int = this.threadCount) {
        if (this.modelFileName != newModelFileName || this.threadCount != newThreadCount) {
            this.modelFileName = newModelFileName
            this.threadCount = newThreadCount
            initInterpreter()
        }
    }

    @Synchronized
    fun setThreads(newThreadCount: Int) {
        if (this.threadCount != newThreadCount) {
            this.threadCount = newThreadCount
            initInterpreter()
        }
    }

    private fun initInterpreter() {
        interpreter?.close()
        val options = Interpreter.Options().apply {
            setNumThreads(threadCount)
            setUseXNNPACK(true)
        }
        val modelBuffer = loadModelFile(modelFileName)
        interpreter = Interpreter(modelBuffer, options)
    }

    private fun loadModelFile(path: String): ByteBuffer {
        val fileDescriptor: AssetFileDescriptor = context.assets.openFd(path)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel: FileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    @Synchronized
    fun classify(bitmap: Bitmap): ClassificationResult {
        val totalStartTime = SystemClock.uptimeMillis()

        // 1. Resize bitmap to input size (448x448)
        val resizedBitmap = if (bitmap.width != inputSize || bitmap.height != inputSize) {
            Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        } else {
            bitmap
        }

        // 2. Convert to float32 ByteBuffer [1, 448, 448, 3] in range [0.0, 255.0]
        val inputBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * channelCount * pixelBytes).apply {
            order(ByteOrder.nativeOrder())
            rewind()
        }

        val intValues = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)

        for (pixelValue in intValues) {
            // Extract RGB (0..255)
            val r = ((pixelValue shr 16) and 0xFF).toFloat()
            val g = ((pixelValue shr 8) and 0xFF).toFloat()
            val b = (pixelValue and 0xFF).toFloat()

            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }

        // 3. Prepare output buffer [1, 4]
        val outputBuffer = Array(1) { FloatArray(classCount) }

        // 4. Run inference
        val inferStartTime = SystemClock.uptimeMillis()
        interpreter?.run(inputBuffer, outputBuffer)
        val inferEndTime = SystemClock.uptimeMillis()
        val inferenceTimeMs = inferEndTime - inferStartTime
        val totalTimeMs = inferEndTime - totalStartTime

        val probabilities = outputBuffer[0]

        // 5. Determine predicted class (argmax)
        var maxIndex = 0
        var maxProb = probabilities[0]
        for (i in 1 until classCount) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i]
                maxIndex = i
            }
        }

        val predictedLabel = classLabels[maxIndex]
        val (clinicalStatus, clinicalDetail) = when (maxIndex) {
            0 -> Pair(
                "Negative (Score 0)",
                "No HER2 membrane staining or <10% membrane staining. Targeted therapy not indicated."
            )
            1 -> Pair(
                "Negative (Score 1+)",
                "Faint / barely perceptible membrane staining in >10% tumor cells. Targeted therapy not indicated."
            )
            2 -> Pair(
                "Equivocal (Score 2+)",
                "Weak-to-moderate complete membrane staining in >10% tumor cells. Reflex dual-probe FISH/CISH mandatory per ASCO/CAP guidelines."
            )
            3 -> Pair(
                "Positive (Score 3+)",
                "Circumferential, intense membrane staining in >10% tumor cells. Eligible for anti-HER2 targeted therapy (Trastuzumab/Pertuzumab)."
            )
            else -> Pair("Unknown", "Uncertain prediction")
        }

        return ClassificationResult(
            predictedClassIndex = maxIndex,
            predictedLabel = predictedLabel,
            clinicalStatus = clinicalStatus,
            clinicalDetail = clinicalDetail,
            probabilities = probabilities,
            inferenceTimeMs = inferenceTimeMs,
            totalTimeMs = totalTimeMs,
            threadCount = threadCount,
            modelName = if (modelFileName.contains("int8")) "INT8 Quantized (448px)" else "FP16 Primary (448px)"
        )
    }

    fun close() {
        interpreter?.close()
    }
}
