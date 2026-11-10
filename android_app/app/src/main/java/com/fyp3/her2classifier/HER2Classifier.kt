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
    val preprocessTimeMs: Long,
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

        // 1. Validated FYP3 preprocessing, bit-exact with training/evaluation (PIL):
        //    native tile -> LANCZOS 512 -> BILINEAR 448 (see PilResample.kt). Earlier versions used a single
        //    Bitmap.createScaledBitmap(1024 -> 448), which differs from the validated path.
        val w = bitmap.width
        val h = bitmap.height
        val argb = IntArray(w * h)
        bitmap.getPixels(argb, 0, w, 0, 0, w, h)
        val rgb = PilResample.fyp3Preprocess(argb, w, h)

        // 2. float32 tensor [1, 448, 448, 3] in [0, 255] (the model's own layers rescale)
        val inputBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * channelCount * pixelBytes).apply {
            order(ByteOrder.nativeOrder())
            rewind()
        }
        for (v in rgb) inputBuffer.putFloat(v.toFloat())
        inputBuffer.rewind()
        val preprocessTimeMs = SystemClock.uptimeMillis() - totalStartTime

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
                "HER2-negative (IHC 0)",
                "ASCO/CAP: no staining, or incomplete faint/barely perceptible membrane staining in \u226410% of tumour cells."
            )
            1 -> Pair(
                "HER2-negative, HER2-low (IHC 1+)",
                "ASCO/CAP: incomplete faint/barely perceptible membrane staining in >10% of tumour cells. HER2-low status may be relevant for therapy (e.g. trastuzumab deruxtecan); pathologist confirmation required."
            )
            2 -> Pair(
                "Equivocal (IHC 2+)",
                "ASCO/CAP: weak-to-moderate complete membrane staining in >10% of tumour cells. Reflex in situ hybridisation (ISH) required."
            )
            3 -> Pair(
                "HER2-positive (IHC 3+)",
                "ASCO/CAP: complete, intense circumferential membrane staining in >10% of tumour cells."
            )
            else -> Pair("Unknown", "Uncertain prediction")
        }

        return ClassificationResult(
            predictedClassIndex = maxIndex,
            predictedLabel = predictedLabel,
            clinicalStatus = clinicalStatus,
            clinicalDetail = clinicalDetail,
            probabilities = probabilities,
            preprocessTimeMs = preprocessTimeMs,
            inferenceTimeMs = inferenceTimeMs,
            totalTimeMs = totalTimeMs,
            threadCount = threadCount,
            modelName = if (modelFileName.contains("int8")) "INT8 448px (FAILED accuracy gate: speed test only)" else "FP16 448px (validated)"
        )
    }

    fun close() {
        interpreter?.close()
    }
}
