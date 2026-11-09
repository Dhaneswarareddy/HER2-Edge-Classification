package com.fyp3.her2classifier

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.fyp3.her2classifier.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.InputStream
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var classifier: HER2Classifier? = null
    private var currentBitmap: Bitmap? = null

    private var selectedModelPath = "models/primary_hires_fp16.tflite"
    private var selectedThreads = 4

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { handleSelectedImageUri(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        detectHardwarePlatform()
        setupClassifier()
        setupListeners()

        // Load initial reference tile (Score 0) on launch
        loadSampleTile("samples/sample_tile_class_0.png")
    }

    private fun detectHardwarePlatform() {
        val manufacturer = android.os.Build.MANUFACTURER.replaceFirstChar { it.uppercase() }
        val model = android.os.Build.MODEL
        val hardwareText = if (model.contains("RMX2001", ignoreCase = true) || manufacturer.contains("realme", ignoreCase = true)) {
            "Realme 6 (RMX2001) | MediaTek Helio G90T (2× A76 + 6× A55)"
        } else if (model.contains("M35", ignoreCase = true) || model.contains("SM-M356B", ignoreCase = true)) {
            "Samsung Galaxy M35 5G | Exynos 1380 (4× A78 + 4× A55)"
        } else {
            "$manufacturer $model | Edge AI Neural Engine"
        }
        binding.tvHardwareBadge.text = hardwareText
    }

    private fun setupClassifier() {
        try {
            classifier = HER2Classifier(this, selectedModelPath, selectedThreads)
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to initialize classifier: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun setupListeners() {
        // Model Selection Listener
        binding.rgModelPrecision.setOnCheckedChangeListener { _, checkedId ->
            selectedModelPath = when (checkedId) {
                R.id.rbInt8 -> "models/primary_hires_int8.tflite"
                else -> "models/primary_hires_fp16.tflite"
            }
            updateClassifierConfig()
        }

        // Thread Selection Listener
        binding.rgThreads.setOnCheckedChangeListener { _, checkedId ->
            selectedThreads = when (checkedId) {
                R.id.rb1Thread -> 1
                R.id.rb2Threads -> 2
                R.id.rb8Threads -> 8
                else -> 4 // 4 threads optimal for 4x A78 big cores
            }
            updateClassifierConfig()
        }

        // Reference Clinical Tile Buttons
        binding.btnSample0.setOnClickListener { loadSampleTile("samples/sample_tile_class_0.png") }
        binding.btnSample1.setOnClickListener { loadSampleTile("samples/sample_tile_class_1.png") }
        binding.btnSample2.setOnClickListener { loadSampleTile("samples/sample_tile_class_2.png") }
        binding.btnSample3.setOnClickListener { loadSampleTile("samples/sample_tile_class_3.png") }

        // Gallery Picker
        binding.btnPickGallery.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }
    }

    private fun updateClassifierConfig() {
        try {
            classifier?.setModel(selectedModelPath, selectedThreads)
            val modelName = if (selectedModelPath.contains("int8")) "INT8 Quantized (448px)" else "FP16 Primary (448px)"
            binding.tvModelSpec.text = "Model: $modelName • Threads: $selectedThreads • TFLite XNNPACK"
            // Re-run classification on current bitmap if available
            currentBitmap?.let { runInference(it) }
        } catch (e: Exception) {
            Toast.makeText(this, "Error reconfiguring classifier: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun loadSampleTile(assetPath: String) {
        try {
            val inputStream: InputStream = assets.open(assetPath)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream.close()
            currentBitmap = bitmap
            binding.ivTilePreview.setImageBitmap(bitmap)
            runInference(bitmap)
        } catch (e: Exception) {
            Toast.makeText(this, "Could not load sample: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun handleSelectedImageUri(uri: Uri) {
        try {
            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()
            if (bitmap != null) {
                currentBitmap = bitmap
                binding.ivTilePreview.setImageBitmap(bitmap)
                runInference(bitmap)
            }
        } catch (e: Exception) {
            Toast.makeText(this, "Error reading image: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun runInference(bitmap: Bitmap) {
        lifecycleScope.launch(Dispatchers.Default) {
            val result = classifier?.classify(bitmap)
            withContext(Dispatchers.Main) {
                result?.let { updateUiWithResult(it) }
            }
        }
    }

    private fun updateUiWithResult(result: ClassificationResult) {
        // Update Score Badge & Colors
        binding.tvPredictionScore.text = result.predictedLabel
        binding.tvPredictionStatus.text = result.clinicalStatus
        binding.tvClinicalAction.text = result.clinicalDetail

        val badgeColor = when (result.predictedClassIndex) {
            0 -> ContextCompat.getColor(this, R.color.score_0)
            1 -> ContextCompat.getColor(this, R.color.score_1)
            2 -> ContextCompat.getColor(this, R.color.score_2)
            3 -> ContextCompat.getColor(this, R.color.score_3)
            else -> ContextCompat.getColor(this, R.color.primary)
        }
        binding.tvPredictionScore.background.setTint(badgeColor)

        // Update Confidence Bars
        val p = result.probabilities
        val p0 = (p[0] * 100f)
        val p1 = (p[1] * 100f)
        val p2 = (p[2] * 100f)
        val p3 = (p[3] * 100f)

        binding.pbScore0.progress = p0.toInt().coerceIn(0, 100)
        binding.tvScore0.text = String.format(Locale.US, "%.1f%%", p0)

        binding.pbScore1.progress = p1.toInt().coerceIn(0, 100)
        binding.tvScore1.text = String.format(Locale.US, "%.1f%%", p1)

        binding.pbScore2.progress = p2.toInt().coerceIn(0, 100)
        binding.tvScore2.text = String.format(Locale.US, "%.1f%%", p2)

        binding.pbScore3.progress = p3.toInt().coerceIn(0, 100)
        binding.tvScore3.text = String.format(Locale.US, "%.1f%%", p3)

        // Update Latency and Throughput
        val latencyMs = result.inferenceTimeMs.coerceAtLeast(1)
        val throughputFps = 1000.0 / latencyMs
        binding.tvLatency.text = String.format(Locale.US, "%d ms", latencyMs)
        binding.tvThroughput.text = String.format(Locale.US, "%.1f FPS", throughputFps)
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier?.close()
    }
}
