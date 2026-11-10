package com.fyp3.her2classifier

/**
 * Port of Pillow's separable resampler (libImaging/Resample.c, 8-bit fixed-point path), so that the phone reproduces
 * the training/validation preprocessing exactly: native tile -> LANCZOS 512 -> BILINEAR 448
 * (PIL Image.resize, horizontal pass then vertical pass, uint8 between passes).
 * The Python twin experiments_v2/08_edge/pil_resample_port.py is bit-identical to PIL on BCI tiles.
 */
object PilResample {
    private const val PRECISION_BITS = 32 - 8 - 2

    enum class Filter(val support: Double) { BILINEAR(1.0), LANCZOS(3.0) }

    private fun sinc(x0: Double): Double {
        if (x0 == 0.0) return 1.0
        val x = x0 * Math.PI
        return Math.sin(x) / x
    }

    private fun kernel(f: Filter, x: Double): Double = when (f) {
        Filter.BILINEAR -> { val a = if (x < 0) -x else x; if (a < 1.0) 1.0 - a else 0.0 }
        Filter.LANCZOS -> if (x >= -3.0 && x < 3.0) sinc(x) * sinc(x / 3.0) else 0.0
    }

    private class Coeffs(val xmin: IntArray, val xlen: IntArray, val k: IntArray, val ksize: Int)

    private fun coeffs(inSize: Int, outSize: Int, f: Filter): Coeffs {
        val scale = inSize.toDouble() / outSize
        val fscale = if (scale < 1.0) 1.0 else scale
        val support = f.support * fscale
        val ksize = Math.ceil(support).toInt() * 2 + 1
        val xminA = IntArray(outSize); val xlenA = IntArray(outSize); val kk = IntArray(outSize * ksize)
        val tmp = DoubleArray(ksize)
        for (xx in 0 until outSize) {
            val center = (xx + 0.5) * scale
            val ss = 1.0 / fscale
            var xmin = (center - support + 0.5).toInt(); if (xmin < 0) xmin = 0
            var xmax = (center + support + 0.5).toInt(); if (xmax > inSize) xmax = inSize
            xmax -= xmin
            var ww = 0.0
            for (x in 0 until xmax) { val w = kernel(f, (x + xmin - center + 0.5) * ss); tmp[x] = w; ww += w }
            for (x in 0 until xmax) {
                val v = if (ww != 0.0) tmp[x] / ww else tmp[x]
                kk[xx * ksize + x] = if (v < 0) (-0.5 + v * (1 shl PRECISION_BITS)).toInt() else (0.5 + v * (1 shl PRECISION_BITS)).toInt()
            }
            xminA[xx] = xmin; xlenA[xx] = xmax
        }
        return Coeffs(xminA, xlenA, kk, ksize)
    }

    private fun clip8(v: Int): Int {
        if (v >= (1 shl PRECISION_BITS shl 8)) return 255
        if (v <= 0) return 0
        return v shr PRECISION_BITS
    }

    /** Horizontal pass on 3 interleaved channels (RGB bytes as Int 0..255), row-major w*h*3. */
    private fun horizontal(src: IntArray, w: Int, h: Int, outW: Int, f: Filter): IntArray {
        val c = coeffs(w, outW, f)
        val out = IntArray(outW * h * 3)
        val half = 1 shl (PRECISION_BITS - 1)
        for (y in 0 until h) {
            val row = y * w * 3
            for (xx in 0 until outW) {
                var s0 = half; var s1 = half; var s2 = half
                val xmin = c.xmin[xx]; val base = xx * c.ksize
                for (x in 0 until c.xlen[xx]) {
                    val k = c.k[base + x]; val p = row + (xmin + x) * 3
                    s0 += src[p] * k; s1 += src[p + 1] * k; s2 += src[p + 2] * k
                }
                val o = (y * outW + xx) * 3
                out[o] = clip8(s0); out[o + 1] = clip8(s1); out[o + 2] = clip8(s2)
            }
        }
        return out
    }

    private fun vertical(src: IntArray, w: Int, h: Int, outH: Int, f: Filter): IntArray {
        val c = coeffs(h, outH, f)
        val out = IntArray(w * outH * 3)
        val half = 1 shl (PRECISION_BITS - 1)
        for (yy in 0 until outH) {
            val ymin = c.xmin[yy]; val base = yy * c.ksize
            for (x in 0 until w) {
                var s0 = half; var s1 = half; var s2 = half
                for (y in 0 until c.xlen[yy]) {
                    val k = c.k[base + y]; val p = ((ymin + y) * w + x) * 3
                    s0 += src[p] * k; s1 += src[p + 1] * k; s2 += src[p + 2] * k
                }
                val o = (yy * w + x) * 3
                out[o] = clip8(s0); out[o + 1] = clip8(s1); out[o + 2] = clip8(s2)
            }
        }
        return out
    }

    /** PIL Image.resize((outW, outH), filter) for an RGB image given as interleaved 0..255 ints. */
    fun resize(rgb: IntArray, w: Int, h: Int, outW: Int, outH: Int, f: Filter): IntArray {
        var a = rgb; var cw = w
        if (outW != w) { a = horizontal(a, w, h, outW, f); cw = outW }
        if (outH != h) a = vertical(a, cw, h, outH, f)
        return a
    }

    /** Validated FYP3 path: native -> LANCZOS 512 -> BILINEAR 448. A 448x448 input is used unchanged. */
    fun fyp3Preprocess(argb: IntArray, w: Int, h: Int): IntArray {
        val rgb = IntArray(w * h * 3)
        for (i in 0 until w * h) {
            val p = argb[i]
            rgb[i * 3] = (p shr 16) and 0xFF; rgb[i * 3 + 1] = (p shr 8) and 0xFF; rgb[i * 3 + 2] = p and 0xFF
        }
        if (w == 448 && h == 448) return rgb
        val mid = if (w == 512 && h == 512) rgb else resize(rgb, w, h, 512, 512, Filter.LANCZOS)
        return resize(mid, 512, 512, 448, 448, Filter.BILINEAR)
    }
}
