import java.util

/**
 * Created by ytsegay on 10/2/2014.
 *
 * This class implements simplified Sequential Minimal Optimization (a much simplified version of John Platt's algo).
 * It is implemented as discussed in this paper
 * The algorithm over several iterations computes parameters associated with each training sample. Each training
 * sample is optimized and only support vector instances will have theta values at the end of the optimization
 * ... TODO: finish this off.
 */
class SimplifiedSMO {


	/* initialize kernel distances depending on the type of kernel we are using.
	 * for optimization reasons this is done once
	 */
	private def initializeKernels(p:SVMParam) = {
		// compute kernel distances once
		p.kernelDistances = Array.ofDim[Double](p.xRowCount,p.xRowCount)
		for (i <- 0 until p.xRowCount) {
			for (j <- 0 until p.xRowCount) p.kernelDistances(i)(j) = kernelDistance(p.x(i), p.x(j), p.kernel, p.gamma)
		}

		// reset
		for (i <- 0 until p.xRowCount) p.alphas(i) = 0.0
	}

	private def kernelDistance(x1:Array[Double], x2:Array[Double], kernel:String, gamma:Double) :Double = {
		if (kernel == "linear") {
			return MatrixAlgebraUtil.dotProduct(x1, x2)
		}
		else if(kernel == "rbf"){
			val diff = MatrixAlgebraUtil.subtract(x1, x2)
			val product = MatrixAlgebraUtil.dotProduct(diff, diff)
			return (product*product)/(-2*gamma*gamma)
		}
		// should not get here
		throw new IllegalArgumentException("Kernel " + kernel + " is not supported")
	}

	// compute prediction for an instance
	private def fx(p:SVMParam, current: Int): Double = {

		val xxP = MatrixAlgebraUtil.dotProduct(p.x, p.x(current))
		val alphasAndYs = MatrixAlgebraUtil.multiply(p.y, p.alphas)
		return MatrixAlgebraUtil.dotProduct(alphasAndYs, xxP) + p.b
	}

	private def clipBoundries(alpha: Double, min: Double, max: Double): Double = {
		if (alpha < min) {
			return min
		}

		if (alpha > max) {
			return max
		}

		return alpha
	}

	private def computeBoundries(p:SVMParam, i:Int, j:Int) :(Double, Double) = {
		var HB = 0.0
		var LB = 0.0

		if (p.y(i) != p.y(j)) {
			LB = Math.max(0.0, p.alphas(j) - p.alphas(i))
			HB = Math.min(p.C, p.C + p.alphas(j) - p.alphas(i))
		}
		else {
			LB = Math.max(0, p.alphas(i) + p.alphas(j) - p.C)
			HB = Math.min(p.C, p.alphas(i) + p.alphas(j))
		}
		return (LB, HB)
	}

	private def computeN(p:SVMParam, i:Int, j:Int): Double = {
		return 2 * p.kernelDistances(i)(j) - p.kernelDistances(i)(i) - p.kernelDistances(j)(j)
	}

	private def computeAlphaJ(alphaJ: Double, yJ: Double, errorDelta: Double, n: Double): Double = {
		return alphaJ - ((yJ * errorDelta) / n)
	}

	private def computeAlphaI(alphaI: Double, yi: Double, yj: Double, alphaJOld: Double, alphaJ: Double): Double = {
		return alphaI + (yi * yj * (alphaJOld - alphaJ))
	}

	private def computeB(p:SVMParam, i:Int, j:Int, err: Double, alphaI: Double, oldAlphaI: Double, alphaJ: Double, oldAlphaJ: Double): Double = {
		return p.b - err - (p.y(i) * (alphaI - oldAlphaI) * p.kernelDistances(i)(i)) - (p.y(j) * (alphaJ - oldAlphaJ) * p.kernelDistances(j)(j))
	}

	private def randIndex(max: Int, current: Int): Int = {
		val rnd = new scala.util.Random
		var rndIndex = -1
		do {
			rndIndex = rnd.nextInt(max)
		}
		while (rndIndex == current)

		return rndIndex
	}

	def train(p:SVMParam) = {

		initializeKernels(p)
		var iter = 0

		while (iter < p.maxIterations) {
			var alphasChanged = 0
			for (i <- 0 until p.xRowCount) {

				val fxI = fx(p, i)
				val errI = fxI - p.y(i)

				if ((p.y(i) * errI < -p.tol && p.alphas(i) < p.C) || (p.y(i) * errI > p.tol && p.alphas(i) > 0)) {

					// randomly pick another instance j where j != i
					val j = randIndex(p.xRowCount, i)
					val fxJ = fx(p, j)
					val errJ = fxJ - p.y(j)

					// save old alpha j and alpha i
					val oldAlphaJ = p.alphas(j)
					val oldAlphaI = p.alphas(i)

					// compute L and H
					val (lb, hb) = computeBoundries(p, i, j)
					if (Math.abs(lb - hb) > 0.01) {

						// compute N
						val N = computeN(p, i, j)
						if (N < 0.0) {

							// compute and clip alphaJ
							var newAlphaJ = computeAlphaJ(p.alphas(j), p.y(j), errI - errJ, N)
							newAlphaJ = clipBoundries(newAlphaJ, lb, hb)
							p.alphas(j) = newAlphaJ

							if (Math.abs(newAlphaJ - oldAlphaJ) >= 1e-5) {

								// compute alphaI
								val newAlphaI = computeAlphaI(p.alphas(i), p.y(i), p.y(j), oldAlphaJ, newAlphaJ)
								p.alphas(i) = newAlphaI

								// compute b1 and b2 and therefore determine b
								val b1 = computeB(p, i, j, errI, newAlphaI, oldAlphaI, newAlphaJ, oldAlphaJ)
								val b2 = computeB(p, i, j, errJ, newAlphaI, oldAlphaI, newAlphaJ, oldAlphaJ)

								if (newAlphaI > 0 && newAlphaI < p.C) {
									p.b = b1
								}
								else if (newAlphaJ > 0 && newAlphaJ < p.C) {
									p.b = b2
								}
								else {
									p.b = (b1 + b2) / 2
								}

								alphasChanged += 1
							}
						}
					}
				}
			}
			if (alphasChanged == 0) {
				iter += 1
			}
			else{
				iter = 0
			}
		}

		// identify support vectors, are those with theta > 0. They are the only ones at the end of the day
		// that should contribute to the classification
		for(i <- 0 until p.alphas.length){
			if(p.alphas(i) > 0.0) {
				p.supportVectIndices.add(i)
			}
		}
	}


	def classify(p:SVMParam, x:Array[Double]):Double = {

		val xTrans = new Array[Double](p.xRowCount)
		for(i <- 0 until xTrans.length) xTrans(i) = 0.0

		for(i <- 0 until p.supportVectIndices.size()){
			val index = p.supportVectIndices.get(i)
			xTrans(index) = kernelDistance(p.x(index), x, p.kernel, p.gamma)
		}
		val p1 = MatrixAlgebraUtil.multiply(p.alphas, p.y)
		val predictValue = MatrixAlgebraUtil.dotProduct(xTrans, p1) + p.b
		if (predictValue > 0.0){
			return 1.0
		}
		else if (predictValue < 0.0){
			return -1.0
		}
		throw new Exception("Prediction with zero. Shouldn't be here")
	}
}


/* a class that performs elementary matrix algebra. Simulates dot product and multiplication of matrices */
object MatrixAlgebraUtil {
	/* 1D array dot product and returns a single value*/
	def dotProduct(v1: Array[Double], v2: Array[Double]): Double = {
		assert(v1.length == v2.length)

		var sum = 0.0
		for (i <- 0 until v1.length) {
			sum += v1(i) * v2(i)
		}
		return sum
	}

	/*
	 * dot product between 1D array and a 2D array and returns an array which is the dot product
	 * between the the 1D array and each of the arrays in the 2D array
	 */
	def dotProduct(v1: Array[Array[Double]], v2: Array[Double]): Array[Double] = {
		assert(v1.length > 0 && v1(0).length == v2.length)
		val mtx = for (i <- 0 until v1.length) yield dotProduct(v1(i), v2)

		return mtx.toArray
	}

	/* multiplication of 1D arrays. */
	def multiply(v1: Array[Double], v2: Array[Double]): Array[Double] = {
		assert(v1.length == v2.length)
		val arr = for (i <- 0 until v1.length) yield v1(i) * v2(i)
		return arr.toArray
	}

	/* subtraction of 1D arrays. */
	def subtract(v1: Array[Double], v2: Array[Double]): Array[Double] = {
		assert(v1.length == v2.length)
		val arr = for (i <- 0 until v1.length) yield v1(i) - v2(i)
		return arr.toArray
	}
}

/* to be used to maintain model params */
class SVMParam (xParam: Array[Array[Double]], yParam: Array[Double], maxIterationsParam: Double, tolParam: Double, cParam: Double, kernelParam:String, gammaParam:Double) {

	/* private members */
	private val _x = xParam
	private val _y = yParam
	private val _maxIterations = maxIterationsParam
	private val _tol = tolParam
	private val _C = cParam
	private var _kernelDistances = new Array[Array[Double]](0)
	private val _supportVectIndex = new util.ArrayList[Int]()
	private val _alphas = new Array[Double](_x.length)
	var _b = 0.0
	private val _xRowCount = x.length
	private val _xColCount = x(0).length
	private val _kernel = kernelParam
	private val _gamma = gammaParam

	/* getters */
	def C = _C
	def x = _x
	def y = _y
	def maxIterations = _maxIterations
	def tol = _tol
	def kernelDistances = _kernelDistances
	def supportVectIndices = _supportVectIndex
	def alphas = _alphas
	def b = _b
	def b_=(_b:Double) = this._b = _b

	def kernelDistances_=(_kernelDistances:Array[Array[Double]]) = this._kernelDistances = _kernelDistances
	def xRowCount = _xRowCount
	def xColCount = _xColCount
	def kernel = _kernel
	def gamma = _gamma
}
