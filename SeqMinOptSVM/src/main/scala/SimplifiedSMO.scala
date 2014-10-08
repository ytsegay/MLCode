import java.util

/**
 * Created by ytsegay on 10/2/2014.
 *
 * This class implements simplified Sequential Minimal Optimization (a much simplified version of John Platt's algo).
 * It is implemented as discussed in this paper
 * The algorithm over several iterations computes parameters associated with each training sample. Each training
 * sample is optimized and only support vector instances will have theta values at the end of the optimization
 *
 * http://common-lisp.net/p/cl-machine-learning/git/cl-svm/research/platt-smo-book.pdf
 * kernels: http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html
 *
 * // TODO: isoptimized == false works really well but true doesn't
 * // TODO: support for multiclass classification one V one or one V many
 *
 */
class SimplifiedSMO {


	/* initialize kernel distances, alphas and error cache */
	private def initializeKernels(p:SVMParam) = {
		// compute kernel distances once
		p.kernelDistances = Array.ofDim[Double](p.xRowCount,p.xRowCount)
		for (i <- 0 until p.xRowCount) for (j <- 0 until p.xRowCount) p.kernelDistances(i)(j) = kernelDistance(p.x(i), p.x(j), p.kernel, p.sigma)

		for (i <- 0 until p.xRowCount) p.alphas(i) = 0.0

		for (i <- 0 until p.xRowCount) p.errorCache(i) = 0.0
	}


	/* implementation of similarity kernels. */
	private def kernelDistance(x1:Array[Double], x2:Array[Double], kernel:String, sigma:Double) :Double = {
		if (kernel == "linear") {
			return MatrixAlgebraUtil.dotProduct(x1, x2)
		}
		else if(kernel == "rbf"){
			// compute euclidean distance sans sqrt of sum as we will take the square of the sum anyway
			val diff = MatrixAlgebraUtil.subtract(x1, x2)
			val product = MatrixAlgebraUtil.dotProduct(diff, diff)
			return scala.math.exp((product)/(-2*sigma*sigma))
		}
		// should not get here
		throw new IllegalArgumentException("Kernel " + kernel + " is not supported")
	}


	/* compute prediction for an instance's feature set */
	private def fx(p:SVMParam, current: Int): Double = {
		val alphasAndYs = MatrixAlgebraUtil.multiply(p.y, p.alphas)
		return MatrixAlgebraUtil.dotProduct(alphasAndYs, p.kernelDistances(current)) + p.b
	}

	/* */
	private def clipBoundries(alpha: Double, min: Double, max: Double): Double = {
		if (alpha < min)
			return min
		else if (alpha > max)
			return max
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
		}while (rndIndex == current)

		return rndIndex
	}

	/* TODO: needs love. Optimized version of SMO */
	private def runOuterLoopOverNonBoundedSet(p:SVMParam) :Int ={
		var alphasChanged = 0

		// get non-bound alphas
		val indexOfNonBoundAlphas = for {i <- 0 until p.xRowCount if (p.alphas(i) > 0 && p.alphas(i) < p.C)} yield i

		for (i <- indexOfNonBoundAlphas) {
			// compute errorI
			val errI = fx(p, i) - p.y(i)

			if ((p.y(i) * errI < -p.tol && p.alphas(i) < p.C) || (p.y(i) * errI > p.tol && p.alphas(i) > 0)) {
				p.errorCache(i) = errI

				// Now pick a J that will maximize the errorDelta i != j
				val indexOfNonZeroErrorCache = for {m <- 0 until p.xRowCount if (p.errorCache(m) != 0.0 && m != i)} yield m
				var maxJErr = -1.0
				var maxJ = -1
				if (indexOfNonZeroErrorCache.length > 0) {
					for (j <- 0 until indexOfNonZeroErrorCache.length) {
						val errJ = fx(p, j) - p.y(j)

						if (Math.abs(errJ - errI) > maxJErr) {
							maxJErr = Math.abs(errJ - errI)
							maxJ = j
						}
					}
				}
				else{
					maxJ = randIndex(p.xRowCount, i)
					maxJErr = fx(p, maxJ) - p.y(maxJ)
				}
				alphasChanged += runInnerLoop(p, i, maxJ, errI, maxJErr)
			}
		}

		return alphasChanged
	}


	/* runs the outer loop of SMO, shared by both optimzed and unoptimized versions */
	private def runOuterLoopOverEntireSet(p:SVMParam): Int ={
		var alphasChanged = 0
		for (i <- 0 until p.xRowCount) {

			val errI = fx(p, i) - p.y(i)

			if ((p.y(i) * errI < -p.tol && p.alphas(i) < p.C) || (p.y(i) * errI > p.tol && p.alphas(i) > 0)) {
				p.errorCache(i) = errI

				// randomly pick another instance j where j != i
				val j = randIndex(p.xRowCount, i)
				val errJ = fx(p, j) - p.y(j)

				alphasChanged += runInnerLoop(p, i, j, errI, errJ)
			}
		}
		return alphasChanged
	}


	/* runs the internal loop logic used by optimized and none optimzed versions of smo */
	private def runInnerLoop(p:SVMParam, i:Int, j:Int, errI:Double, errJ:Double): Int = {
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

				// recompute the error and store in error cache for J
				p.errorCache(j) = fx(p, j) - p.y(j)

				if (Math.abs(newAlphaJ - oldAlphaJ) >= 1e-5) {

					// compute alphaI
					val newAlphaI = computeAlphaI(p.alphas(i), p.y(i), p.y(j), oldAlphaJ, newAlphaJ)
					p.alphas(i) = newAlphaI

					// recompute the error and store in error cache for I
					p.errorCache(i) = fx(p, i) - p.y(i)

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

					return 1
				}
			}
		}
		return 0
	}

	/* public interface to train */
	def train(p:SVMParam, isOptimized:Boolean)={
		initializeKernels(p)
		var iter = 0

		if (isOptimized) {
			// this is an implementation of the optimizations as described in section TODO:
			runOuterLoopOverEntireSet(p)

			while (runOuterLoopOverNonBoundedSet(p) > 0 && iter < p.maxIterations) {
				iter += 1
			}
			runOuterLoopOverEntireSet(p)
		}
		else{
			// this uses the original implementation of the training phase, where the lagrange multipliers are computed
			// for every instance until there are maxIterations count of alphas changed (sequential changes)
			while (iter < p.maxIterations) {
				if (runOuterLoopOverEntireSet(p) == 0) {
					iter += 1
				}
				else {
					iter = 0
				}
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

	/* public interface to do classification. You will need an SVMParam that has been trained successfully and a
	 * feature-set array of doubles of the same length as the training set and it outputs a -1.0 or a 1.0 prediction
	 */
	def classify(p:SVMParam, x:Array[Double]):Double = {

		val xTrans = new Array[Double](p.xRowCount)
		for(i <- 0 until xTrans.length) xTrans(i) = 0.0

		for(i <- 0 until p.supportVectIndices.size()){
			val index = p.supportVectIndices.get(i)
			xTrans(index) = kernelDistance(p.x(index), x, p.kernel, p.sigma)
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