/**
 * Created by ytsegay on 10/2/2014.
 */
object SimplifiedSMO {

  // compute prediction for an instance
  def fx(x:Array[Array[Double]], y:Array[Double], current:Int, alphas:Array[Double], b:Double) :Double = {
	  val util = new MatrixAlgebraUtil

	  val xxP = util.dotProduct(x, x(current))
	  val alphasAndYs = util.multiply(y, alphas)
	  return util.dotProduct(alphasAndYs, xxP)
  }

  def clipBoundries(alpha:Double, min:Double, max:Double) :Double = {
    return 1.0
  }

  def computeBountries(yi:Double, yj:Double, alphai:Double, alphaj:Double, C:Double) {
    var H = 0.0
	  var L = 0.0

	  if (yi != yj) {
		  L = Math.max(0.0, alphaj - alphai)
		  H = Math.min(C, C + alphaj - alphai)
	  }
	  else {
		  L = Math.max(0, alphai + alphaj - C)
		  H = Math.min(C, alphai + alphaj)
	  }
    return (L,H)
  }

	def computeN(xi:Array[Double], xj:Array[Double]):Double = {
		val x = new MatrixAlgebraUtil
		return 2*x.dotProduct(xi, xj) - x.dotProduct(xi, xi) - x.dotProduct(xj,xj)
	}

  def computeAlphaJ(alphaj:Double, yj:Double, errorDelta:Double, n:Double) = {
		alphaj - ((yj*errorDelta)/n)
  }

	def computeAlphai(alphai:Double, yi:Double, yj:Double, alphajOld:Double, alphaj:Double){
		return alphai + yi*yj*(alphajOld - alphaj)
	}

	def smoOpt(x:Array[Array[Double]], y:Array[Double], maxIterations:Double, tol:Double) = {
		val alphas = new Array[Double](x(0).length)
		var b = 0.0

		for(i <- 0 until x.length) {

			val fxi = fx(x, y, i, alphas, b)
			val erri = fxi - y(i)


			if(KKT passes) {
				// randomly pick another instance
				//
				val j = 1
				val fxj = fx(x, y, j, alphas, b)
				val errj = fxi - y(j)

				// save old alpha j and alpha i

				// compute L and H

				// compute N

				// compute and clip alphaj

				// compute alphai

				// compute b1 and b2 and therefore determine b

			}
		}
	}
}


class MatrixAlgebraUtil{
	def dotProduct(v1:Array[Double], v2:Array[Double]):Double = {
		assert(v1.length == v2.length)
		var sum = 0.0
		for(i <- 1 until v1.length){ sum += v1(i)*v2(i)}
		return sum
	}

	def multiply(v1:Array[Double], v2:Array[Double]) :Array[Double] = {
		assert(v1.length == v2.length)
		val arr = for(i <- 1 until v1.length) yield v1(i)*v2(i)
		return arr.toArray
	}

	def dotProduct(v1:Array[Array[Double]], v2:Array[Double]) :Array[Double] = {
		assert(v1.length > 0 && v1(0).length == v2.length)
		val mtx = for(i <- 1 until v1.length) yield dotProduct(v1(i), v2)

		return mtx.toArray
	}
}
