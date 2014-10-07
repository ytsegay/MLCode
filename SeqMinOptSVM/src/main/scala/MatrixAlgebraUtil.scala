/**
 * Created by ytsegay on 10/7/2014.
 */
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

	/* dot product between 1D array and a 2D array and returns an array which is the dot product
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