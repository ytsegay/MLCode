/**
 * Created by ytsegay on 10/4/14.
 */
object TestSmoSVM {


	private def loadFile(fileName:String): (Array[Array[Double]], Array[Double]) = {

		val f = scala.io.Source.fromFile(fileName).getLines().map(_.trim.split("\t")).toList
		val xs = for(i <- 0 until f.length) yield {
			val row = new Array[Double](f(i).length-1)
			for(j<- 0 until f(i).length-1) {
				row(j) = f(i)(j).toDouble
			}
			row
		}
		val ys = for(i <- 0 until f.length) yield (f(i)(f(i).length-1).toDouble)

		return (xs.toArray,ys.toArray)
	}

	def main (args: Array[String]) {

		val iters = 10000
		val tol = 0.0001
		val C = 200
		val kernel = "linear" //can also use rbf for now
		val sigma = 0.6

		val (xs, ys) = loadFile("/Users/ytsegay/git/MLCode/SeqMinOptSVM/src/main/resources/uci2.txt")
		val clfParam = new SVMParam(xs, ys, iters, tol, C, kernel, sigma)
		val clf = new SimplifiedSMO
		clf.train(clfParam, isOptimized=false)

		println("Support vectors: ")
		var counter = 0
		for(i <- 0 until clfParam.alphas.length){
			if(clfParam.alphas(i) > 0) {
				println("" + counter + ". alpha: " + clfParam.alphas(i) + " x1: " + clfParam.x(i)(0) + " x2: " + clfParam.x(i)(1) + " y:" + clfParam.y(i))
				counter += 1
			}
		}
		println("b: " + clfParam.b)

		var countMismatches = 0
		for(i <- 0 until clfParam.xRowCount){
			val pred = clf.classify(clfParam, clfParam.x(i))
			if (pred != clfParam.y(i))
				countMismatches += 1
		}
		println("Train error rate: " + (countMismatches.toDouble/clfParam.xRowCount))

//		countMismatches = 0
//		val (txs, tys) = loadFile("S:\\git\\MLCode\\SeqMinOptSVM\\src\\main\\resources\\testSetRBF2.txt")
//		for(i <- 0 until txs.length){
//			val pred = clf.classify(clfParam, txs(i))
//			if (pred != tys(i))
//				countMismatches += 1
//		}
//		println("Test error rate: " + (countMismatches.toDouble/txs.length))
	}
}
