/**
 * Created by ytsegay on 10/4/14.
 */
object TestSmoSVM {


	def main (args: Array[String]) {
		val fileName = "/Users/ytsegay/git/MLCode/SeqMinOptSVM/src/main/resources/testSetRBF.txt"

		val f = scala.io.Source.fromFile(fileName).getLines().map(_.trim.split("\t")).toList
		val xs = for(i <- 0 until f.length) yield Array(f(i)(0).toDouble, f(i)(1).toDouble)
		val ys = for(i <- 0 until f.length) yield (f(i)(2).toDouble)

		val iters = 1000
		val tol = 0.0001
		val C = 200
		val kernel = "rbf"
		val gamma = 1.3


		val clfParam = new SVMParam(xs.toArray, ys.toArray, iters, tol, C, kernel, gamma)

		val clf = new SimplifiedSMO
		clf.train(clfParam)

		println("Support vectors: ")
		for(i <- 0 until clfParam.alphas.length){
			if(clfParam.alphas(i) > 0) {
				println("alpha: " + clfParam.alphas(i) + " x1: " + clfParam.x(i)(0) + " x2: " + clfParam.x(i)(1) + " y:" + clfParam.y(i))
			}
		}
		println("b: " + clfParam.b)

		var countMatches = 0
		for(i <- 0 until clfParam.xRowCount){
			val pred = clf.classify(clfParam, clfParam.x(i))
			if (pred == clfParam.y(i))
				countMatches += 1
		}
		println("Training accuracy: " + (countMatches.toFloat/clfParam.xRowCount))
	}

}