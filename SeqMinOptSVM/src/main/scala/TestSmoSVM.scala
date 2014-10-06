/**
 * Created by ytsegay on 10/4/14.
 */
object TestSmoSVM {


	def main (args: Array[String]) {
		val fileName = "S:\\git\\MLCode\\SeqMinOptSVM\\src\\main\\resources\\testSet.txt"

		val f = scala.io.Source.fromFile(fileName).getLines().map(_.trim.split("\t")).toList
		val xs = for(i <- 0 until f.length) yield Array(f(i)(0).toDouble, f(i)(1).toDouble)
		val ys = for(i <- 0 until f.length) yield (f(i)(2).toDouble)

		val iters = 40
		val tol = 0.01
		val C = 0.6
		val kernel = "linear" //can also use rbf for now
		val gamma = 1.3

		val clfParam = new SVMParam(xs.toArray, ys.toArray, iters, tol, C, kernel, gamma)

		val clf = new SimplifiedSMO
		clf.train2(clfParam)

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
