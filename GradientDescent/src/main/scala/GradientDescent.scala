/**
 * Created by ytsegay on 9/19/14.
 * Gradient descent with batch and online implementation
 */
object GradientDescent {

  def predict(x:Array[Double], theta:Array[Double]) : Double = {
    var predict:Double = 0.0
    for (i <- 0 until x.size){
      predict += theta(i)*x(i)
    }
    predict
  }

  def stochasticGradientDescent(x:Array[Array[Double]], y:Array[Double], verbose:Boolean, learnRate:Double) = {
    val instancesCount = x.length
    var theta:Array[Double] = new Array[Double](x(0).size)

    // initialize thetas to zeros
    for(i<- 0 until theta.size) theta(i) = 0

    // repeat for all instances
    for(i <- 0 until instancesCount){
      // keep track of new theta
      val newTheta = new Array[Double](theta.size)

      for(j <- 0 until theta.size){
          // get derivative of J theta
          val DJ = (predict(x(i), theta) - y(i)) * x(i)(j)
          newTheta(j) = theta(j) - (learnRate * DJ)
      }

      theta = newTheta

      if (verbose) {
        // what is the mse after the update
        val errSq = for (k <- 0 until instancesCount) yield Math.pow(predict(x(k), theta) - y(k), 2)
        println(f"${errSq.sum / instancesCount}%2.2f\t\tStochastic MSE at iteration $i")
      }
    }
    theta
  }


  def batchGradientDescentRegression(x:Array[Array[Double]], y:Array[Double], verbose:Boolean, learnRate:Double, iterationsCount:Int) = {
    val instancesCount = x.length

    var theta:Array[Double] = new Array[Double](x(0).length)
    // initialize thetas to zeros
    for(i<- 0 until theta.length) theta(i) = 0.0

    // repeat for iteration, last one dedicated to intercept
    for(i <- 0 until iterationsCount){
      // keep track of new theta
      val newTheta = new Array[Double](theta.length)

      for(j <- 0 until theta.length){
        // get derivative of J theta for features
        var DJSum = 0.0
        for (k <- 0 until instancesCount) DJSum += ((predict(x(k), theta) - y(k)) * x(k)(j))
        newTheta(j) = theta(j) - ((learnRate*DJSum)/instancesCount)
      }

      theta = newTheta

      if (verbose) {
        val errSq = for (k <- 0 until instancesCount) yield Math.pow(predict(x(k), theta) - y(k), 2)
        println(f"${(0.5 * errSq.sum) / instancesCount}%2.10f\t\tBatch MSE at iteration $i")
      }
    }

    theta
  }

  def loadFile(fileName: String) = {
    val source = scala.io.Source.fromFile(fileName)
    val lines = source.getLines().toList.map(_.trim.split("   ").map(_.toDouble))
    source.close()

    lines
  }

  def standardize(x:Array[Array[Double]]) = {

    for(i <- 0 until x(0).length-1) {
      val col = for(j <- 0 until x.length) yield x(j)(i)
      val mean = col.sum/x.length.toDouble

      val variance = for(j <- 0 until x.length) yield (col(j) - mean)*(col(j) - mean)
      val stddev = Math.sqrt(variance.sum/(x.length-1))

      for(j <- 0 until x.length) {
        x(j)(i) = (x(j)(i) - mean)/stddev
      }
    }
  }

  def printTheta(theta:Array[Double]) = {
    for(i <- 0 until theta.length) {
      println(f"$i => ${theta(i)}%2.10f")
    }
  }

  def main(args:Array[String]): Unit = {

    val featuresFile = "/Users/ytsegay/git/Scala/GradientDescent/src/main/resources/ex2x.dat"
    val targetFile = "/Users/ytsegay/git/Scala/GradientDescent/src/main/resources/ex2y.dat"
    val learnRate = 0.07
    val iterationsCount = 1500
    val verbose = true
    val shouldStandardise = false
    val mode = "online" // can also be online


    val trainLines = loadFile(featuresFile)
    val yLines = loadFile(targetFile)

    val xs = for (row <- trainLines) yield {
      val t = new Array[Double](row.length + 1)
      for (j <- 0 until row.length) t(j) = row(j)
      t(row.length) = 1.0 // intercept
      t
    }
    val X = xs.toArray

    val ys = for (i <- 0 until yLines.length) yield yLines(i)(0)
    val Y = ys.toArray

    if (shouldStandardise) {
      standardize(X)
    }

    if (mode.toLowerCase == "batch"){
      if (verbose) {
        println("\n\n")
      }

      val theta = batchGradientDescentRegression(X, Y, verbose, learnRate, iterationsCount)
      if (verbose){
        println("\n\nMode: " + mode)
        printTheta(theta)
      }
    }
    else if (mode.toLowerCase == "online") {
      if (verbose)
        println("\n\n")

      val theta = batchGradientDescentRegression(X, Y, verbose, learnRate, iterationsCount)
      if (verbose){
        println("\n\nMode: " + mode)
        printTheta(theta)
      }
    }
  }
}
