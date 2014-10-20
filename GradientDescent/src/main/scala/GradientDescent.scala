import scala.util.Random
import scala.util.Try
/**
 * Created by ytsegay on 9/19/14.
 * Gradient descent with batch and online implementation
 *
 * The code will support
 * 1. both linear and logistic regression models
 * 2. generic predictions and cost functions, providing the flexibility of mixing pred and cost functions
 * 3. one hot encoding in support of categorical values. for this to work we detect if a column contains non-alphanumberic content and one hot encode
 * all those
 *
 */


// TODO: wrap the features in a class and the target in perhaps another class
// TODO: pass config and files are arguments/config file
// TODO: code should be split into helpers, main batch+online code and driver
// TODO: test one-hot-encode
// TODO: all 0 until code should be replaced with indices
object GradientDescent {


  private def predictLinear(x:Array[Double], theta:Array[Double]): Double = {
    var predict:Double = 0.0
    for (i <- 0 until x.size){
      predict += theta(i)*x(i)
    }
    predict
  }

  private def predictLogit(x:Array[Double], theta:Array[Double]): Double = {
    var predict: Double = 0.0
    for (i <- 0 until x.size) {
      predict += theta(i) * x(i)
    }

    1/(1+Math.exp(predict))
  }

/* RSS (residual sq sum) */
private def leastSqCost(yHat: Array[Double], actualY: Array[Double]): Double = {
  val leastSqError = for (k <- 0 until yHat.length) yield Math.pow(yHat(k) - actualY(k), 2)
  (0.5 * leastSqError.sum) / yHat.length
}

/* logistic regression cost function */
private def logRegCost(yHat: Array[Double], actualY: Array[Double]): Double = {
  val logErr = for (k <- 0 until yHat.length) yield (-actualY(k)*Math.log(yHat(k))) - ((1-actualY(k)*Math.log(1-yHat(k))))
  logErr.sum / yHat.length
}


def stochasticGradientDescent(x:Array[Array[Double]], y:Array[Double],
                            verbose:Boolean, learnRate:Double, iterationsCount:Int,
                            predict: (Array[Double], Array[Double]) => Double,
                            cost: (Array[Double], Array[Double]) => Double) = {
  val instancesCount = x.length
  var theta:Array[Double] = new Array[Double](x(0).size)

  // initialize thetas to zeros
  for(i<- 0 until theta.size) theta(i) = 0

  for(t <- 0 until iterationsCount) {
    // random order is important
    val randomOrdered = Random.shuffle((0 until instancesCount).toList)
    // repeat for all instances
    for (i <- randomOrdered) {
      // keep track of new theta
      val newTheta = new Array[Double](theta.size)

      for (j <- 0 until theta.size) {
        // get derivative of J theta
        val DJ = (predict(x(i), theta) - y(i)) * x(i)(j)
        newTheta(j) = theta(j) - (learnRate * DJ)
      }

      theta = newTheta

      if (verbose) {
        // what is the mse after the update
        val predictions = for (k <- 0 until instancesCount) yield predict(x(k), theta)
        val err = cost(predictions.toArray, y)
        println(f"${err}%2.10f\t\tOnline MSE at iteration $i")
      }
    }
  }
  theta
}


def batchGradientDescentRegression(x:Array[Array[Double]], y:Array[Double],
                                   verbose:Boolean, learnRate:Double, iterationsCount:Int,
                                   predict: (Array[Double], Array[Double]) => Double,
                                   cost: (Array[Double], Array[Double]) => Double) = {
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
      val predictions = for (k <- 0 until instancesCount) yield predict(x(k), theta)
      val err = cost(predictions.toArray, y)
      println(f"${err}%2.10f\t\tBatch MSE at iteration $i")
    }
  }

  theta
}

def loadFile(fileName: String) = {
  val source = scala.io.Source.fromFile(fileName)
  val lines = source.getLines().toList.map(_.trim.split("   "))
  source.close()

  lines
}

/* to avoid radical jumps when the coefficients are estimated it is important that the
 * features are scaled around the mean within a stddev of 1
 */
def scale(x:Array[Array[Double]]) = {

  for(i <- 0 until x(0).length) {
    val col = for(j <- 0 until x.length) yield x(j)(i)
    val mean = col.sum/x.length.toDouble

    val variance = for(j <- 0 until x.length) yield (col(j) - mean)*(col(j) - mean)
    val stddev = Math.sqrt(variance.sum/(x.length-1))

    if (stddev > 0) {
      for (j <- 0 until x.length) {
        x(j)(i) = (x(j)(i) - mean) / stddev
      }
    }
  }
}

private def parseDouble(s: String): Option[Double] = {Try { s.toDouble }.toOption}

def oneHotEncodeNonNumericFeats(x:Array[Array[String]]): (Array[Array[Double]], scala.collection.mutable.HashMap[String,Int]) = {
  // identify non-numeric columns
  val map = scala.collection.mutable.HashMap.empty[String,Int]

  var currentRunIndex:Int = 0
  for (k <- x(0).indices) {
    val colIndicesWithAlphaNumValues = new collection.mutable.ListBuffer[String]()
    for (j <- x.indices) {
      if (parseDouble(x(j)(k)) == None && !colIndicesWithAlphaNumValues.contains(x(j)(k))) {
        colIndicesWithAlphaNumValues += x(j)(k)
      }
    }
    if (colIndicesWithAlphaNumValues.toArray.length == 0) {
      // map runningIndex to K
      map += (k.toString -> currentRunIndex)
      currentRunIndex += 1
    }
    else {
      // foreach entry in colIndices ... map its "k:value" to runningIndex
      for (entry <- colIndicesWithAlphaNumValues.toArray) {
        map += ((k.toString + ":" + entry) -> currentRunIndex)
        currentRunIndex += 1
      }
    }
  }

  // at this point you should have the length of the new 2d array (x.length) and its width (currentRunIndex+1)
  // + 1 for intercept
  // create an empty array and set its values to zer0
  val newX = Array.ofDim[Double](x.length, currentRunIndex+1)
  for{
    i <- newX.indices
    j <- newX(0).indices
  } newX(i)(j) = 0.0

  // traverse over the data again.
  for (k <- x(0).indices){
    if (map.contains(k.toString)){
      // copy its values as is
      for(j <- 0 until x.length) {
        newX(j)(k) = x(j)(k).toDouble
      }
    }
    else{
      for(j <- x.indices) {
        val indx = map(k.toString + ":" + x(j)(k))
        println(indx)
        newX(j)(indx) = 1.0
      }
    }
  }
  // this time if k is not in the hashtable see if k:value is ... it must be

  // one last order of business is to set the intercept
  for(j <- x.indices) {
    newX(j)(currentRunIndex) = 1.0
  }
  (newX, map)
}

def printTheta(theta:Array[Double]) = {
  for(i <- 0 until theta.length) {
    println(f"$i => ${theta(i)}%2.10f")
  }
}

def main(args:Array[String]): Unit = {

  val featuresFile = "C:\\git\\MLCode\\GradientDescent\\src\\main\\resources\\ex3x.dat"
  val targetFile = "C:\\git\\MLCode\\GradientDescent\\src\\main\\resources\\ex3y.dat"
  val learnRate = 1.0
  val iterationsCount = 1000
  val verbose = true
  val shouldScale = true
  val mode = "batch" // can also be online
  val lambda = 0.1    // regularization parameter
  val shouldOneHotEncode = false


  val trainLines = loadFile(featuresFile)
  val yLines = loadFile(targetFile)

  var X:Array[Array[Double]] = Array.ofDim[Double](0,0)
  if (shouldOneHotEncode){

    val (xx, map) = oneHotEncodeNonNumericFeats(trainLines.toArray)
    X = xx
    println(map.mkString(", "))
    for(v <- X) {
      println(v.mkString(", "))
    }
  }
  else {

    val xs = for (row <- trainLines) yield {
      val t = new Array[Double](row.length + 1)
      for (j <- 0 until row.length) t(j) = row(j).toDouble
      t(row.length) = 1.0 // intercept
      t
    }
    X = xs.toArray
  }

  val ys = for (i <- 0 until yLines.length) yield yLines(i)(0).toDouble
  val Y = ys.toArray

  if (shouldScale) {
    scale(X)
  }

  if (mode.toLowerCase == "batch"){
    if (verbose) {
      println("\n\n")
    }

    val theta = batchGradientDescentRegression(X, Y, verbose, learnRate, iterationsCount, predictLinear, leastSqCost)
    if (verbose){
      println("\n\nMode: " + mode)
      printTheta(theta)
    }
  }
  else if (mode.toLowerCase == "online") {
    if (verbose)
      println("\n\n")

//      val theta = stochasticGradientDescent(X, Y, verbose, learnRate, iterationsCount)
//      if (verbose){
//        println("\n\nMode: " + mode)
//        printTheta(theta)
//      }
  }
}
}
