import java.util

/**
 * Created by ytsegay on 10/7/2014.
 */
/* to be used to maintain model params */
class SVMParam (xParam: Array[Array[Double]], yParam: Array[Double], maxIterationsParam: Double, tolParam: Double, cParam: Double, kernelParam:String, sigmaParam:Double) {

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
	private val _sigma = sigmaParam
	private val _errorCache = new Array[Double](_x.length)

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
	def sigma = _sigma
	def errorCache = _errorCache
}

