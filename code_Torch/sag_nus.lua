--[[ An implementation of SAG-NUS (Stochastic Gradient Descent with Non-Uniform Sampling)

This implementation of L-BFGS relies on a user-provided line
search function (state.lineSearch). If this function is not
provided, then a simple learningRate is used to produce fixed
size steps. Fixed size steps are much less costly than line
searches, and can be useful for stochastic problems.

The learning rate is used even when a line search is provided.
This is also useful for large-scale stochastic problems, where
opfunc is a noisy approximation of f(x). In that case, the learning
rate allows a reduction of confidence in the step size.

ARGS:

- `opfunc` : a function that takes a single input (X), the point of
evaluation, and returns f(X) and df/dX
- `x` : the initial point
- `state` : a table describing the state of the optimizer; after each
call the state is modified
- `state.maxIter` : Maximum number of iterations allowed
- `state.maxEval` : Maximum number of function evaluations
- `state.tolFun` : Termination tolerance on the first-order optimality
- `state.tolX` : Termination tol on progress in terms of func/param changes
- `state.lineSearch` : A line search function
- `state.learningRate` : If no line search provided, then a fixed step size is used

RETURN:
- `x*` : the new `x` vector, at the optimal point
- `f`  : a table of all function values:
`f[1]` is the value of the function before any optimization and
`f[#f]` is the final fully optimized value, at `x*`

(Clement Farabet, 2012)
]]
function sag_nus(opfunc, x, config, state)
	-- get/update state
	local config = config or {}
	local state = state or config
	local maxIter = tonumber(config.maxIter) or 20
	local maxEval = tonumber(config.maxEval) or maxIter*1.25
	local tolFun = config.tolFun or 1e-5
	local tolX = config.tolX or 1e-9
	local nCorrection = config.nCorrection or 100
	local lineSearch = config.lineSearch
	local lineSearchOpts = config.lineSearchOptions
	local learningRate = config.learningRate or 1
	local isverbose = config.verbose or false
	local monitor = optimState.monitor
	local sample = config.sampler
	local fevalIntervel = config.fevalIntervel

	state.updateCount = state.updateCount or 0
	state.funcEval = state.funcEval or 0
	state.nIter = state.nIter or 0
	local start_time = sys.clock()

	-- verbose function
	local verbose
	if isverbose then
		verbose = function(...) print('<sag-nus> ', ...) end
	else
		verbose = function() end
	end

	-- import some functions
	local abs = math.abs
	local min = math.min

	-- evaluate initial f(x) and df/dx
	local f,g = opfunc(x)

	-- check optimality of initial point
	state.tmp1 = state.tmp1 or g.new(g:size()):zero(); local tmp1 = state.tmp1
	tmp1:copy(g):abs()
	gtol = tmp1:sum()
	if tmp1:sum() <= tolFun then
		-- optimality condition below tolFun
		verbose('optimality condition below tolFun')
		return x,f_hist
	end

	-- !!! UNDONE: g_old need to be stored
	local g_old = g.new(g:size()):zero()
	local f_old = 0

	-- optimize for a max of maxIter iterations
	local nIter = 0	-- iteration number
	local d = g.new(g:size()):zero()		-- sum of gradients, initialized zero

	while nIter < maxIter do
		if fmod(state.updateCount, fevalIntervel) == 0 then
			f_old = fx
			fx,g = opfunc(x)

			nIter = nIter + 1
			state.nIter = state.nIter + 1
			state.funcEval = state.funcEval + 1
			io.write(string.format("%d %.4f %.4f %.3f %d ", nIter-1, fx, gtol, sys.clock()-start_time, state.funcEval))
			if monitor then monitor(x) end
			print('')
		end
		
		i, fi,gi = sample(x)
		Lmean = torch.mean(state.L)
		Lmax = torch.max(state.L)
		-- if ith example has been sampled before
		if state.sampleRecord[i] == 1 then
			state.L[i] = 0.9*state.L[i]
		else
			state.sampleRecord[i] = 1
			state.L[i] = 0.5*Lmean
		end

		d = d - gi + g_old[i]
		g_old[i]:copy(gi)

		if torch.norm(gi) > 1e-8 then
			-- !!! UNDONE: line search
		end
		-- !!! UNDONE: alpha, w, (L) update

		f_old = f

		------------------------------------------------------------
		-- check conditions
		------------------------------------------------------------
		if nIter == maxIter then
			-- no use to run tests
			verbose('reached max number of iterations')
			break
		end

		if currentFuncEval >= maxEval then
			-- max nb of function evals
			verbose('max nb of function evals')
			break
		end

		tmp1:copy(g):abs()
		gtol = tmp1:sum()
		if tmp1:sum() <= tolFun then
			-- check optimality
			verbose('optimality condition below tolFun')
			break
		end

		tmp1:copy(d):mul(t):abs()
		if tmp1:sum() <= tolX then
			-- step size below tolX
			verbose('step size below tolX')
			break
		end

		if abs(f-f_old) < tolX then
			-- function value changing less than tolX
			verbose('function value changing less than tolX')
			break
		end

		if state.sampleRecord:sum() == state.sampleRecord:size(1) then
			verbose('all the training examples are sampled')
			break
		end
	end

	-- save state
	state.old_dirs = old_dirs
	state.old_stps = old_stps
	state.Hdiag = Hdiag
	state.g_old = g_old
	state.f_old = f_old
	state.t = t
	state.d = d

	-- return optimal x, and history of f(x)
	return x,f_hist,currentFuncEval
end
