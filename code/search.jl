ENV["GKSwstype"] = "100"
ENV["JULIA_NUM_THREADS"] = string(Sys.CPU_THREADS)
# fast math allow approx in 1e-10 +- with speed perf
@fastmath function delta(t)
    return -0.5 * pi * sin(2 * pi * 0.1 * t)
end
# ! in function definition defines the rigidity can be ignored
@fastmath function odes!(du, u, p, t)
    f = p
    # @inbounds remove check of bounds in array for speed
    @inbounds du[1] = v0 * sin(u[2] + atan(a * tan(f(t)), b))
    @inbounds du[2] = v0 * sin(atan(a * tan(f(t)), b)) / a
end

@fastmath function linearode(du, u, p, t)
    f = p
    @inbounds du[1] = v0 * a * f(t) / b + v0 * u[2]
    @inbounds du[2] = v0 * f(t) / b
end

function ode2(du, u, p, t)
    f, dmu, sigma1, sigma2, a1, a2 = p
    @inbounds du[1] = v0 * sin(u[2] + atan(a * tan(f(t, dmu, sigma1, sigma2, a1, a2)), b))
    @inbounds du[2] = v0 * sin(atan(a * tan(f(t, dmu, sigma1, sigma2, a1, a2)), b)) / a
end

@fastmath function gaussian(t, sigma, mu, A)
    A * exp(-(((t - mu) * (t - mu)) / (2 * sigma * sigma))) / sqrt(2 * pi * sigma * sigma)
end

function gauss3(t, dmu, sigma1, sigma2, a1, a2)
    gaussian(t, sigma1, 50 - dmu, a1) + gaussian(t, sigma2, 50, a2) +
    gaussian(t, sigma1, 50 + dmu, a1)
end

# precompute data to save calculation
const real_val = gaussian.(t_eval, 5, 50, 25)
function mse(p)
    dmu, sigma1, sigma2, a1, a2 = p
    # problem + solve = solveivp  
    @inbounds prob = ODEProblem(
        ode2, [0.0, 0.0], [0, 100], (gauss3, dmu, sigma1, sigma2, a1, a2))
    @inbounds sol = solve(prob, saveat = t_eval)[1, :]
    # . means operation over the array 
    # ie .- is equal to a for loop where result[i]=real[i]-sol[i] but optimized
    @inbounds mse = mean((real_val .- sol) .^ 2)
    mse
end
"
performs a random search accros the interval given
This method is an alternative of the grid method and get an idea where to find min value
"
function random_search(n)
    #dmu_range = (5, 6)
    #sigma1_range = (3, 5)
    #sigma2_range = (3, 5)
    #a1_range = (0.01, 0.05)
    #a2_range = (-0.08, -0.04)
    min_mse = Float64(Inf)
    dmu_range = (10, 50)
    sigma1_range = (0, 10)
    sigma2_range = (3, 5)
    a1_range = (0, 5)
    a2_range = (-5, 0)
    minval = Vector{Float64}(undef, 5)
    it_count = Int(0)
    lock = Threads.ReentrantLock()
    progress = Progress(n, 1)
    println("starting multithreading with $(Sys.CPU_THREADS) threads :")
    Threads.@threads for _ in 1:1:n
        # generating random value
        dmu = rand() * (dmu_range[2] - dmu_range[1]) + dmu_range[1]
        sigma1 = rand() * (sigma1_range[2] - sigma1_range[1]) + sigma1_range[1]
        sigma2 = rand() * (sigma2_range[2] - sigma2_range[1]) + sigma2_range[1]
        a1 = rand() * (a1_range[2] - a1_range[1]) + a1_range[1]
        a2 = rand() * (a2_range[2] - a2_range[1]) + a2_range[1]
        # get the local MSE for those value
        local_mse = mse([dmu, sigma1, sigma2, a1, a2])
        # lock allow the use of variable accros thread (ie  not 2 writer at the same time )
        Threads.lock(lock) do
            if min_mse > local_mse
                min_mse = local_mse
                minval .= dmu, sigma1, sigma2, a1, a2
            end
            it_count += 1
            # fancy stuff
            next!(progress;
                showvalues = [("Iterations", "$(it_count) / $(n)"),
                    ("min val", "$(min_mse)"), ("best params", "$(minval)")])
        end
    end
    println(minval)
    println(min_mse)
end
"
grid search accros the interval
maybe used to refine random search later 
"
const dt = 0.01
function grid_search()
    dmu_range = 5:dt:6
    sigma1_range = 3:dt:5
    sigma2_range = 3:dt:5
    a1_range = 0.01:dt:0.05
    a2_range = -0.08:dt:-0.04
    min_mse = Float64(Inf)
    minval = Vector{Float64}(undef, 5)
    it_count = Int(0)
    lock = Threads.ReentrantLock()
    n = length(dmu_range) * length(sigma1_range) * length(sigma2_range) * length(a1_range) *
        length(a2_range)
    progress = Progress(n, 1)
    println("starting $(n) task ,multithreading with $(Sys.CPU_THREADS) threads :")
    total_it = length(dmu_range) * length(sigma1_range) * length(sigma2_range) *
               length(a1_range) * length(a2_range)
    progress = Progress(total_it, 1)
    Threads.@threads for dmu in dmu_range
        for sigma1 in sigma1_range
            for sigma2 in sigma2_range
                for a1 in a1_range
                    for a2 in a2_range
                        local_mse = mse([dmu, sigma1, sigma2, a1, a2])
                        # lock allow the use of variable accros thread (ie  not 2 writer at the same time )
                        Threads.lock(lock) do
                            if min_mse > local_mse
                                min_mse = local_mse
                                minval .= dmu, sigma1, sigma2, a1, a2
                            end
                            it_count += 1
                            # fancy stuff
                            next!(progress;
                                showvalues = [("Iterations", "$(it_count) / $(n)"),
                                    ("min val", "$(min_mse)"),
                                    ("best params", "$(minval)")])
                        end
                    end
                end
            end
        end
    end
    println(minval)
    println(min_mse)
end
