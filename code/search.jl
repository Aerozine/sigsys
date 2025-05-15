ENV["GKSwstype"] = "100"
ENV["JULIA_NUM_THREADS"] = string(Sys.CPU_THREADS)
# fast math allow approx in 1e-10 +- with speed perf
@fastmath function delta(t)
    return -0.5 * pi * sin(2 * pi * 0.1 * t)
end # ! in function definition defines the rigidity can be ignored
@fastmath function odes!(du, u, p, t)
    f = p
    # @inbounds remove check of bounds in array for speed
    @inbounds du[1] = v0 * sin(u[2] + atan(a * tan(f(t)), b))
    @inbounds du[2] = v0 * sin(atan(a * tan(f(t)), b)) / a
end

@fastmath function linearode!(u, p, t)
    f = p
    @inbounds return A * u + B * f(t)
end

function ode2!(du, u, p, t)
    f, mu1, mu2, mu3, sigma1, sigma2, a1, a2 = p
    @inbounds du[1] = v0 * sin(u[2] +
                          atan(a * tan(f(t, mu1, mu2, mu3, sigma1, sigma2, a1, a2)), b))
    @inbounds du[2] = v0 *
                      sin(atan(a * tan(f(t, mu1, mu2, mu3, sigma1, sigma2, a1, a2)), b)) / a
end
@fastmath function linearode2!(u, p, t)
    f, mu1, mu2, mu3, sigma1, sigma2, a1, a2 = p
    @inbounds return A * u + B * f(t, mu1, mu2, mu3, sigma1, sigma2, a1, a2)
end

function gaussian(t, sigma, mu, A)
    A * exp(-(((t - mu) * (t - mu)) / (2 * sigma * sigma))) / sqrt(2 * pi * sigma * sigma)
end

function gauss3(t, mu1, mu2, mu3, sigma1, sigma2, a1, a2)
    gaussian(t, sigma1, mu1, a1) + gaussian(t, sigma2, mu2, a2) +
    gaussian(t, sigma1, mu3, a1)
end

# precompute data to save calculation
const real_val = gaussian.(t_eval, 5, 50, 25)
function mse(p)
    @inbounds mu1, mu2, mu3, sigma1, sigma2, a1, a2 = p
    # problem + solve = solveivp  
    @inbounds prob = ODEProblem(
        ode2!, [0.0, 0.0], [0, 100], (gauss3, mu1, mu2, mu3, sigma1, sigma2, a1, a2))
    @inbounds sol = solve(prob, saveat = t_eval, abstol = 1e-16)[1, :]
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

    #mu1,mu2,mu3,sigma1,sigma2,a1,a2 = [50-5.470934017973268,50,
    #    5.470934017973268+50, 4.474173680361383, 4.4214549387804185,
    #    0.027108495080634055, -0.054222576328522225]
    min_mse = Float64(Inf)
    mu1_range = (43.4, 44)
    mu2_range = (50, 50.2)
    mu3_range = (56.5, 56.8)
    sigma1_range = (4.3, 4.4)
    sigma2_range = (3.72, 3.73)
    a1_range = (0.0165, 0.018)
    a2_range = (-0.036, -0.033)
    minval = Vector{Float64}(undef, 7)
    it_count = Int(0)
    lock = Threads.ReentrantLock()
    progress = Progress(n, 1)
    println("starting multithreading with $(Sys.CPU_THREADS) threads :")
    Threads.@threads for _ in 1:1:n
        # generating random value
        mu1 = rand() * (mu1_range[2] - mu1_range[1]) + mu1_range[1]
        mu2 = rand() * (mu2_range[2] - mu2_range[1]) + mu2_range[1]
        mu3 = rand() * (mu3_range[2] - mu3_range[1]) + mu3_range[1]
        sigma1 = rand() * (sigma1_range[2] - sigma1_range[1]) + sigma1_range[1]
        sigma2 = rand() * (sigma2_range[2] - sigma2_range[1]) + sigma2_range[1]
        a1 = rand() * (a1_range[2] - a1_range[1]) + a1_range[1]
        a2 = rand() * (a2_range[2] - a2_range[1]) + a2_range[1]
        # get the local MSE for those value
        local_mse = mse([mu1, mu2, mu3, sigma1, sigma2, a1, a2])
        # lock allow the use of variable accros thread (ie  not 2 writer at the same time )
        Threads.lock(lock) do
            if min_mse > local_mse
                min_mse = local_mse
                minval .= mu1, mu2, mu3, sigma1, sigma2, a1, a2
            end
            it_count += 1
            # fancy stuff
            next!(progress;
                showvalues = [("Iterations", "$(it_count) / $(n)"),
                    ("min val", "$(min_mse)"), ("best params", "$(minval)")])
        end
    end
    #println(minval)
    #println(min_mse)
    initialguess=minval
    #initialguess=[44.47466746957489, 50.10972389695686, 55.7445931340356, 4.504552894450979, 4.151233595972581, 0.023696291835675492, -0.047392683335653336]
    res=optimize(mse,initialguess,
        f_abstol = 1e-16,
        x_abstol = 1e-16,
        iterations = 1_000_000,
        store_trace = false,
        show_trace = false)
    #println(Optim.minimum(res))
    #println(Optim.minimizer(res))

    open("bestresult.txt", "w") do io
      println(io, Optim.minimum(res))
      println(io, Optim.minimizer(res))
    end
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

function paufiner()
    # make sure initial guess lies within [lower,upper] for each dim:
    init = [
        44.474001685431354,
        50.109728232042926,
        55.74527249527475,
        4.504498952675612,
        4.150817864670581,
        0.0236891455876399,
        -0.04737838730301528
    ]

    # all bounds must be Float64, and lower ≤ initialguess ≤ upper
    lb = [43.4, 50.0, 55.7, 4.3, 3.72, 0.0165, -0.048]
    ub = [44.0, 50.2, 56.8, 4.4, 3.73, 0.018, -0.033]
    res=optimize(mse, init,
        f_abstol = 1e-16,
        x_abstol = 1e-16,
        iterations = 1_000_000_000,
        store_trace = false,
        show_trace = false)
    println(Optim.minimum(res))
    println(Optim.minimizer(res))
end
#    ____            _     ____  
#   |  _ \ __ _ _ __| |_  |___ \ 
#   | |_) / _` | '__| __|   __) |
#   |  __/ (_| | |  | |_   / __/ 
#   |_|   \__,_|_|   \__| |_____|

"
A=A-KB
A 
  0  0 
B=krB
C=C-KD
D=Dkr
"
@fastmath function linearcontroller(u, p, t)
    d, K, kr = p
    Atild = A - B * K
    Btild = B * kr
    Atild * u + Btild * d(t)
end
"
@fastmath function stepfunction(cdt)
  cdt?1:0
end
"
