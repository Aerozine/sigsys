using Base.Threads
using ProgressMeter
using LinearAlgebra
using Plots
using Optim
using OrdinaryDiffEq
using Statistics
const t_eval = 0:0.1:100
const tspan = [0.0, 100.0]
const a = 1.1
const b = 3.3
const v0 = 10.0
# search.jl basicly include all ODE function , search algorithm etc
# all the funny stuff is inside
include("search.jl")
# the main file is used for plot 
"
todolist
- pimp all graph
- document code  
- part 2 n 3
"

#    ____            _     _ 
#   |  _ \ __ _ _ __| |_  / |
#   | |_) / _` | '__| __| | |
#   |  __/ (_| | |  | |_  | |
#   |_|   \__,_|_|   \__| |_|

@fastmath function Q1_3()
    for (func, name) in zip((delta, x -> gaussian(x, 5, 50, 25)), ("_delta", "_zero"))
        prob = ODEProblem(odes!, [0.0, 0.0], tspan, (func))
        sol = solve(prob, Tsit5(), reltol = 1e-5, abstol = 1e-5)
        plot(sol, layout = (2, 1), label = ["y(t)" "θ(t)"])
        savefig("jlplots/Q3$name.png")
    end
end
@fastmath function Q1_7()
    for (ftype, ftype_name) in zip((odes!, linearode), ("ODE", "linearODE"))
        for (case, case_name) in zip((x -> 0, x -> 5, x -> 15), ("0", "5", "15"))
            prob = ODEProblem(ftype, [2.0, 0.0], tspan, (case))
            sol = solve(prob, Tsit5(), reltol = 1e-5, abstol = 1e-5)
            plot!(sol, layout = (2, 1), label = ["y(t)" "θ(t)"])
        end
        savefig("jlplots/Q7$ftype_name.png")
    end
end
@fastmath function Q1_8()
    t_eval = 0:0.1:100
    # best guess
    dmu, sigma1, sigma2, a1, a2 = [
        5.470934017973268, 4.474173680361383, 4.4214549387804185,
        0.027108495080634055, -0.054222576328522225]
    println("MSE : $(mse([dmu,sigma1,sigma2,a1,a2]))")
    @inbounds prob = ODEProblem(
        ode2, [0.0, 0.0], [0, 100], (gauss3, dmu, sigma1, sigma2, a1, a2))
    @inbounds sol = solve(prob, saveat = t_eval)[1, :]
    real_val = gaussian.(t_eval, 5, 50, 25)
    plot(t_eval, sol, label = ["y(t)" "θ(t)"])
    plot!(t_eval, real_val)
    savefig("jlplots/Q8_similarity.pdf")
end
# ploting 
#Q1_3();Q1_7();
#Q1_8()
# for doing random search
random_search(Int(5e8))
#grid_search()
#    ____            _     ____  
#   |  _ \ __ _ _ __| |_  |___ \ 
#   | |_) / _` | '__| __|   __) |
#   |  __/ (_| | |  | |_   / __/ 
#   |_|   \__,_|_|   \__| |_____|

@fastmath function Q2_3()
end
