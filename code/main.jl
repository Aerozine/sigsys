using Base.Threads
using ProgressMeter
using LinearAlgebra
using Plots
using Optim
using OrdinaryDiffEq
using Statistics
using FFTW
const t_eval = 0:1:100
const tspan = [0.0, 100.0]
const a = 1.1
const b = 3.3
const v0 = 10.0
const A = [0.0 v0; 0.0 0.0]
const B = [v0 * a / b; v0 / b]
const C = [0.0 1.0]
const D = [0.0]
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
    for (func, name) in zip((delta, x -> 0), ("delta", "zero"))
        prob = ODEProblem(odes!, [0.0, 0.0], tspan, (func))
        sol = solve(prob, Tsit5(), reltol = 1e-5, abstol = 1e-5)
        p1_3_y = plot(sol.t, sol[1, :], xlabel = "Temps t", ylabel = "y(t)", label = "y")
        p1_3_θ = plot(sol.t, sol[2, :], xlabel = "Temps t", ylabel = "θ(t)", label = "θ")
        p1_3 = plot(
            p1_3_y, p1_3_θ, layout = (2, 1), plot_title = "Simulation pour delta=$(name)")
        # on the first 
        savefig("jlplots/Q1_3_$(name).pdf")
    end
end

@fastmath function Q1_7()
    for (ftype, ftype_name) in zip((odes!, linearode!), ("ODE", "linearODE"))
        p1_7 = plot(xlabel = "t", ylabel = ["y(t)" "θ(t)"], layout = (2, 1))
        for (case, case_name) in zip((0, 5 * pi / 180, 15 * pi / 180), ("0", "5", "15"))
            prob = ODEProblem(ftype, [0.0, 0.0], tspan, (x -> case))
            sol = solve(prob, Tsit5(), saveat = 0:1:100)
            plot!(p1_7, sol, layout = (2, 1),
                label = ["delta=$(case_name)" "delta=$(case_name)"])
        end
        plot(p1_7, plot_title = "Simulation pour $(ftype_name)")
        savefig("jlplots/Q1_7_$ftype_name.pdf")
    end
end
@fastmath function Q1_8()
    t_eval = 0:0.1:100
    # best guess
    dmu, sigma1, sigma2, a1, a2 = [
        5.470934017973268, 4.474173680361383, 4.4214549387804185,
        0.027108495080634055, -0.054222576328522225]
    #println("MSE : $(mse([dmu,sigma1,sigma2,a1,a2]))")
    @inbounds prob = ODEProblem(
        ode2!, [0.0, 0.0], [0, 100], (gauss3, dmu, sigma1, sigma2, a1, a2))
    @inbounds sol = solve(prob, saveat = t_eval)[1, :]
    real_val = gaussian.(t_eval, 5, 50, 25)
    plot(t_eval, sol, label = ["y(t)" "θ(t)"])
    plot!(t_eval, real_val)
    savefig("jlplots/Q1_8_similarity.pdf")
end
# ploting 
#Q1_3()
#Q1_7()
#Q1_8()
# for doing random search
#random_search(Int(1e9))
#    ____            _     ____  
#   |  _ \ __ _ _ __| |_  |___ \ 
#   | |_) / _` | '__| __|   __) |
#   |  __/ (_| | |  | |_   / __/ 
#   |_|   \__,_|_|   \__| |_____|

#
@fastmath function Q2_3()
  rplot=plot()
  uplot=plot()
  tplot=plot()
  yplot=plot()
    stepfunction(x)=(x > 10) ? 1 : 0
    kr=1
    for k in ([0.132 -0.0132], [0.132 0.5148], [0.132 1.1088])
        prob = ODEProblem(
            linearcontroller, [0.0, 0.0], tspan, (stepfunction, k, kr))
        t_eval = 0:0.01:100
        sol = solve(prob, Tsit5(), saveat = t_eval)
        plot!(tplot,sol.t, sol[2, :], label = ["theta $(k[2])"])
        plot!(yplot,sol.t, sol[1, :], label = ["y $(k[2])"])
        #println(-k.*x )
        
        p = .-dot.(Ref(k), sol.u) .+ stepfunction.(t_eval)
        plot!(uplot,t_eval,p, label =["u $(k[2])"])
        #plot!(uplot,t_eval,(-k.*x)[2,:])
    end
        plot!(rplot,0:0.01:100,stepfunction.(0:0.01:100),xlim=(0,100), ylim=(0,2),label="step")
    plot(tplot,yplot,rplot,uplot, layout=(2,2))
    savefig("jlplots/Q2_3.pdf")
end
function Q2_4()
    for freq in (10,1,0.2,0.01)
      k=[0.132 0.5148]
    #0.01 0.2 1 10
      kr=k[1]
    prob = ODEProblem(linearcontroller, [0.0, 0.0], tspan,
        ((x -> 2 * sin(2 * pi *freq *x + pi / 6)), k, kr))
    sol = solve(prob, Tsit5())
    plot!(sol,label=["f=$(freq)"])
  end
    #plot!(sol)
    #time = sol.t
    #y = sol[1, :] .^ 2
    #plot(time[y .> 0.0], y[y .> 0.0], yscale = :log2)
    #t = sol[2, :] .^ 2
    #plot!(time[t .> 0.0], t[t .> 0.0], yscale = :log2)
    savefig("jlplots/Q2_4.pdf")
end
Q2_4()
function Q2_6()
    rplot=plot()
    splot=plot()
    dt=0.01
    t_eval=0:dt:100
    for freq in (10,1,0.2,0.01)
      k=[0.132 0.5148]
    #0.01 0.2 1 10
      kr=k[1]
    prob = ODEProblem(linearcontroller, [0.0, 0.0], tspan,
        ((x -> 2 * sin(2 * pi *freq *x + pi / 6)), k, kr))
    sol = solve(prob, Tsit5())

    F = fft(signal) |> fftshift
    freqs = fftfreq(length(sol.u), 1.0/dt) |> fftshift
freq_domain = plot(freqs[mask], abs.(F[mask])/length(F), title = "Spectrum", xlim=(-70, +70),ylim=(0,10),line= :stem ) 
  end
end

function Q2_7()
# Number of points 
N = 2^14 - 1 
# Sample period
Ts = 1 / (1.1 * N) 
# Start time 
t0 = 0 
tmax = t0 + N * Ts
# time coordinate
t = t0:Ts:tmax

# signal 
signal = sin.(2π * 60 .* t) # sin (2π f t) 
Ts=dt
F = fft(signal) |> fftshift
freqs = fftfreq(length(t), 1.0/Ts) |> fftshift
mask = abs.(F) .>= 5000
println(1.0/Ts)
println(length(F))
# plots 
time_domain = plot(t, signal, title = "Signal")
freq_domain = plot(freqs[mask], abs.(F[mask])/length(F), title = "Spectrum", xlim=(-70, +70),ylim=(0,10),line= :stem ) 
plot(time_domain, freq_domain, layout = 2)
savefig("Wave.pdf")
end
Q2_7()
