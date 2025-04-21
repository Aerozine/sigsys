using Base.Threads
using Symbolics
using ControlSystems
using ProgressMeter
using LinearAlgebra
using Plots
using Optim
using OrdinaryDiffEq
using Statistics
using FFTW
const t_eval = 0:0.01:100
const tspan = [0.0, 100.0]
const a = 1.1
const b = 3.3
const v0 = 10.0
const A = [0.0 v0; 0.0 0.0]
const B = [v0 * a / b; v0 / b]
const C = [1.0 0.0]
const D = [0.0]
# search.jl basicly include all ODE function , search algorithm etc
# all the funny stuff is inside
include("search2.jl")
# the main file is used for plot and questions

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
        savefig("jlplots/Q1_3_$(name).pdf")
    end
end

@fastmath function Q1_7()
    for (ftype, ftype_name) in zip((odes!, linearode!), ("ODE", "linearODE"))
        p1_7 = plot(xlabel = "t", ylabel = ["y(t)" "θ(t)"], layout = (2, 1))
        for (case, case_name) in zip((0, 5 * pi / 180, 15 * pi / 180), ("0", "5", "15"))
            prob = ODEProblem(ftype, [0.0, 0.0], tspan, (x -> case))
            sol = solve(prob, Tsit5(), saveat = t_eval)
            plot!(p1_7, sol, layout = (2, 1),
                label = ["delta=$(case_name)" "delta=$(case_name)"])
        end
        plot(p1_7, plot_title = "Simulation pour $(ftype_name)")
        savefig("jlplots/Q1_7_$ftype_name.pdf")
    end
end
@fastmath function Q1_8()
    # best guess with random_search
    #dmu, sigma1,
    #sigma2,
    #a1,
    #a2 = [
    #5.470934017973268, 4.474173680361383, 4.4214549387804185,
    #0.027108495080634055, -0.054222576328522225]

    mu1,mu2,mu3,sigma1,sigma2,a1,a2 = [50-5.470934017973268,50,
        5.470934017973268+50, 4.474173680361383, 4.4214549387804185,
        0.027108495080634055, -0.054222576328522225]
    #mu1,mu2,mu3,sigma1,sigma2,a1,a2= [21.286644028118854, 58.997679172695115, 76.43477063793996, 0.1016826337641108, 
    #3.134927252309066, 1.5432272684688175, -1.3848950884321622]

    #mu1, mu2,
    #mu3,
    #sigma1,
    #sigma2,
    #a1,
    #a2 = [44.230719624796386, 48.96366028224329, 53.20962706874073, 3.406846505976441,
    #    3.1681776456999535, 0.0352139637276877, -0.07094820532273971]#[50-5.470934017973268,50,
    #        5.470934017973268+50, 4.474173680361383, 4.4214549387804185,
    #        0.027108495080634055, -0.054222576328522225]
    @inbounds prob = ODEProblem(
        ode2!, [0.0, 0.0], [0, 100], (gauss3, mu1, mu2, mu3, sigma1, sigma2, a1, a2))
    @inbounds sol = solve(prob, saveat = t_eval)[1, :]
    real_val = gaussian.(t_eval, 5, 50, 25)
    plot(t_eval, sol, label = ["y(t)" "θ(t)"], linestyle = :dot)
    plot!(t_eval, real_val)
    savefig("jlplots/Q1_8.pdf")
    println(mse([mu1,mu2,mu3,sigma1,sigma2,a1,a2]))
end
#Q1_8()
# for doing random search
#random_search(Int(1e9))
#    ____            _     ____  
#   |  _ \ __ _ _ __| |_  |___ \ 
#   | |_) / _` | '__| __|   __) |
#   |  __/ (_| | |  | |_   / __/ 
#   |_|   \__,_|_|   \__| |_____|
#
#TODO
#check graph quality 
#beautify and consistency
#2_3 -> change label 
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
        sol = solve(prob, Tsit5(), saveat = t_eval)
        plot!(tplot, sol.t, sol[2, :], label = ["theta $(k[2])"])
        plot!(yplot, sol.t, sol[1, :], label = ["y $(k[2])"])

        p = .-dot.(Ref(k), sol.u) .+ stepfunction.(t_eval)
        plot!(uplot, t_eval, p, label = ["u $(k[2])"])
    end
    plot!(rplot, t_eval, stepfunction.(t_eval),
        xlim = (0, 100), ylim = (0, 2), label = "step")
    plot(tplot, yplot, rplot, uplot, layout = (2, 2))
    savefig("jlplots/Q2_3.pdf")
end

function Q2_4()
    subplots=(plot(), plot(), plot(), plot())
    for (freq, plots, xim) in zip((10, 1, 0.2, 0.01), subplots, (3, 10, 20, 100))
        k=[0.132 0.5148]
        kr=k[1]
        prob = ODEProblem(linearcontroller, [0.0, 0.0], [0, xim],
            ((x -> 2 * sin(2 * pi * freq * x + pi / 6)), k, kr))
        sol = solve(prob, Tsit5(), saveat = 0:(xim / 100000):xim)
        plot!(plots, sol.t, sol[2, :], xlim = (0, xim),
            title = "frequency=$(freq)Hz", legend = false)
    end
    plot(subplots[1], subplots[2], subplots[3], subplots[4], layout = (4, 1))
    savefig("jlplots/Q2_4.pdf")
end
# 2_6 -> do a mask to hide smol freq <0.001 or justify it on the report
function Q2_6()
    dt=t_eval[2]-t_eval[1]
    subplots=(plot(), plot(), plot(), plot())
    for (freq, plots, xim, yim) in
        zip((10, 1, 0.2, 0.01), subplots, (11, 2, 1, 0.1), (0.01, 0.1, 0.20, 0.009))
        k=[0.132 0.5148]
        input(x) = 2 * sin(2 * pi * freq * x + pi / 6)
        kr=k[1]
        prob = ODEProblem(linearcontroller, [0.0, 0.0], tspan,
            (input, k, kr))
        r=input.(t_eval)
        sol = solve(prob, Tsit5(), saveat = t_eval)
        for (signal, name) in zip((r, sol[2, :]), ("input", "output"))
            F = fft(signal) |> fftshift
            freqs = fftfreq(length(signal), 1.0/dt) |> fftshift
            plot!(plots, freqs, abs.(F)/length(signal), title = "Spectrum", line = :stem,
                label = ["f=$(freq) $(name)"], xlim = (-xim, +xim), ylim = (0, yim))
        end
    end
    plot(subplots[1], subplots[2], subplots[3], subplots[4], layout = (2, 2))
    savefig("jlplots/Q2_5r.pdf")
end

#    ____            _   _____ 
#   |  _ \ __ _ _ __| |_|___ / 
#   | |_) / _` | '__| __| |_ \ 
#   |  __/ (_| | |  | |_ ___) |
#   |_|   \__,_|_|   \__|____/ 
function Q3_1()
    k=[0.132 0.5148]
    kr=k[1]
    Atild = A - B * k
    Btild = B * kr
    Ctild = C-D * k
    Dtild=D*kr
    
    sys_ss = ss(Atild, Btild, Ctild, Dtild)

    sys_tf = tf(sys_ss)

    display(sys_tf)
    #w = exp10.(LinRange(-1, 2, 500))
setPlotScale("dB")                                          # :contentReference[oaicite:1]{index=1}

# 2) Generate the standard 2‑panel Bode plot, with grid and French title
fig = bodeplot(sys_tf; plotphase=true, grid=true,
               title="diagram de bode")                    # :contentReference[oaicite:2]{index=2}

# 3) Relabel axes explicitly to show units
plot!(fig[1]; xlabel="ω (rad/s)", ylabel="Magnitude (dB)")  # :contentReference[oaicite:3]{index=3}
plot!(fig[2]; xlabel="ω (rad/s)", ylabel="Phase (°)")   
    # Generate Bode plot
#    bodeplot(sys_tf, w; title="Bode Plot of the System", xlabel="Frequency (rad/s)", ylabel="Magnitude (dB) / Phase (deg)", layout=(2,1))
    savefig("jlplots/Q3.pdf")
end

#    ____  _       _   _   _             
#   |  _ \| | ___ | |_| |_(_)_ __   __ _ 
#   | |_) | |/ _ \| __| __| | '_ \ / _` |
#   |  __/| | (_) | |_| |_| | | | | (_| |
#   |_|   |_|\___/ \__|\__|_|_| |_|\__, |
#                                  |___/ 
#

# plotting asynchronously to speed up the process
@sync begin
#    @async Q1_3()
#    @async Q1_7()
#    @async Q1_8()
#    @async Q2_3()
#    @async Q2_4()
#    @async Q2_6()
    @async Q3_1()

end
#"
