using Plots, ForwardDiff
pyplot(size=(600,600*3/4), reuse=true, grid=false)

include("goldensearch.jl")

function linesearch_armijo(f, x, v, tol=1e-6)
  ϕ(t) = f(x + t*v)
  dϕ(t) = ForwardDiff.derivative(ϕ, t)
  t = 1.0
  ϕ0 = ϕ(0)
  dϕ0 = dϕ(0)
  while ϕ(t) > ϕ(0) + 0.5*t*dϕ0
    t *= 0.9
  end
  return t
end

function linesearch(f, x, v, tol=1e-6)
  ϕ(t) = f(x + t*v)
  dϕ(t) = ForwardDiff.derivative(ϕ, t)
  d2ϕ(t) = ForwardDiff.derivative(dϕ, t)
  t = 0.0
  while abs(dϕ(t)) > tol
    t = t - dϕ(t)/d2ϕ(t)
    if t < 0
      return linesearch_armijo(f, x, v, tol)
    end
  end
  return t
end

function cauchy(f, x; tol=1e-2, maxiter=1000)
  X = Any[x]
  ∇f(x) = ForwardDiff.gradient(f, x)
  iter = 1
  while norm(∇f(x)) > tol && iter <= maxiter
    t = golden_ls(f, x, -∇f(x))
    x = x - t*∇f(x)
    push!(X, x)
    iter += 1
    if norm(x) > 10
      break
    end
  end
  return X
end

function newton_ls(f, x; tol=1e-2, maxiter=1000)
  X = Any[x]
  ∇f(x) = ForwardDiff.gradient(f, x)
  H(x) = ForwardDiff.hessian(f, x)
  iter = 1
  while norm(∇f(x)) > tol && iter <= maxiter
    d = -H(x)\∇f(x)
    if dot(d,∇f(x)) >= 0
      break
    end
    t = linesearch_armijo(f, x, d)
    x = x + t*d
    push!(X, x)
    iter += 1
    if norm(x) > 10
      break
    end
  end
  return X
end

function newton(f, x; tol=1e-2, maxiter=10)
  X = Any[x]
  ∇f(x) = ForwardDiff.gradient(f, x)
  H(x) = ForwardDiff.hessian(f, x)
  iter = 1
  while norm(∇f(x)) > tol && iter <= maxiter
    d = -H(x)\∇f(x)
    x = x + d
    push!(X, x)
    iter += 1
    if norm(x) > 10
      break
    end
  end
  return X
end

function bfgs(f, x; tol=1e-2, maxiter=10)
  X = Any[x]
  ∇f(x) = ForwardDiff.gradient(f, x)
  B = eye(length(x))
  iter = 1
  while norm(∇f(x)) > tol && iter <= maxiter
    d = -B\∇f(x)
    t = linesearch_armijo(f, x, d)
    s = t*d
    y = ∇f(x+s) - ∇f(x)
    x = x + s
    push!(X, x)
    iter += 1
    if norm(x) > 10
      break
    end
    Bs = B*s
    if dot(s,y) > 0
      B = B - Bs*Bs'/dot(s,Bs) + y*y'/dot(s,y)
    end
  end
  return X
end

function foo()
  #f(x) = (1-x[1])^2 + 100*(x[2]-x[1]^2)^2
  f(x) = 0.8*(x[1]-3)^2 - (x[2]-1)^3 - 0.1*x[1] + x[2]
  lx, ux, ly = 1.5, 4.0, 0.0
  uy = ly + 3(ux-lx)/4

  maxf = -Inf
  for a in [lx;ux]
    for b in [ly;uy]
      maxf = max(maxf, f([a;b]))
    end
  end

  contour(linspace(lx,ux,200), linspace(ly,uy,200),
    (x,y)->f([x;y]), levels=linspace(0.0,maxf,100), leg=false)
  png("contour")

  x₀s = Any[[2.0;0.9], [2.0;1.1]]
  for i=1:10
    push!(x₀s, [lx+rand()*(ux-lx);ly+rand()*(uy-ly)])
  end

  for mtd in [cauchy, newton, newton_ls, bfgs]
    for (x₀i,x₀) in enumerate(x₀s)
      contour(linspace(lx,ux,200), linspace(ly,uy,200),
        (x,y)->f([x;y]), levels=30, leg=false)
      X = mtd(f, x₀, maxiter=10)
      F = sort([f(x) for x in X])
      n = length(X)
      println("$n pontos")
      println("sol: $(X[end])")
      xs = [x[1] for x in X]
      ys = [x[2] for x in X]
      scatter!(xs, ys, c=:blue, ms=3)
      for i = 1:n-1
        x = X[i]
        y = X[i+1]
        plot!([x[1]; y[1]], [x[2]; y[2]], c=:blue)
      end
      x = X[n]
      scatter!([x[1]], [x[2]], c=:blue, ms=3)
      xlims!(lx, ux)
      ylims!(ly, uy)
      png("$mtd-$x₀i")
    end
  end
end

foo()
