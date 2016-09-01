using Plots, ForwardDiff
pyplot(size=(600,600*3/4), reuse=true, grid=false)

include("goldensearch.jl")

function plotcountours(f, F, lx, ux, ly, uy)
  contour(linspace(lx,ux,200), linspace(ly,uy,200), (x,y)->f([x;y]),
    levels=[F; 1000], leg=false)
end

function contourmodel!(m, lx, ux, ly, uy)
  contour!(linspace(lx,ux,200), linspace(ly,uy,200), (x,y)->m([x;y]),
    levels=50, c=:grays, leg=false)
end

function plotpointmove!(X; c=:blue)
  n = length(X)
  plotpoint!(X[1], c=c, name=0)
  for i = 1:n-1
    moveplot!(X[i], X[i+1]-X[i])
    plotpoint!(X[i+1], c=c)
  end
end

function plotpoint!(x; name=-1, annpos=[-0.3;0.0], c=:blue)
  scatter!([x[1]], [x[2]], c=c, ms=4)
  if !(typeof(name) <: Number) || name >= 0
    annotate!(x[1]+annpos[1],x[2]+annpos[2],text("\$x^$name\$"))
  end
end

function gradplot!(x, g; name=-1)
  v = g(x)
  plot!([x[1]; x[1]+v[1]], [x[2]; x[2]+v[2]], c=:blue, lw=2, l=:arrow)
  if name >= 0
    annotate!(x[1]+v[1],x[2]+v[2]-0.3,text("\$\\nabla f(x^$name)\$"))
  end
end

function traceplot!(x, v)
  plot!([x[1]; x[1]+v[1]], [x[2]; x[2]+v[2]], c=:black, lw=1, ls=:dash)
end

function moveplot!(x, v)
  plot!([x[1]; x[1]+v[1]], [x[2]; x[2]+v[2]], c=:blue, lw=1)
end

function saveimg(k, lx, ux, ly, uy)
  xlims!(lx, ux)
  ylims!(ly, uy)
  png(@sprintf("newton-steps-%03d", k))
  return k+1
end

function foo(;all_levels=false)
  #f(x) = 0.05*( (x[1]-2)^2 + 4*(x[2]-1.5)^2 )
  #f(x) = (1-x[1])^2 + 100*(x[2]-x[1]^2)^2
  f(x) = 0.8*(x[1]-3)^2 - (x[2]-1)^3 - 0.1*x[1] + x[2]
  x = [-2.0; 0.0]
  lx, ux, ly = -3.2, 6.3, -1.2
  ∇f(x) = ForwardDiff.gradient(f, x)
  H(x) = ForwardDiff.hessian(f, x)

  uy = ly + 3(ux-lx)/4

  minf, maxf = Inf, -Inf
  for a in linspace(lx,ux,100)
    for b in linspace(ly,uy,100)
      fx = f([a;b])
      maxf = max(maxf, fx)
      minf = min(minf, fx)
    end
  end

  imgc = 1
  if all_levels
    contour(linspace(lx,ux,200), linspace(ly,uy,200), (x,y)->f([x;y]),
      levels=linspace(minf, maxf, 20), leg=false)
  else
    #=
    contour(linspace(lx,ux,200), linspace(ly,uy,200), (x,y)->f([x;y]),
      c=:grays, levels=linspace(minf, maxf, 20), leg=false)
    plotpoint!([-10;-10])
    imgc = saveimg(imgc, lx, ux, ly, uy)
    contour(linspace(lx,ux,200), linspace(ly,uy,200), (x,y)->f([x;y]),
      levels=[f(x)], leg=false)
    =#
  end
  X = Any[x]
  F = [f(x)]

  # One image of the point alone
  iter = 1
  ∇fx = ∇f(x)
  Bx = H(x)
  while norm(∇fx) > 1e-1 && iter <= 10
    # plot all x and f(x) so far
    plotcountours(f, F, lx, ux, ly, uy)
    plotpointmove!(X)
    imgc = saveimg(imgc, lx, ux, ly, uy)

    d = -Bx\∇fx

    m(d) = f(x) + dot(d,∇fx) + 0.5*dot(d,Bx*d)
    contourmodel!(y->m(y-x)-m(d), lx, ux, ly, uy)
    imgc = saveimg(imgc, lx, ux, ly, uy)

    moveplot!(x, d)
    x = x + d
    push!(X, x)
    push!(F, f(x))
    sort!(F)

    plotpoint!(x)
    imgc = saveimg(imgc, lx, ux, ly, uy)

    ∇fx = ∇f(x)
    Bx = H(x)
    iter += 1
    if norm(x) > 10
      break
    end
  end

  # Now really solving (using Newton)
  while norm(∇fx) > 1e-8 && iter <= 1000
    d = -H(x)\∇fx
    if dot(d,∇fx) > -1e-8
      t = golden_ls(f, x, -∇fx)
      x = x - t*∇fx
    else
      x = x + d
    end
    ∇fx = ∇f(x)
    iter += 1
    if norm(x) > 10
      break
    end
  end
  plotpoint!(x, name="*", annpos=[0.3;0.0], c=:red)
  imgc = saveimg(imgc, lx, ux, ly, uy)
end

foo()
