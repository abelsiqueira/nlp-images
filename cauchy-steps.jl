using Plots, ForwardDiff
pyplot(size=(600,600*3/4), reuse=true, grid=false)

include("goldensearch.jl")

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
  png(@sprintf("cauchy-steps-%03d", k))
  return k+1
end

function foo(;all_levels=true)
  #f(x) = 0.05*( (x[1]-2)^2 + 4*(x[2]-1.5)^2 )
  #x = zeros(2)
  #lx, ux, ly = -1.2, 6.3, -1.2
  #f(x) = (1-x[1])^2 + 100*(x[2]-x[1]^2)^2
  sc = 0.1
  f(x) = sc*( 0.8*(x[1]-3)^2 - (x[2]-1)^3 - 0.1*x[1] + x[2] )
  x = [-2.0; 0.0]
  lx, ux, ly = -3.2, 6.3, -1.2
  ∇f(x) = ForwardDiff.gradient(f, x)

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
      levels=linspace(minf, maxf, 40), leg=false)
  else
    #=
    contour(linspace(lx,ux,200), linspace(ly,uy,200), (x,y)->f([x;y]),
      c=:grays, levels=linspace(minf, maxf, 20), leg=false)
    plotpoint!([-10;-10])
    imgc = saveimg(imgc, lx, ux, ly, uy)
    =#
    contour(linspace(lx,ux,200), linspace(ly,uy,200), (x,y)->f([x;y]),
      levels=[f(x)], leg=false)
  end
  X = Any[x]

  # One image of the point alone
  plotpoint!(x, name=0)
  imgc = saveimg(imgc, lx, ux, ly, uy)
  # One image of the gradient
  gradplot!(x, ∇f, name=0)
  imgc = saveimg(imgc, lx, ux, ly, uy)
  traceplot!(x, -∇f(x)*15)
  imgc = saveimg(imgc, lx, ux, ly, uy)
  iter = 1
  ∇fx = ∇f(x)
  while norm(∇fx) > 1e-1*sc && iter <= 10
    t = golden_ls(f, x, -∇fx)
    moveplot!(x, -t*∇fx)
    x = x - t*∇fx
    plotpoint!(x)
    if !all_levels
      contour!(linspace(lx,ux,200), linspace(ly,uy,200), (x,y)->f([x;y]),
        levels=[f(x)], leg=false)
    end
    imgc = saveimg(imgc, lx, ux, ly, uy)
    ∇fx = ∇f(x)
    iter += 1
    if norm(x) > 10
      break
    end
  end

  # Now really solving (using Newton)
  H(x) = ForwardDiff.hessian(f, x)
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
