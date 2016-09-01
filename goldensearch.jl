function golden_ls(f, x, v; tol=1e-6)
  x, ef, fx, nf = min1d_gold(t->f(x+t*v), tol_f=tol)
  if ef > 0
    println("t = $x")
    error("Deu ruim")
  elseif x < 0
    error("Deu ruim neg")
  end
  return x
end

function min1d_gold(f, a=0, b=a+2-φ; tol_f=1e-6, tol_x=1e-3, maxnf=1000, verbose=false)
  # Encontrar o intervalo
  fa, fb = f(a), f(b)
  nf = 2
  ef = 0

  seq = 0.1

  s = 1.0
  # f(a) e f(b) devem ser diferentes
  while abs(fa - fb) < tol_f*(1 + abs(fa))
    s *= 2.0
    b = b + s
    fb = f(b)
    nf += 1
    if nf >= maxnf
      return b, 1, fb, nf
    elseif abs(b) > 1e6
      return b, -1, fb, nf
    end
  end
  # Direção de decréscimo
  if fa < fb
    a,b,fa,fb = b,a,fb,fa
  end
  # Agora é preciso achar tres pontos com o do meio menor que o os outros.
  c = b + (b - a)*φ
  fc = f(c)
  nf += 1
  if nf >= maxnf
    return b, 1, fb, nf
  end
  verbose && println("$nf: (a,b,c)=($a,$b,$c) f=($fa,$fb,$fc)")
  # Vamos andar até que f(c) seja maior que f(b)
  while fb - fc >= -tol_f*(1 + abs(fb))
    a,b,fa,fb = b,c,fb,fc
    c = b + (b - a)*φ
    fc = f(c)
    verbose && println("$nf: (a,b,c)=($a,$b,$c) f=($fa,$fb,$fc)")
    nf += 1
    if nf >= maxnf
      return b, 1, fb, nf
    elseif abs(c) > 1e6
      return c, -1, fc, nf
    end
  end
  # Calculando d e colocando na ordem certa
  # Se a < b
  # |---|-----|
  # a   b     c
  # Corrige para
  # |---|-|---|
  # a   d c   b
  # c = a + (b-a)φ
  # Se a > b
  # |-----|---|
  # c     b   a
  # Corrige para
  # |---|-|---|
  # a   d c   b
  # d = c + (a-c)φ
  if verbose
    println("a: $a $fa")
    println("b: $b $fb")
    println("c: $c $fc")
  end
  if a < b
    b,d,fb,fd = c,b,fc,fb
    c = a + (b-a)/φ
    fc = f(c)
  elseif a > b
    a,b,c,fa,fb,fc = c,a,b,fc,fa,fb
    d = b - (b-a)/φ
    fd = f(d)
  end
  if verbose
    println("a: $a $fa")
    println("b: $b $fb")
    println("c: $c $fc")
    println("d: $d $fd")
  end

  while b-a > tol_x || max(fa,fb) - min(fc,fd) > tol_f
    if fd < fc
      c,b,fc,fb = d,c,fd,fc
      d = b - (b-a)/φ
      fd = f(d)
    else
      a,d,fa,fd = d,c,fd,fc
      c = a + (b-a)/φ
      fc = f(c)
    end
    if verbose
      println("$nf: (a,d,c,b)=($a,$d,$c,$b)")
      println("  f(a,d,c,b)=($fa,$fd,$fc,$fb)")
    end
    nf += 1
    if nf >= maxnf
      ef = 1
      break
    end
  end
  if fd < fc
    return d, ef, fd, nf
  else
    return c, ef, fc, nf
  end
end
