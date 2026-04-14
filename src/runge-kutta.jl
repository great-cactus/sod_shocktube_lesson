module RungeKutta

struct ButcherTableau{T<:Real}
  A::Matrix{T}
  b::Vector{T}
  c::Vector{T}
end

function RK2(T=Float64)
  ButcherTableau{T}(
    [0 0;
    1//2 0],
    [0, 1],
    [0, 1//2]
  )
end

function SSPRK2(T=Float64)
  ButcherTableau{T}(
    [0 0;
    1 0],
    [1//2, 1//2],
    [0, 1]
  )
end

function RK4(T=Float64)
  ButcherTableau{T}(
    [0 0 0 0;
    1//2 0 0 0;
    0 1//2 0 0;
    0 0 1 0],
    [1//6, 1//3, 1//3, 1//6],
    [0, 1//2, 1//2, 1]
  )
end

function SSPRK3(T=Float64)
  ButcherTableau{T}(
    [0 0 0;
    1 0 0;
    1//4 1//4 0],
    [1//6, 1//6, 2//3],
    [0, 1, 1//2]
  )
end

function nstages(tab::ButcherTableau)
  length(tab.b)
end

function rk_step(f, t, u, dt, tab::ButcherTableau)
  s = nstages(tab)
  k = Vector{typeof(u)}(undef, s)

  for i in 1:s
    du = zero(u)
    for j in 1:i-1
      du += tab.A[i, j] * k[j]
    end
    k[i] = f(t + tab.c[i] * dt, u + dt * du)
  end

  u_new = copy(u)
  for i in 1:s
    u_new += dt * tab.b[i] * k[i]
  end

  return u_new
end

end # module
