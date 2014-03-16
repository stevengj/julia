import Base.DFT: Plan
import Base: plan_fft, plan_bfft, *, A_mul_B!

# 1d Cooley-Tukey FFTs, using an FFTW-like (version 2) approach: automatic
# generation of fixed-size FFT kernels (with and without twiddle factors)
# which are combined to make arbitrary-size FFTs (plus generic base
# cases for large prime factors).

# return biggest factor <= sqrt(n):
function biggest_factor(n)
    for r = isqrt(n):-1:2
        if n % r == 0
            return r
        end
    end
    return 1
end

#############################################################################
# FFT code generation:

# Choose the default radix for pregenerated FFT of length `n`.  To get
# a cache-oblivious algorithm (for register usage), we will choose the
# biggest factor of `n` that is $\le \sqrt{n}$.
function choose_radix(n)
    # TODO: find relatively-prime radices for prime-factor algorithm?
    return biggest_factor(n)
end

# `twiddle(T, forward, n, k, x)` multiplies the expression `x` by
# $\omega_n^k$, where $\omega_n$ is the `n`-th primitive root of unity
# for the field `T`, returning a new expression.  For complex `T`,
# $\omega_n = \exp(s*2\pi i/n)$, where `s=-1` for `forward=true` and
# `s=+1` for `forward=false`.  (These are traditionally called
# "twiddle factors" in FFT algorithms.) Note also that algebraic
# simplifications occur for several `k`.
#
# (In FFTW's generator, we first generate all expressions in terms of
# real arithmetic and then apply a simplifier to eliminate
# multiplications by `1` and similar.  That is a more powerful
# approach, but this is quicker to implement.)
function twiddle{Tr<:FloatingPoint}(T::Type{Complex{Tr}}, forward::Bool, n::Integer, k::Integer, x)
    k == 0 && return x
    2k == n && return :(-$x)
    if 4k == n || 4k == 3n
        tmpvar = gensym("twiddle")
        if (4k == n && !forward) || (4k == 3n && forward) # ω = +im
            return :(let $tmpvar=$x; complex(-imag($tmpvar), real($tmpvar)); end)
        else  # ω = -im
            return :(let $tmpvar=$x; complex(imag($tmpvar), -real($tmpvar)); end)
        end
    end
    if 8k == n || 8k == 3n || 8k == 5n || 8k == 7n # ω = [(1,-1), (-1,-1), (-1,1), or (1,1)] * sqrt(0.5)
        sqrthalf = sqrt(one(Tr)/2)
        tmpvar = gensym("twiddle")
        if (8k == n && forward) || (8k == 7n && !forward)
            return :(let $tmpvar=$x; $sqrthalf*complex(real($tmpvar)+imag($tmpvar), imag($tmpvar)-real($tmpvar)); end)
        elseif (8k == 3n && forward) || (8k == 5n && !forward)
            return :(let $tmpvar=$x; $sqrthalf*complex(imag($tmpvar)-real($tmpvar), -real($tmpvar)-imag($tmpvar)); end)
        elseif (8k == 5n && forward) || (8k == 3n && !forward)
            return :(let $tmpvar=$x; $sqrthalf*complex(-real($tmpvar)-imag($tmpvar), real($tmpvar)-imag($tmpvar)); end)
        elseif (8k == 7n && forward) || (8k == n && !forward)
            return :(let $tmpvar=$x; $sqrthalf*complex(real($tmpvar)-imag($tmpvar), real($tmpvar)+imag($tmpvar)); end)
        end
    end
    # make sure trig factors are correctly rounded:
    c, s = with_bigfloat_precision(2*precision(one(Tr))) do
        φ = (big(2) * k) / n
        convert(Tr, cospi(φ)), convert(Tr, sinpi(φ))
    end
    ω = complex(c, forward ? -s : s)
    return :($ω * $x)
end

# Like `fftgen`, below, but generates the naïve $\Theta(n^2)$ DFT algorithm:
function dftgen(T, forward::Bool, n::Integer, x, y)
    n == 1 && return :($(y(0)) = $(x(0)))
    tmpvars = Symbol[ gensym(string("dftgen_", j)) for j in 0:n-1 ]
    n == 2 && return Expr(:let, Expr(:block, :($(y(0)) = $(tmpvars[1]) + $(tmpvars[2])), :($(y(1)) = $(tmpvars[1]) - $(tmpvars[2]))), :($(tmpvars[1]) = $(x(0))), :($(tmpvars[2]) = $(x(1))))
    Expr(:let,
         Expr(:block, [:($(y(k)) = $(Expr(:call, :+, [twiddle(T, forward, n, j*k, tmpvars[j+1]) for j in 0:n-1]...))) for k=0:n-1]...),
         [:($(tmpvars[j+1]) = $(x(j))) for j = 0:n-1]...)
end

# `fftgen(n, true, x, y)` generates an expression (`Expr`) for an FFT
# of length `n` with inputs `x(i)` and outputs `y(i)` for `i=0:n-1`.
# For `forward=false`, the unscaled backward transform is returned.
# Note that `x` and `y` are *functions* returning expressions.  `T` is
# the type of the field over which we perform the DFT,
# e.g. `Complex{Float64}`.
function fftgen(T, forward::Bool, n::Integer, x, y)
    r = choose_radix(n)
    if r == 1
        return dftgen(T, forward, n, x, y)
    # TODO: prime-factor algorithm?  split-radix?
    else # radix-r Cooley-Tukey step
        m = div(n, r)
        # store results of first r sub-transforms in r*m temporary variables
        z = Symbol[ gensym(string("fftgen_", j1, "_", k2))
                    for j1 = 0:r-1, k2 = 0:m-1 ]
        # get expressions to perform r sub-FFTs of length m
        Fm = Expr[ fftgen(T, forward, m, 
                          j2 -> x(r*j2+j1), k2 -> z[j1+1,k2+1])
                   for j1 = 0:r-1 ]
        # get expressions to perform m sub-FFTs of length r
        Fr = Expr[ fftgen(T, forward, r, 
                          j1 -> twiddle(T, forward, n, j1*k2, z[j1+1,k2+1]),
                          k1 -> y(m*k1+k2))
                   for k2 in 0:m-1 ]
        Expr(:block, [Expr(:local, Z) for Z in z]..., Fm..., Fr...)
    end
end
fftgen(T, forward::Bool, n::Integer, X::Symbol, Y::Symbol) = fftgen(T, forward, n, j -> :($X[$(j+1)]), k -> :($Y[$(k+1)]))

# Analogous to FFTW's nontwiddle codelets (direct solvers), we
# generate a bunch of solvers for small fixed sizes.  Each solver is
# of the form `_fft_N(vn, X, x0, xs, xvs, Y, y0, ys, yvs)` and
# computes `i in 0:vn-1` transforms, with the `i`-th transform
# performing `X[x0 + xvs*i + (0:N-1)*xs] = fft(Y[x0 + yvs*i +
# (0:N-1)*ys])`.  Each such solver is generated by the
# `@nontwiddle(T,forward,n)` macro:
nontwiddle_name(forward::Bool, n::Integer) = symbol(string(forward ? "_fft_" : "_bfft_", n))
macro nontwiddle(T, forward, n)
    name = nontwiddle_name(forward, n)
    quote
        function $(esc(name))(vn::Integer, 
                              X::AbstractArray{$T},
                              x0::Integer, xs::Integer, xvs::Integer,
                              Y::AbstractArray{$T},
                              y0::Integer, ys::Integer, yvs::Integer)
            for i in 0:vn-1
                $(fftgen(eval(T), forward, n, 
                         j -> :(X[(x0 + xvs*i) + xs*$j]), 
                         k -> :(Y[(y0 + yvs*i) + ys*$k])))
            end
            Y
        end
    end
end

# Analogous to FFTW's twiddle codelets, we also generate solvers that
# perform *in-place* FFTs of small fixed sizes where the data is
# pre-multipled by a precomputed 2d array `W[j+1,i+1]` of twiddle
# factors (with `W[1,_] = 1`).  These are called `_twiddle_N(vn, X,
# x0, xs, xvs, W)`, and the meaning of the parameter is otherwise
# identical to the nontwiddle codelets with `Y=X`.
twiddle_name(forward::Bool, n::Integer) = symbol(string(forward ? "_twiddle_" : "_btwiddle_", n))
macro twiddle(T, forward, n)
    name = twiddle_name(forward, n)
    quote
        function $(esc(name))(vn::Integer, X::AbstractArray{$T}, 
                              x0::Integer, xs::Integer, xvs::Integer,
                              W::AbstractMatrix{$T})
            for i in 0:vn-1
                $(fftgen(eval(T), forward, n, 
                         j -> j == 0 ? :(X[(x0 + xvs*i) + xs*$j]) : 
                              :(W[$(j+1),i+1] * X[(x0 + xvs*i) + xs*$j]),
                         j -> :(X[(x0 + xvs*i) + xs*$j])))
            end
            X
        end
    end
end

# Now, we will generate nontwiddle and twiddle kernels for a set of
# fixed sizes, to be composed to build an arbitrary-$n$ FFT algorithm.
const fft_kernel_sizes = Set(1:10..., 12, 14, 15, 16, 20, 25, 32)
const nontwiddle_kernels = Dict{(Bool,Int), Function}()
const twiddle_kernels = Dict{(Bool,Int), Function}()
for forward in (true,false)
    for n in fft_kernel_sizes
        for T in (Complex{Float32}, Complex{Float64})
            @eval @nontwiddle($T, $forward, $n)
            @eval @twiddle($T, $forward, $n)
        end
        @eval nontwiddle_kernels[$forward,$n] = $(nontwiddle_name(forward,n))
        @eval twiddle_kernels[$forward,$n] = $(twiddle_name(forward,n))
    end
end
const fft_kernel_sizes_sorted = sort!(Int[n for n in fft_kernel_sizes],
                                      rev=true)

#############################################################################
# Combining pregenerated kernels into generic-size FFT plans:

# Break n into a series of factors, avoiding small kernels if possible
function fft_factors(n::Integer)
    factors = Int[]
    if n == 1
        push!(factors, 1)
    else
        m = n
        for r in fft_kernel_sizes_sorted
            if r > 1
                while m % r == 0
                    push!(factors, r)
                    m = div(m, r)
                end
                m == 1 && break
            end
        end
        # sometimes there will be a small factor (e.g. 2) left over at the end;
        # try to combine this with larger earlier factors if possible:
        if length(factors) > 1
            for i = 1:length(factors)-1
                factors[end] >= 16 && break
                while factors[i] % 2 == 0 && div(factors[i], 2) > factors[end]
                    factors[i] = div(factors[i], 2)
                    factors[end] *= 2
                end
            end
        end
        # get any leftover prime factors:
        for (f,k) in factor(m)
            for i=1:k
                push!(factors, f)
            end
        end
    end
    factors
end

# now, we define a CTPlan (Cooley-Tukey plan) as a sequence of twiddle steps
# followed by a nontwiddle step:

abstract TwiddleStep{T}
abstract NontwiddleStep{T}

immutable CTPlan{T,forward} <: Plan{T}
    n::Int
    tsteps::Vector{TwiddleStep{T}}
    nstep::NontwiddleStep{T}
end

# steps for pregenerated kernels: 
immutable TwiddleKernelStep{T} <: TwiddleStep{T}
    r::Int # radix
    m::Int # n / r
    kernel::Function
    W::Array{T}
    function TwiddleKernelStep(n::Int, r::Int, forward::Bool)
        m = div(n, r)
        twopi = forward ? -2π : 2π
        new(r, m, twiddle_kernels[forward, r],
            T[exp((twopi*mod(j1*k2,n)/n)*im) for j1=0:r-1, k2=0:m-1])
    end
end
function applystep{T}(ts::TwiddleKernelStep, y::AbstractVector{T}, y0, ys)
    ts.kernel(ts.m, y, y0, ts.m * ys, ys, ts.W)
end

immutable NontwiddleKernelStep{T} <: NontwiddleStep{T}
    kernel::Function
    NontwiddleKernelStep(n::Int, forward::Bool) =
        new(nontwiddle_kernels[forward, n])
end
function applystep{T}(ns::NontwiddleKernelStep, r, m,
                      x::AbstractVector{T}, x0, xs,
                      y::AbstractVector{T}, y0, ys)
    ns.kernel(r, x,x0,xs*r,xs, y,y0,ys,ys*m)
end

typealias CTComplex Union(Complex64,Complex128)

function CTPlan{T<:CTComplex}(::Type{T}, forward::Bool, n::Int)
    factors = fft_factors(n)
    m = n
    tsteps = Array(TwiddleStep{T}, length(factors)-1)
    for i = 1:length(tsteps)
        if factors[i] in fft_kernel_sizes
            tsteps[i] = TwiddleKernelStep{T}(m, factors[i], forward)
        else
            error("generic factors not implemented yet")
        end
        m = tsteps[i].m
    end
    @assert m == factors[end]
    if m in fft_kernel_sizes
        nstep = NontwiddleKernelStep{T}(m, forward)
    else
        error("generic base case not implemented yet")
    end
    CTPlan{T,forward}(n, tsteps, nstep)
end

function plan_fft{T<:CTComplex}(x::AbstractVector{T}, dims)
    collect(dims) != [1] && throw(ArgumentError("invalid fft dims"))
    CTPlan(T, true, length(x))
end
function plan_bfft{T<:CTComplex}(x::AbstractVector{T}, dims)
    collect(dims) != [1] && throw(ArgumentError("invalid fft dims"))
    CTPlan(T, false, length(x))
end

function applystep{T}(p::CTPlan{T}, 
                      x::AbstractVector{T}, x0, xs,
                      y::AbstractVector{T}, y0, ys,
                      step::Int)
    nsteps = length(p.tsteps)
    if step > nsteps
        applystep(p.nstep, 1,p.n, x,x0,xs, y,y0,ys)
    else
        # decimation in time: perform r DFTs of length m
        tstep = p.tsteps[step]
        m = tstep.m
        r = tstep.r
        if step == nsteps
            applystep(p.nstep, r,m, x,x0,xs, y,y0,ys)
        else
            xs_ = xs*r
            x0_ = x0
            y0_ = y0
            for i = 1:r-1
                applystep(p, x,x0_,xs_, y,y0_,ys, step+1)
                x0_ += xs
                y0_ += m
            end
            applystep(p, x,x0_,xs_, y,y0_,ys, step+1)
        end
        # combine sub-transforms with twiddle step:
        applystep(tstep, y,y0,ys)
    end
end

function A_mul_B!{T}(y::AbstractVector{T}, p::CTPlan{T}, x::AbstractVector{T}) 
    length(y) != length(x) && throw(BoundsError())
    applystep(p, x,1,1, y,1,1, 1)
    return y
end

*{T}(p::CTPlan{T}, x::AbstractVector{T}) = A_mul_B!(similar(x), p, x)

#############################################################################
