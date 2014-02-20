module DFT

# DFT plan where the inputs are an array of eltype T
abstract Plan{T}

import Base: show, size, eltype, *

eltype{T}(::Plan{T}) = T

##############################################################################
export fft, ifft, bfft, fft!, ifft!, bfft!,
       plan_fft, plan_ifft, plan_bfft, plan_fft!, plan_ifft!, plan_bfft!,
       rfft, irfft, brfft, plan_rfft, plan_irfft, plan_brfft

complexfloat{T<:FloatingPoint}(x::AbstractArray{Complex{T}}) = x

# return an Array, rather than similar(x), to avoid an extra copy for FFTW
# (which only works on StridedArray types).
complexfloat{T<:Complex}(x::AbstractArray{T}) = copy!(Array(typeof(float(one(T))), size(x)), x)
complexfloat{T<:FloatingPoint}(x::AbstractArray{T}) = copy!(Array(typeof(complex(one(T))), size(x)), x)
complexfloat{T<:Real}(x::AbstractArray{T}) = copy!(Array(typeof(complex(float(one(T)))), size(x)), x)

# implementations only need to provide plan_X(x, dims)
# for X in (:fft, :bfft, ...):
for f in (:fft, :bfft, :ifft, :fft!, :bfft!, :ifft!, :rfft)
    pf = symbol(string("plan_", f))
    @eval begin
        $f(x::AbstractArray) = $pf(x) * x
        $f(x::AbstractArray, dims) = $pf(x, dims) * x
        $pf(x::AbstractArray; kws...) = $pf(x, 1:ndims(x); kws...)
    end
end

# promote to a complex floating-point type (out-of-place only),
# so implementations only need Complex{Float} methods
for f in (:fft, :bfft, :ifft)
    @eval begin
        $f{T<:Real}(x::AbstractArray{T}, dims=1:ndims(x)) = $f(complexfloat(x), dims)
        $f{T<:Union(Integer,Rational)}(x::AbstractArray{Complex{T}}, dims=1:ndims(x)) = $f(complexfloat(x), dims)
    end
end
rfft{T<:Union(Integer,Rational)}(x::AbstractArray{T}, dims=1:ndims(x)) = rfft(float(x), dims)

# only require implementation to provide *(::Plan{T}, ::Array{T})
*{T}(p::Plan{T}, x::AbstractArray) = p * copy!(Array(x, T, size(x)), x)

# Implementations should also implement A_mul_B!(Y, plan, X) so as to support
# pre-allocated output arrays.  We don't define * in terms of A_mul_B!
# generically here, however, because of subtleties for in-place and rfft plans.

##############################################################################
# implementations only need to provide the unnormalized backwards FFT,
# similar to FFTW, and we do the scaling generically to get the ifft:

immutable ScaledPlan{T} <: Plan{T}
    p::Plan{T}
    scale::Number # not T, to avoid unnecessary promotion to Complex
end
ScaledPlan{T}(p::Plan{T}, scale::Number) = ScaledPlan{T}(p, scale)
ScaledPlan{T}(p::ScaledPlan{T}, α::Number) = ScaledPlan{T}(p.p, p.scale * α)

size(p::ScaledPlan) = size(p.p)

show(io::IO, p::ScaledPlan) = print(io, p.p, ", * ", p.scale)

*(p::ScaledPlan, x::AbstractArray) = scale!(p.p * x, p.scale)
*(α::Number, p::Plan) = ScaledPlan(p, α)
*(p::Plan, α::Number) = ScaledPlan(p, α)

# Normalization for ifft, given unscaled bfft, is 1/prod(dimensions)
normalization(T, sz, region) = one(T) / prod([sz...][[region...]])
realtype{T<:Real}(::Type{T}) = T
realtype{T<:Real}(::Type{Complex{T}}) = T
realtype{T}(x::AbstractArray{T}) = realtype(T)
normalization(X, region) = normalization(realtype(X), size(X), region)

plan_ifft(x::AbstractArray, region; kws...) =
    ScaledPlan(plan_bfft(x, region; kws...), normalization(x, region))
plan_ifft!(x::AbstractArray, region; kws...) =
    ScaledPlan(plan_bfft!(x, region; kws...), normalization(x, region))

##############################################################################
# Real-input DFTs are annoying because the output has a different size
# than the input if we want to gain the full factor-of-two(ish) savings
# For backward real-data transforms, we must specify the original length
# of the first dimension, since there is no reliable way to detect this
# from the data (we can't detect whether the dimension was originally even
# or odd).

for f in (:brfft, :irfft)
    pf = symbol(string("plan_", f))
    @eval begin
        $f(x::AbstractArray, d::Integer) = $pf(x, d) * x
        $f(x::AbstractArray, d::Integer, dims) = $pf(x, d, dims) * x
        $pf(x::AbstractArray, d::Integer;kws...) = $pf(x, d, 1:ndims(x);kws...)
    end
end

for f in (:brfft, :irfft)
    @eval begin
        $f{T<:Real}(x::AbstractArray{T}, d::Integer, dims=1:ndims(x)) = $f(complexfloat(x), d, dims)
        $f{T<:Union(Integer,Rational)}(x::AbstractArray{Complex{T}}, d::Integer, dims=1:ndims(x)) = $f(complexfloat(x), d, dims)
    end
end

function rfft_output_size(x::AbstractArray, region)
    d1 = first(region)
    osize = [size(X)...]
    osize[d1] = osize[d1]>>1 + 1
    return osize
end

function brfft_output_size(x::AbstractArray, d::Integer, region)
    d1 = first(region)
    osize = [size(X)...]
    @assert osize[d1] == d>>1 + 1
    osize[d1] = d
end

plan_irfft{T}(x::AbstractArray{Complex{T}}, d::Integer, region; kws...) = 
    ScaledPlan(plan_brfft(x, d, region; kws...),
               normalization(T, brfft_output_size(x, d, region), region))

##############################################################################

export fftshift, ifftshift

fftshift(x) = circshift(x, div([size(x)...],2))

function fftshift(x,dim)
    s = zeros(Int,ndims(x))
    s[dim] = div(size(x,dim),2)
    circshift(x, s)
end

ifftshift(x) = circshift(x, div([size(x)...],-2))

function ifftshift(x,dim)
    s = zeros(Int,ndims(x))
    s[dim] = -div(size(x,dim),2)
    circshift(x, s)
end

##############################################################################

# FFTW module (may move to an external package at some point):
include("fft/FFTW.jl")
export FFTW, dct, idct, dct!, idct!, plan_dct, plan_idct, plan_dct!, plan_idct!

end

