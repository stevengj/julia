module DFT

# DFT plan where the inputs are an array of type T
abstract Plan{T}

import Base: show, size, eltype, *

eltype{T}(::Plan{T}) = T

##############################################################################
export fft, ifft, bfft, fft!, ifft!, bfft!,
       plan_fft, plan_ifft, plan_bfft, plan_fft!, plan_ifft!, plan_bfft!

floatcomplex{T<:FloatingPoint}(x::AbstractArray{T}) = complex(x)
floatcomplex{T<:FloatingPoint}(x::AbstractArray{Complex{T}}) = x
floatcomplex{T<:Complex}(x::AbstractArray{T}) = float(x)
floatcomplex{T<:Real}(x::AbstractArray{T}) = copy!(similar(x, typeof(complex(float(one(T))))), x)

# implementations only need to provide plan_X(x, dims) and plan_X!(x, dims)
# for X in (:fft, :bfft, :ifft):
for f in (:fft, :bfft, :ifft, :fft!, :bfft!, :ifft!)
    pf = symbol(string("plan_", f))
    @eval begin
        $f(x::AbstractArray) = $pf(x) * x
        $f(x::AbstractArray, dims) = $pf(x, dims) * x
        $pf(x::AbstractArray) = $pf(x, 1:ndims(x))
    end
    # promote to a complex floating-point type (out-of-place only),
    # so implementations only need Complex{Float} methods
    if string(f)[end] != "!"
        @eval begin
            $f{T<:Real}(x::AbstractArray{T}, dims=1:ndims(x)) = $f(floatcomplex(x), dims)
            $f{T<:Integer}(x::AbstractArray{Complex{T}}, dims=1:ndims(x)) = $f(floatcomplex(x), dims)
        end
    end
end

##############################################################################
# implementations only need to provide the unnormalized backwards FFT,
# similar to FFTW, and we do the scaling generically to get the ifft:

immutable ScaledPlan{T} <: Plan{T}
    p::Plan{T}
    scale::T
end
ScaledPlan{T}(p::Plan{T}, scale::Number) = ScaledPlan{T}(p, scale)
ScaledPlan{T}(p::ScaledPlan{T}, α::Number) = ScaledPlan{T}(p, p.scale * α)

size(p::ScaledPlan) = size(p.p)

show(io::IO, p::ScaledPlan) = print(io, p.p, "; scaled by ", p.scale)

*(p::ScaledPlan, x::AbstractArray) = scale!(p.p * x, p.scale)
*(α::Number, p::Plan) = ScaledPlan(p, α)
*(p::Plan, α::Number) = ScaledPlan(p, α)

# Normalization for ifft
normalization(X, region) = one(eltype(X)) / prod([size(X)...][[region...]])

plan_ifft(x::AbstractArray, region) = ScaledPlan(plan_bfft(x, region),
                                               normalization(x, region))
plan_ifft!(x::AbstractArray, region) = ScaledPlan(plan_bfft!(x, region),
                                                normalization(x, region))

##############################################################################
# real-input DFTs are annoying because the output has a different size
# than the input if we want to gain the full factor-of-two(ish) savings



##############################################################################
# A DFT is unambiguously defined as just the identity operation for scalars

fft(x::Number) = x
ifft(x::Number) = x
bfft(x::Number) = x
rfft(x::Real) = x
irfft(x::Number, d::Integer) = d == 1 ? real(x) : throw(BoundsError())
brfft(x::Number, d::Integer) = d == 1 ? real(x) : throw(BoundsError())
fft(x::Number, dims) = length(dims) == 0 || dims[1] == 1 ? x : throw(BoundsError())
ifft(x::Number, dims) = length(dims) == 0 || dims[1] == 1 ? x : throw(BoundsError())
bfft(x::Number, dims) = length(dims) == 0 || dims[1] == 1 ? x : throw(BoundsError())
fft(x::Number, dims) = length(dims) == 0 || dims[1] == 1 ? x : throw(BoundsError())
rfft(x::Real, dims) = dims[1] == 1 ? x : throw(BoundsError())
irfft(x::Number, d::Integer, dims) = d == 1 && dims[1] == 1 ? real(x) : throw(BoundsError())
brfft(x::Number, d::Integer, dims) = d == 1 && dims[1] == 1 ? real(x) : throw(BoundsError())
##############################################################################

end
