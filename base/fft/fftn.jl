# Multi-dimensional FFTs based on the 1d FFTs in ctfft.jl

include("ctfft.jl")

immutable MultiDimPlan{T,forward} <: Plan{T}
    p::Vector{CTPlan{T,forward}} # 1d plans along each transformed dimension,
                                 # with p.n == 0 for untransformed dims.
    w::Vector{T} # workspace (for in-place 1d transforms)
end

# create null plans for untransformed dimensions
immutable NullNontwiddleStep{T} <: NontwiddleStep{T} end
CTPlan(T,forward) = CTPlan{T,forward}(0, Array(TwiddleStep{T},0),
                                      NullNontwiddleStep{T}())

summary{T,forw}(p::MultiDimPlan{T,forw}) =
    string(forw ? "forward" : "backward", length(p.p),
           "-dimensional MultiDimPlan{$T} of size ",
           join(map(p -> p.n == 0 ? "_" : string(p.n), p.p), "x"))

function show(io::IO, p::MultiDimPlan)
    print(io, summary(p), ", via:\n",
          join(map(string, p.p), "\n"))
end

function MultiDimPlan{T<:Complex}(::Type{T}, forward::Bool, dims, sz)
    sdims = sort(Int[d for d in dims])
    N = length(sz)
    p = Array(CTPlan{T,forward}, N)
    i = 0
    for d in sdims
        (d < 0 || d > N) && throw(ArgumentError("invalid dimension $d"))
        d == i && throw(ArgumentError("dimension $d specified twice"))
        for j = i+1:d-1
            p[j] = CTPlan(T,forward) # non-transformed dimensions
        end
        p[d] = CTPlan(T, forward, sz[d])
        i = d
    end
    w = Array(T, length(sdims) <= 1 ? 0 : maximum(sz[sdims[1:end-1]]))
    MultiDimPlan{T,forward}(p, w)
end

plan_fft{Tr<:FloatingPoint}(x::AbstractArray{Complex{Tr}}, dims) =
    MultiDimPlan(Complex{Tr}, true, dims, size(x))
plan_bfft{Tr<:FloatingPoint}(x::AbstractArray{Complex{Tr}}, dims) =
    MultiDimPlan(Complex{Tr}, false, dims, size(x))

# recursive execution of a MultiDim plan, starting at dimension d, for
# strided arrays (so that we can use linear indexing):
function applydims{T}(p::MultiDimPlan{T}, d,
                      x::StridedArray{T}, x0, y::StridedArray{T}, y0)
    if d == ndims(y)
        if p.p[d].n != 0
            applystep(p.p[d], x,x0,stride(x,d), y,y0,stride(y,d), 1)
        end
    else
        sx = stride(x,d)
        sy = stride(y,d)
        y0_ = y0
        for i = 1:size(x,d)
            applydims(p, d+1, x,x0, y,y0)
            x0 += sx
            y0 += sy
        end
        if p.p[d].n != 0
            applydim(p, d, d+1, y::StridedArray{T}, y0_)
        end
    end
end
# apply p to dimension d of y, in-place, looping over dimensions >= k
function applydim{T}(p::MultiDimPlan{T}, d, k, y::StridedArray{T}, y0)
    sy_k = stride(y,k)
    if k == ndims(y)
        P = p.p[d]
        w = p.w
        sy = stride(y,d)
        ny = size(y,d)
        for i = 1:size(y,k)
            applystep(P, y,y0,sy, w,1,1, 1)
            jy = y0
            @inbounds for j = 1:ny
                 y[jy] = w[j]
                jy += sy
            end
            y0 += sy_k
        end
    else
        for i = 1:size(y,k)
            applydim(p, d, k+1, y, y0)
            y0 += sy_k
        end
    end
end

function A_mul_B!{T}(y::StridedArray{T},
                     p::MultiDimPlan{T}, x::StridedArray{T}) 
    N = ndims(x)
    ndims(y) != N && throw(BoundsError())
    for i = 1:N
        P = p.p[i]
        P.n == 0 || (P.n == size(x,i) == size(y,i)) || throw(BoundsError())
    end
    if N > 0
        applydims(p, 1, x,1, y,1)
    else
        y[1] = x[1]
    end
    return y
end

*{T}(p::MultiDimPlan{T}, x::StridedArray{T}) = A_mul_B!(similar(x), p, x)
