module DSP

export filt, deconv, conv, conv2, xcorr

function filt{T<:Number}(b::Union(AbstractVector{T}, T),a::Union(AbstractVector{T}, T),x::AbstractVector{T})
    if isempty(b); error("b must be non-empty"); end
    if isempty(a); error("a must be non-empty"); end
    if a[1]==0; error("a[1] must be nonzero"); end

    as = length(a)
    bs = length(b)
    sz = max(as, bs)

    if sz == 1
        return (b[1]/a[1]).*x
    end

    if bs<sz
        newb = zeros(T,sz)
        newb[1:bs] = b
        b = newb
    end

    xs = size(x,1)
    y = Array(T, xs)
    silen = sz-1
    si = zeros(T, silen)

    if a[1] != 1
        norml = a[1]
        a ./= norml
        b ./= norml
    end

    if as > 1
        if as<sz
            newa = zeros(T,sz)
            newa[1:as] = a
            a = newa
        end

        @inbounds begin
            for i=1:xs
                y[i] = si[1] + b[1]*x[i]
                for j=1:(silen-1)
                    si[j] = si[j+1] + b[j+1]*x[i] - a[j+1]*y[i]
                end
                si[silen] = b[silen+1]*x[i] - a[silen+1]*y[i]
            end
        end
    else
        @inbounds begin
            for i=1:xs
                y[i] = si[1] + b[1]*x[i]
                for j=1:(silen-1)
                    si[j] = si[j+1] + b[j+1]*x[i]
                end
                si[silen] = b[silen+1]*x[i]
            end
        end
    end
    return y
end

function deconv{T}(b::StridedVector{T}, a::StridedVector{T})
    lb = size(b,1)
    la = size(a,1)
    if lb < la
        return [zero(T)]
    end
    lx = lb-la+1
    x = zeros(T, lx)
    x[1] = 1
    filt(b, a, x)
end

function conv{T<:Base.LinAlg.BlasFloat}(u::StridedVector{T}, v::StridedVector{T})
    nu = length(u)
    nv = length(v)
    n = nu + nv - 1
    np2 = n > 1024 ? nextprod([2,3,5], n) : nextpow2(n)
    upad = [u, zeros(T, np2 - nu)]
    vpad = [v, zeros(T, np2 - nv)]
    if T <: Real
        p = plan_rfft(upad)
        y = irfft((p*upad).*(p*vpad), np2)
    else
        p = plan_fft!(upad)
        y = ifft!((p*upad).*(p*vpad))
    end
    return y[1:n]
end
conv{T<:Integer}(u::StridedVector{T}, v::StridedVector{T}) = int(conv(float(u), float(v)))
conv{T<:Integer, S<:Base.LinAlg.BlasFloat}(u::StridedVector{T}, v::StridedVector{S}) = conv(float(u), v)
conv{T<:Integer, S<:Base.LinAlg.BlasFloat}(u::StridedVector{S}, v::StridedVector{T}) = conv(u, float(v))

function conv2{T}(u::StridedVector{T}, v::StridedVector{T}, A::StridedMatrix{T})
    m = length(u)+size(A,1)-1
    n = length(v)+size(A,2)-1
    B = zeros(T, m, n)
    B[1:size(A,1),1:size(A,2)] = A
    u = fft([u;zeros(T,m-length(u))])
    v = fft([v;zeros(T,n-length(v))])
    C = ifft(fft(B) .* (u * v.'))
    if T <: Real
        return real(C)
    end
    return C
end

function conv2{T}(A::StridedMatrix{T}, B::StridedMatrix{T})
    sa, sb = size(A), size(B)
    At = zeros(T, sa[1]+sb[1]-1, sa[2]+sb[2]-1)
    Bt = zeros(T, sa[1]+sb[1]-1, sa[2]+sb[2]-1)
    At[1:sa[1], 1:sa[2]] = A
    Bt[1:sb[1], 1:sb[2]] = B
    p = plan_fft(At)
    C = ifft((p*At).*(p*Bt))
    if T <: Real
        return real(C)
    end
    return C
end
conv2{T<:Integer}(A::StridedMatrix{T}, B::StridedMatrix{T}) = int(conv2(float(A), float(B)))
conv2{T<:Integer}(u::StridedVector{T}, v::StridedVector{T}, A::StridedMatrix{T}) = int(conv2(float(u), float(v), float(A)))

function xcorr(u, v)
    su = size(u,1); sv = size(v,1)
    if su < sv
        u = [u;zeros(eltype(u),sv-su)]
    elseif sv < su
        v = [v;zeros(eltype(v),su-sv)]
    end
    flipud(conv(flipud(u), v))
end

end # module
