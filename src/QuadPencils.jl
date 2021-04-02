module QuadPencils

using LinearAlgebra, SparseArrays, SuiteSparse

export linearize, deflate, eigbasis

### PencilScaling ##########################################################################
# Compute the optimal scaling
############################################################################################
struct PencilScaling{T}
    γ::T
    δ::T
end

function pencilscaling(A0::M, A1::M, A2::M; scaling = nothing, kw...) where {T,M<:AbstractMatrix{T}}
    scaling === nothing && return PencilScaling(T(1), T(1))
    n0, n1, n2 = opnorm.(Matrix.((A0, A1, A2)))
    τ = n1 / sqrt(n0 * n2)
    if τ <= 1
        γ = sqrt(n0 / n2)
        δ = 2 / (n0 + γ * n1)
    elseif scaling === -
        # distinct tropical roots, choose γ+ to favor large eigenvalues
        γ = n1 / n2
        δ = 1 / max(n2 * γ^2, n1 * γ, n0)
    else
        # distinct tropical roots, choose γ- to favor small eigenvalues
        γ = n0 / n1
        δ = 1 / max(n2 * γ^2, n1 * γ, n0)
    end
    return PencilScaling(T(γ), T(δ))
end

function scale!(A0, A1, A2, s::PencilScaling)
    γ, δ = scalingfactors(s)
    if γ != 1 || δ != 1
        A0 .*= γ^2 * δ
        A1 .*= γ * δ
        A2 .*= δ
    end
    return A0, A1, A2
end

scalingfactors(s::PencilScaling) = s.γ, s.δ

### QuadPencil #############################################################################
# Encodes the quadratic pencil problem
############################################################################################
struct QuadPencil{T<:Complex,M<:AbstractMatrix{T}}
    A0::M
    A1::M
    A2::M
    scaling::PencilScaling{T}
end

function quadpencil(A0, A1, A2; kw...)
    size(A0) == size(A1) == size(A2) && size(A0, 1) == size(A0, 2) ||
        throw(DimensionMismatch("Expected square matrices of the same size"))
    ptype = complex(promote_type(eltype(A0), eltype(A1), eltype(A2)))
    A0´ = copy_to_eltype(A0, ptype)
    A1´ = copy_to_eltype(A1, ptype)
    A2´ = copy_to_eltype(A2, ptype)
    s = pencilscaling(A0´, A1´, A2´; kw...)
    scale!(A0´, A1´, A2´, s)
    return QuadPencil(A0´, A1´, A2´, s)
end

# This also ensures any Adjoint/Transpose wrapper is removed
copy_to_eltype(A, ::Type{T}) where {T} = copy!(similar(A, T, size(A)), A)

Base.size(q::QuadPencil, n...) = size(q.A0, n...)

### Linearized Pencils #####################################################################
# Build second companion linearization, or its Q*C2*V rotation
############################################################################################
struct Linearization{T,M<:AbstractMatrix{T},R}
    A::M
    B::M
    V::M
    deflate_tol::R
end

Linearization(A::M, B::M, V::M; tol = zero(real(T))) where {M} = Linearization(A, B, V, tol)
Linearization(A::M, B::M) where {M} = Linearization(A, B, one(A), zero(real(T)))

linearize(As...; kw...) = linearize(quadpencil(As...; kw...); kw...)

function linearize(p::QuadPencil{T}; type = 1) where {T}
    A, B = AB(p, type)
    return Linearization(A, B, one(A), zero(T))
end

AB(p, type) = type == 1 ? C1(p) : type == 2 ? C2(p) :
    throw(ArgumentError("Unrecognized linearization `type`"))

C1(p) = [0I I; p.A0 p.A1], [I 0I; 0I -p.A2]
C2(p) = [p.A1 -I; p.A0 0I], [-p.A2 0I; 0I -I]

### deflate #############################################################################
# Deflates zero and infinite eigenvalues in a pencil
############################################################################################
deflate(A::AbstractMatrix, B::AbstractMatrix; kw...) = deflate(pqr(A), B; kw...)

function deflate(qrA::Factorization, B; kw...)
    Adef0, Bdef0, Pdef0 = deflate_zeros(qrA, B; kw...)
    Bdef, Adef, Pdef∞ = deflate_infinites(Adef0, Bdef0; kw...)
    Pdef = Pdef0 * Pdef∞
    return Linearization(Adef, Bdef, Pdef; kw...)

end

function deflate_zeros(qrA::Factorization{T}, B::AbstractMatrix; atol = default_tol(T)) where {T}
    Q´A = getRP´(qrA)
    #Qdef = getQ´(qrA)
    Q´B = qrA.Q' * B
    n = size(qrA, 1)
    r = nonzero_rows(RP´, atol)
    A1 = view(Q´A, 1:r, :)
    B1 = view(Q´B, 1:r, :)
    B2 = view(Q´B, r+1:n, :)
    qrB2´ = pqr(B2')
    P = getQ_rev(qrB2´)   # (Q' * B2')' = B2 * Q
    Pdef = view(P, :, 1:r)
    Adef = A1 * Pdef
    Bdef = B1 * Pdef
    return Adef, Bdef, Pdef
end

function deflate_infinites(A, B; kw...)
    Bdef, Adef, Pdef = deflate_zeros(pqr(B), A; kw...)
    return Adef, Bdef, Pdef
end

default_tol(::Type{T}) where {T} = sqrt(eps(real(T)))

### Tools ##################################################################################

pqr(a::SparseMatrixCSC) = qr(a)
pqr(a) = qr(a, Val(true))

# getQ(qr::Factorization) = qr.Q * Idense(size(qr, 1))
# getQ(qr::SuiteSparse.SPQR.QRSparse) =
#     Isparse(size(qr, 1), :, qr.prow) * sparse(qr.Q * Idense(size(qr, 1)))

getQ_rev(qr::Factorization) = qr.Q * Idense(size(qr, 1), reverse(1:size(qr,1)))
getQ_rev(qr::SuiteSparse.SPQR.QRSparse) =
    Isparse(size(qr, 1), :, qr.prow) * sparse(qr.Q * Idense(size(qr, 1), reverse(1:size(qr,1))))

getQ´(qr::Factorization) = qr.Q' * Idense(size(qr, 1))
getQ´(qr::SuiteSparse.SPQR.QRSparse) =
    sparse((qr.Q * Idense(size(qr, 1)))') * Isparse(size(qr,1), qr.prow, :)

# getQ´_rev(qr::Factorization) = Idense(size(qr, 1), reverse(1:size(qr,1))) * qr.Q'
# getQ´_rev(qr::SuiteSparse.SPQR.QRSparse) =
#     sparse((qr.Q * Idense(size(qr, 1), reverse(1:size(qr,1))))') * Isparse(size(qr,1), qr.prow, :)

getRP´(qr::Factorization) = qr.R * qr.P'
getRP´(qr::SuiteSparse.SPQR.QRSparse) = qr.R * Isparse(size(qr, 2), qr.pcol, :)

# getPR´(qr::Factorization) = qr.P * qr.R'
# getPR´(qr::SuiteSparse.SPQR.QRSparse) = Isparse(size(qr, 2), :, qr.pcol) * qr.R'

Idense(n) = Matrix(I, n, n)

function Idense(n, cols)
    m = zeros(Bool, n, length(cols))
    for (j, col) in enumerate(cols)
        m[col, j] = true
    end
    return m
end

# Equivalent to I(n)[rows, cols], but faster
Isparse(n, rows, cols) = Isparse(inds(rows, n), inds(cols, n))

function Isparse(rows, cols)
    rowval = Int[]
    nzval = Bool[]
    colptr = Vector{Int}(undef, length(cols) + 1)
    colptr[1] = 1
    for (j, col) in enumerate(cols)
        push_rows!(rowval, nzval, rows, col)
        colptr[j+1] = length(rowval) + 1
    end
    return SparseMatrixCSC(length(rows), length(cols), colptr, rowval, nzval)
end

inds(::Colon, n) = 1:n
inds(is, n) = is

function push_rows!(rowval, nzval, rows::AbstractUnitRange, col)
    if col in rows
        push!(rowval, 1 + rows[1 + col - first(rows)] - first(rows))
        push!(nzval, true)
    end
    return nothing
end

function push_rows!(rowval, nzval, rows, col)
    for (j, row) in enumerate(rows)
        if row == col
            push!(rowval, j)
            push!(nzval, true)
        end
    end
    return nothing
end

function nonzero_rows(m::AbstractMatrix{T}, atol = sqrt(eps(real(T)))) where {T}
    n = 0
    for row in eachrow(m)
        all(z -> abs(z) < atol, row) && break
        n += 1
    end
    return n
end

# function chop!(A::AbstractArray{T}, atol = sqrt(eps(real(T)))) where {T}
#     for (i, a) in enumerate(A)
#         if abs(a) < atol
#             A[i] = zero(T)
#         elseif abs(a) > 1/atol || isnan(a)
#             A[i] = T(Inf)
#         end
#     end
#     return A
# end

end
