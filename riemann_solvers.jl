# ---------------------------------------------------------------------------
# Riemann Solvers (HLL, HLLC)
# ---------------------------------------------------------------------------

function HLL(U_L::Vec3, U_R::Vec3, gamma::Float64)::Vec3
    W_L = conservative_to_primitive(U_L, gamma)
    W_R = conservative_to_primitive(U_R, gamma)
    F_L = compute_flux(U_L, gamma)
    F_R = compute_flux(U_R, gamma)
    a_L = sound_speed(W_L[3], W_L[1], gamma)
    a_R = sound_speed(W_R[3], W_R[1], gamma)
    u_L = W_L[2]
    u_R = W_R[2]

    S_L = min(u_L, u_R) - max(a_L, a_R)
    S_R = max(u_L, u_R) + max(a_L, a_R)

    if S_L > 0.0
        F_HLL = F_L
    elseif S_R < 0.0
        F_HLL = F_R
    else
        F_HLL = (S_R * F_L - S_L * F_R + S_R * S_L * (U_R - U_L)) / (S_R - S_L)
    end

    return F_HLL
end

function HLLC(U_L::Vec3, U_R::Vec3, gamma::Float64)::Vec3
    W_L = conservative_to_primitive(U_L, gamma)
    W_R = conservative_to_primitive(U_R, gamma)
    F_L = compute_flux(U_L, gamma)
    F_R = compute_flux(U_R, gamma)
    u_L = W_L[2]
    u_R = W_R[2]
    p_L = W_L[3]
    p_R = W_R[3]
    rho_L = W_L[1]
    rho_R = W_R[1]
    a_L = sound_speed(p_L, rho_L, gamma)
    a_R = sound_speed(p_R, rho_R, gamma)

    S_L = min(u_L, u_R) - max(a_L, a_R)
    S_R = max(u_L, u_R) + max(a_L, a_R)
    S_mid = ( p_R-p_L + rho_L*u_L*(S_L-u_L) - rho_R*u_R*(S_R-u_R) ) / ( rho_L*(S_L-u_L)-rho_R*(S_R-u_R) )

    if S_L > 0.0
        F_HLLC = F_L
    elseif S_R < 0.0
        F_HLLC = F_R
    elseif S_L <= 0.0 && 0.0 <= S_mid
        coef = rho_L * (S_L - u_L)/(S_L - S_mid)
        U1L_mid = coef
        U2L_mid = coef * S_mid
        E_L = U_L[3]
        U3L_mid = coef * ( E_L/rho_L + (S_mid-u_L)*( S_mid+p_L/(rho_L*(S_L-u_L)) ) )
        UL_mid = [U1L_mid, U2L_mid, U3L_mid]

        F_HLLC = F_L + S_L*(UL_mid - U_L)
    elseif S_mid <= 0.0 && 0.0 <= S_R
        coef = rho_R * (S_R - u_R)/(S_R - S_mid)
        U1R_mid = coef
        U2R_mid = coef * S_mid
        E_R = U_R[3]
        U3R_mid = coef * ( E_R/rho_R + (S_mid-u_R)*( S_mid+p_R/(rho_R*(S_R-u_R)) ) )
        UR_mid = [U1R_mid, U2R_mid, U3R_mid]

        F_HLLC = F_R + S_R*(UR_mid - U_R)
    end

    return F_HLLC
end
