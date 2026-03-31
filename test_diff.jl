using StaticArrays
using Printf

const Vec3 = SVector{3, Float64}
const R_UNIVERSAL = 8.314_462_62
const GAMMA = 1.4
const MW = 28.96e-3

get_rho(p, T) = p * MW / (R_UNIVERSAL * T)
con2prim(U) = Vec3(U[1], U[2]/U[1], (GAMMA-1)*(U[3]-0.5*U[2]^2/U[1]))
prim2con(W) = Vec3(W[1], W[1]*W[2], W[3]/(GAMMA-1)+0.5*W[1]*W[2]^2)
eflux(U) = (p=(GAMMA-1)*(U[3]-0.5*U[2]^2/U[1]); Vec3(U[2], U[2]^2/U[1]+p, U[2]/U[1]*(U[3]+p)))
snd(p, rho) = sqrt(GAMMA*p/rho)

function compute_dt(U_arr, dx, cfl)
    ms = 0.0
    for U in U_arr
        W = con2prim(U); ms = max(ms, abs(W[2])+snd(W[3],W[1]))
    end
    cfl * dx / ms
end

function HLL(UL::Vec3, UR::Vec3)::Vec3
    WL=con2prim(UL); WR=con2prim(UR); FL=eflux(UL); FR=eflux(UR)
    aL=snd(WL[3],WL[1]); aR=snd(WR[3],WR[1])
    SL=min(WL[2],WR[2])-max(aL,aR); SR=max(WL[2],WR[2])+max(aL,aR)
    SL>0 ? FL : SR<0 ? FR : (SR*FL-SL*FR+SR*SL*(UR-UL))/(SR-SL)
end

function HLLC(UL::Vec3, UR::Vec3)::Vec3
    WL=con2prim(UL); WR=con2prim(UR); FL=eflux(UL); FR=eflux(UR)
    uL,pL,rL = WL[2],WL[3],WL[1]; uR,pR,rR = WR[2],WR[3],WR[1]
    aL=snd(pL,rL); aR=snd(pR,rR)
    SL=min(uL,uR)-max(aL,aR); SR=max(uL,uR)+max(aL,aR)
    Sm=(pR-pL+rL*uL*(SL-uL)-rR*uR*(SR-uR))/(rL*(SL-uL)-rR*(SR-uR))
    if SL>0; return FL; elseif SR<0; return FR
    elseif SL<=0 && 0<=Sm
        c=rL*(SL-uL)/(SL-Sm)
        Um=[c, c*Sm, c*(UL[3]/rL+(Sm-uL)*(Sm+pL/(rL*(SL-uL))))]
        return Vec3((FL+SL*(Um-UL))...)
    else
        c=rR*(SR-uR)/(SR-Sm)
        Um=[c, c*Sm, c*(UR[3]/rR+(Sm-uR)*(Sm+pR/(rR*(SR-uR))))]
        return Vec3((FR+SR*(Um-UR))...)
    end
end

function run_test()
    dx=1e-2; cfl=0.9; t_max=0.008; ng=1
    n_cells=round(Int,10.0/dx)+1; np=n_cells+2*ng
    rL=get_rho(1e5,348.24); rR=get_rho(1e4,278.24)
    x=[-5.0-ng*dx+dx*(i-1) for i in 1:np]
    U0=[prim2con(x[i]<0 ? Vec3(rL,0.0,1e5) : Vec3(rR,0.0,1e4)) for i in 1:np]
    Uh=copy(U0); Uc=copy(U0); bh=copy(Uh); bc=copy(Uc)
    is_=ng+1; ie_=np-ng; t=0.0; step=0

    while t < t_max
        dt=min(compute_dt(Uh,dx,cfl), compute_dt(Uc,dx,cfl))
        copyto!(bh,Uh); copyto!(bc,Uc)
        for i in is_:ie_
            bh[i]=Uh[i]+dt*(-(HLL(Uh[i],Uh[i+1])-HLL(Uh[i-1],Uh[i]))/dx)
            bc[i]=Uc[i]+dt*(-(HLLC(Uc[i],Uc[i+1])-HLLC(Uc[i-1],Uc[i]))/dx)
        end
        for g in 1:ng; bh[g]=bh[is_]; bh[end-g+1]=bh[ie_]; bc[g]=bc[is_]; bc[end-g+1]=bc[ie_]; end
        Uh,bh=bh,Uh; Uc,bc=bc,Uc; t+=dt; step+=1
    end

    rho_h=[con2prim(Uh[i])[1] for i in 1:np]
    rho_c=[con2prim(Uc[i])[1] for i in 1:np]

    @printf("t = %.6e, step = %d\n", t, step)
    @printf("Max |rho_HLL - rho_HLLC| = %.6e\n", maximum(abs.(rho_h-rho_c)))
    println("\n   x        rho_HLL    rho_HLLC     diff")
    for i in 1:np
        d = abs(rho_h[i]-rho_c[i])
        if 0.0<=x[i]<=3.0 && d>0.001
            @printf("%7.3f   %10.6f  %10.6f  %+.6e\n", x[i], rho_h[i], rho_c[i], rho_c[i]-rho_h[i])
        end
    end
end

run_test()
