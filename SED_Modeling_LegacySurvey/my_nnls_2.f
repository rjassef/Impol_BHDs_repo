ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Implementation of the NNLS algorithm of Lawson & Hanson (1974)
c     
c     The main difference is that this code uses the numerical recipes
c     SVD algorithm to solve the E_p * z = b equation in step 6 and that
c     the E_p matrix is not only cut in columns but also in rows. The
c     latter seems to make a huge difference.
c
c     The code solves the problem s*x = b, where 
c
c     a is a m x n matrix with declared dimensions maxdim x maxdim.
c
c     b is a m component vector of declared dimension maxdim.
c     
c     x is a n component vector of declared dimension maxdim.
c
c
c     The difference with my_nnls is that the code decides what w_j to
c     use based on solving w = E^T x f rather on solving the Kuhn-Tacker
c     equation, just as my_nnls.f, but cannot exit unless the
c     Kuhn-Tacker conditions have been met (either subspace Z is empty
c     or w_j<0 for all j where w = E^T (f-Ex).
c
c     Bugs: In svdcmp, condition of a minimum of 30 iterations has been
c     changed for 60.
c
c
c     Written by: Roberto J. Assef (02/20/2008)
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      subroutine my_nnls_2(a,maxdim,m,n,b,x,MODE,its,iop)
      implicit real*8(a-h,o-z)

      real*8 a(maxdim,maxdim),a_t(maxdim,maxdim)
      real*8 b(maxdim),x(maxdim)
      real*8 w(maxdim),z(maxdim),zaux(maxdim)

      real*8 asave(maxdim,maxdim),bsave(maxdim)
      real*8 a_p(maxdim,maxdim)

      real*8 tempw(maxdim),tempv(maxdim,maxdim)

      integer p(maxdim)

      integer test_kuhn_tacker

      external transpose


c     Set P:=NULL, Z:={1,2,..,n} and x:=0.
 1    continue
      MODE = 1
      iter = 0
      itermax = 100*n
      do l=1,n
         p(l) = 0
         x(l) = 0.d0
      enddo
      call transpose(a,a_t,maxdim,m,n)

c     Compute the n-vector w:=a^T (b-ax)
 2    continue
      do l1=1,n
         w(l1)=0.d0
         if(p(l1).eq.0) then
            do l2=1,m
               w(l1) = w(l1) + a_t(l1,l2)*b(l2)
            enddo
         endif
      enddo
c      call get_w(a,a_t,b,x,w,maxdim,m,n)
      do l=1,n
         if(p(l).eq.1) w(l)=0.d0
      enddo

c     If the set Z is empty or if w(j) <= 0 for all j belonging to Z, go
c     to step 12.
 3    continue
      iter = iter + 1
c      if(iop.eq.1) write(0,*)iter
      if(iop.eq.1.and.iter.gt.10*n) then
c         write(0,*),'Exited with problems from NNLS'
         goto 12
      endif
      iz = 0
      do l=1,n
         iz = iz+p(l)
      enddo
      if(iz.eq.n) then
c         write(0,*)'Exiting for empty Z array'
         goto 12
      endif
      ineg = 0
      ip   = 0
      do l=1,n
         if(p(l).eq.0) then
            ip = ip + 1
            if(w(l).le.0.d0) ineg = ineg + 1
         endif
      enddo
      if(ip.eq.ineg) then
c     Only exit if the kuhn-tacker conditions are met.
         if(test_kuhn_tacker(a,a_t,b,x,w,p,maxdim,m,n).eq.1) then
c            write(0,*)'Exiting because all w(j) are negative'
c            write(0,*)'There are ',ip,' elements in Z'
c            write(0,*)(w(l),l=1,n)
            goto 12 
         else
            goto 3
         endif
      endif         

c     Find an index t belonging to Z such that w(t) = max{w(j),j in Z}.
 4    continue
      wmax = -1.d32
      do l=1,n
         if(w(l).gt.wmax.and.p(l).eq.0) then
            wmax = w(l)
            itmax = l
         endif
      enddo
      if(wmax.le.0.d0) goto 12         

c     Move the index t from Z to P.
 5    continue
      p(itmax) = 1
      icome5 = 1

c     Compute the n-vector z that solves a_p*z \approx f. Define z(j) =
c     0 for j belonging to Z.
 6    continue      
      nm1 = 0
      do l1=1,m
         if(p(l1).eq.1) then
            nm1 = nm1 + 1
            bsave(nm1) = b(l1)
            nm2 = 0
            do l2=1,n
               if(p(l2).eq.1) then
                  nm2 = nm2 + 1
                  asave(nm1,nm2) = a(l1,l2)
               else
c                  bsave(nm1) = bsave(nm1) - x(l2)*a(l1,l2)
               endif
            enddo
         endif
      enddo
      call svdcmp(asave,nm1,nm1,maxdim,maxdim,tempw,tempv,its)
      call svbksb(asave,tempw,tempv,nm1,nm1,maxdim,maxdim,bsave,zaux)
      nm1 = 0
      do l=1,n
         if(p(l).eq.1) then
            nm1 = nm1 + 1
            z(l) = zaux(nm1)
         else
            z(l) = 0.d0
         endif
      enddo
      if(icome5.eq.1) then
         if(z(itmax).le.0.d0) then
            w(itmax) = 0.d0
            p(itmax) = 0
            goto 3
         endif
      endif

c     If z(j)>0 for all j belonging to P, set x:=z and go to step 2
 7    continue
      ineg = 0
      do l=1,n
         if(p(l).eq.1.and.z(l).lt.0.d0) then
            ineg = ineg + 1
         endif
      enddo
      if(ineg.eq.0) then
         do l=1,n
c            if(p(l).eq.1) x(l) = z(l)
            x(l) = z(l)
         enddo
         goto 2
      endif


c     Find an index q belonging to P such that x(q)/(x(q)-z(q)) =
c     min{x(j)/(x(j)-z(j)),z(j)<=0,j belongs to P}.
 8    continue
      iter = iter + 1
      if(iter.gt.itermax) then
c         print*,'Maximum number of iterations exceeded in my_nnls'
         MODE = 3
         return
      endif
      xzqmin = 1d32
      do l=1,n
         if(z(l).le.0.d0.and.p(l).eq.1) then
            xzq = x(l)/(x(l)-z(l))
            if(xzq.lt.xzqmin) then
               xzqmin = xzq
               iq = l
            endif
         endif
      enddo

c     Set alpha:=x(q)/(x(q)-z(q))
 9    continue
      alpha = x(iq)/(x(iq)-z(iq))

c     Set x:=x+alpha*(z-x)
 10   continue
      do l=1,n
         if(p(l).eq.1) then
            x(l) = x(l) + alpha*(z(l)-x(l))
            if(x(l).le.0.d0) then
               x(l) = 0.d0
               p(l) = 0
            endif
         endif
      enddo

c     Move from set P to set Z all indices j belonging to P for which
c     x(j)=0. Go to step 6.
 11   continue
      do l=1,n
         if(p(l).eq.1.and.x(l).le.0.d0) p(l) = 0
      enddo
      icome5 = 0
      go to 6


c     Computation is completed
 12   continue      
c      MODE = iter

      return
      end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Function that tests the Kuhn-Tacker conditions have been
c     met. Notice that the vector w is recalculated, such that if the
c     conditions are not met, the new solution is used to pick w_max in
c     step 3.
c
      integer function test_kuhn_tacker(a,a_t,b,x,w,p,maxdim,m,n)
      implicit real*8 (a-h,o-z)

      real*8 a(maxdim,maxdim),a_t(maxdim,maxdim)
      real*8 b(maxdim),x(maxdim)
      integer p(maxdim)

      real*8 w(maxdim)

      call get_w(a,a_t,b,x,w,maxdim,m,n)
      
      ip = 0
      ineg = 0
      do l=1,n
         if(p(l).eq.0) then
            ip = ip + 1
            if(w(l).le.0.d0) ineg = ineg + 1
         endif
      enddo

      if(ineg.eq.ip) then
         test_kuhn_tacker = 1
      else
         test_kuhn_tacker = 0
      endif

      return
      end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Routine to transpose a matrix.
c
      subroutine transpose(a,b,maxdim,m,n)
      implicit real*8 (a-h,o-z)

      real*8 a(maxdim,maxdim),b(maxdim,maxdim)

      do l1=1,n
         do l2=l1,m
            b(l1,l2) = a(l2,l1)
            b(l2,l1) = a(l1,l2)
         enddo
      enddo

      return
      end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Solves the Kuhn-Tacker equation (not really called like that I
c     guess) w = E^T (f - Ex).
c
      subroutine get_w(a,a_t,b,x,w,maxdim,m,n)
      implicit real*8 (a-h,o-z)

      real*8 a(maxdim,maxdim),a_t(maxdim,maxdim)
      real*8 b(maxdim),x(maxdim),w(maxdim),c(maxdim)

      do l1=1,m
         c(l1) = 0.d0
         do l2=1,n
            c(l1) = c(l1) + a(l1,l2)*x(l2)
         enddo
      enddo

      do l=1,m
         c(l) = b(l) - c(l) 
      enddo

      do l1=1,n
         w(l1) = 0.d0
         do l2=1,m
            w(l1) = w(l1) + a_t(l1,l2)*c(l2)
         enddo
      enddo

      return
      end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

c     Numerical Recipes SVD routines.

      SUBROUTINE svdcmp(a,m,n,mp,np,w,v,its)
      implicit real*8 (a-h,o-z)
      INTEGER m,mp,n,np,NMAX
      REAL*8 a(mp,np),v(np,np),w(np)
      PARAMETER (NMAX=500)
CU    USES pythag
      INTEGER i,its,j,jj,k,l,nm
      REAL*8 anorm,c,f,g,h,s,scale,x,y,z,rv1(NMAX),pythag
      g=0.0d0
      scale=0.0d0
      anorm=0.0d0
      do 25 i=1,n
        l=i+1
        rv1(i)=scale*g
        g=0.0
        s=0.0
        scale=0.0d0
        if(i.le.m)then
          do 11 k=i,m
            scale=scale+abs(a(k,i))
11        continue
          if(scale.ne.0.0d0)then
            do 12 k=i,m
              a(k,i)=a(k,i)/scale
              s=s+a(k,i)*a(k,i)
12          continue
            f=a(i,i)
            g=-sign(sqrt(s),f)
            h=f*g-s
            a(i,i)=f-g
            do 15 j=l,n
              s=0.0d0
              do 13 k=i,m
                s=s+a(k,i)*a(k,j)
13            continue
              f=s/h
              do 14 k=i,m
                a(k,j)=a(k,j)+f*a(k,i)
14            continue
15          continue
            do 16 k=i,m
              a(k,i)=scale*a(k,i)
16          continue
          endif
        endif
        w(i)=scale *g
        g=0.0d0
        s=0.0d0
        scale=0.0d0
        if((i.le.m).and.(i.ne.n))then
          do 17 k=l,n
            scale=scale+abs(a(i,k))
17        continue
          if(scale.ne.0.0d0)then
            do 18 k=l,n
              a(i,k)=a(i,k)/scale
              s=s+a(i,k)*a(i,k)
18          continue
            f=a(i,l)
            g=-sign(sqrt(s),f)
            h=f*g-s
            a(i,l)=f-g
            do 19 k=l,n
              rv1(k)=a(i,k)/h
19          continue
            do 23 j=l,m
              s=0.0d0
              do 21 k=l,n
                s=s+a(j,k)*a(i,k)
21            continue
              do 22 k=l,n
                a(j,k)=a(j,k)+s*rv1(k)
22            continue
23          continue
            do 24 k=l,n
              a(i,k)=scale*a(i,k)
24          continue
          endif
        endif
        anorm=max(anorm,(abs(w(i))+abs(rv1(i))))
25    continue
      do 32 i=n,1,-1
        if(i.lt.n)then
          if(g.ne.0.0d0)then
            do 26 j=l,n
              v(j,i)=(a(i,j)/a(i,l))/g
26          continue
            do 29 j=l,n
              s=0.0d0
              do 27 k=l,n
                s=s+a(i,k)*v(k,j)
27            continue
              do 28 k=l,n
                v(k,j)=v(k,j)+s*v(k,i)
28            continue
29          continue
          endif
          do 31 j=l,n
            v(i,j)=0.0d0
            v(j,i)=0.0d0
31        continue
        endif
        v(i,i)=1.0d0
        g=rv1(i)
        l=i
32    continue
      do 39 i=min(m,n),1,-1
        l=i+1
        g=w(i)
        do 33 j=l,n
          a(i,j)=0.0d0
33      continue
        if(g.ne.0.0d0)then
          g=1.0d0/g
          do 36 j=l,n
            s=0.0d0
            do 34 k=l,m
              s=s+a(k,i)*a(k,j)
34          continue
            f=(s/a(i,i))*g
            do 35 k=i,m
              a(k,j)=a(k,j)+f*a(k,i)
35          continue
36        continue
          do 37 j=i,m
            a(j,i)=a(j,i)*g
37        continue
        else
          do 38 j= i,m
            a(j,i)=0.0d0
38        continue
        endif
        a(i,i)=a(i,i)+1.0d0
39    continue
      do 49 k=n,1,-1
        do 48 its=1,60
          do 41 l=k,1,-1
            nm=l-1
            if((abs(rv1(l))+anorm).eq.anorm)  goto 2
            if((abs(w(nm))+anorm).eq.anorm)  goto 1
41        continue
1         c=0.0d0
          s=1.0d0
          do 43 i=l,k
            f=s*rv1(i)
            rv1(i)=c*rv1(i)
            if((abs(f)+anorm).eq.anorm) goto 2
            g=w(i)
            h=pythag(f,g)
            w(i)=h
            h=1.0d0/h
            c= (g*h)
            s=-(f*h)
            do 42 j=1,m
              y=a(j,nm)
              z=a(j,i)
              a(j,nm)=(y*c)+(z*s)
              a(j,i)=-(y*s)+(z*c)
42          continue
43        continue
2         z=w(k)
          if(l.eq.k)then
            if(z.lt.0.0d0)then
              w(k)=-z
              do 44 j=1,n
                v(j,k)=-v(j,k)
44            continue
            endif
            goto 3
          endif
          if(its.eq.60) write(0,*) 'no convergence in svdcmp'
          x=w(l)
          nm=k-1
          y=w(nm)
          g=rv1(nm)
          h=rv1(k)
          f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0d0*h*y)
          g=pythag(f,1.0d0)
          f=((x-z)*(x+z)+h*((y/(f+sign(g,f)))-h))/x
          c=1.0d0
          s=1.0d0
          do 47 j=l,nm
            i=j+1
            g=rv1(i)
            y=w(i)
            h=s*g
            g=c*g
            z=pythag(f,h)
            rv1(j)=z
            c=f/z
            s=h/z
            f= (x*c)+(g*s)
            g=-(x*s)+(g*c)
            h=y*s
            y=y*c
            do 45 jj=1,n
              x=v(jj,j)
              z=v(jj,i)
              v(jj,j)= (x*c)+(z*s)
              v(jj,i)=-(x*s)+(z*c)
45          continue
            z=pythag(f,h)
            w(j)=z
            if(z.ne.0.0d0)then
              z=1.0d0/z
              c=f*z
              s=h*z
            endif
            f= (c*g)+(s*y)
            x=-(s*g)+(c*y)
            do 46 jj=1,m
              y=a(jj,j)
              z=a(jj,i)
              a(jj,j)= (y*c)+(z*s)
              a(jj,i)=-(y*s)+(z*c)
46          continue
47        continue
          rv1(l)=0.0d0
          rv1(k)=f
          w(k)=x
48      continue
3       continue
49    continue
      return
      END

      SUBROUTINE svbksb(u,w,v,m,n,mp,np,b,x)
        implicit real*8 (a-h,o-z)
      INTEGER m,mp,n,np,NMAX
      REAL*8 b(mp),u(mp,np),v(np,np),w(np),x(np)
      PARAMETER (NMAX=500)
      INTEGER i,j,jj
      REAL*8 s,tmp(NMAX)
      do 12 j=1,n
        s=0.d0
        if(w(j).ne.0.d0)then
          do 11 i=1,m
            s=s+u(i,j)*b(i)
11        continue
          s=s/w(j)
        endif
        tmp(j)=s
12    continue
      do 14 j=1,n
        s=0.d0
        do 13 jj=1,n
          s=s+v(j,jj)*tmp(jj)
13      continue
        x(j)=s
14    continue
      return
      END

      FUNCTION pythag(a,b)
      implicit real*8 (a-h,o-z)
      REAL*8 a,b,pythag
      REAL*8 absa,absb
      absa=abs(a)
      absb=abs(b)
      if(absa.gt.absb)then
        pythag=absa*sqrt(1.0d0+(absb/absa)**2)
      else
        if(absb.eq.0.0d0)then
          pythag=0.0d0
        else
          pythag=absb*sqrt(1.0d0+(absa/absb)**2)
        endif
      endif
      return
      END
ccccccc
      SUBROUTINE NNLS(A,MDA,M,N,B,X,RNORM,W,ZZ,INDEX,MODE)
C
C     C.L.LAWSON AND R.J.HANSON,JET PROPULSION LABORATORY, 1973 JUNE 15
C     TO APPEAR IN ^SOLVING LEAST SQUARES PROBLEMS^, PRENTICE HALL,1974
C
C     *****NON-NEGATIVE LEAST SQUARES*****
C
C     GIVEN AN M BY N MATRIX, A, AND AN M VECTOR, B, COMPUTE AN N
C     VECTOR, X, WHICH SOLVES THE LEAST SQUARES PROBLEM
C
C               A*X = B  SUBJECT TO X >= 0, AND
C
C     A(),MDA,M,N   MDA IS THE FIRST DIMENSIONING PARAMETER FOR THE
C                   ARRAY A(). ON ENTRY A() CONTAINS THE M BY N
C                   MATRIX, A. ON EXIT A() CONTAINS THE PRODUCT
C                   MATRIX, Q*A, WHERE Q IS A M BY M ORTHOGONAL
C                   MATRIX GENERATED IMPLICITLY BY THIS SUBROUTINE.
C     B()   ON ENTRY B() CONTAINS THE M-VECTOR, B. ON EXIT B()
C           CONTAINS Q*B.
C     X()   ON ENTRY X() NEED NOT BE INITIALIZED. ON EXIT X() WILL
C           CONTAIN THE SOLUTION VECTOR.
C     RNORM   ON EXIT RNORM CONTAINS THE EUCLIDIAN NORN OF THE RESIDUAL
C             VECTOR
C     W()   AN N-ARRAY OF WORKING SPACE. ON EXIT W() WILL CONTAIN THE
C           DUAL SOLUTION VECTOR. W WILL SATISFY W(I) = 0. FOR ALL
C           I IN THE SET P AND W(I) .LE. 0 FOR ALL I IN THE SET Z
C     ZZ()  AN M-ARRAY OF WORKING SPACE
C     INDEX()   AN INTEGER WORKING ARRAY OF LENGTH AT LEAST N
C               ON EXIT THE CONTENTS OF THIS ARRAY DEFINE THE SETS
C               P AND Z AS FOLLOWS
C
C               INDEX(1) THRU INDEX(NSETP) = SETP
C               INDEX(IZ1) THRU INDEX(IZ2) = SET Z
C               IZ1 = NSETP + 1 = NPP1
C               IZ2 = N
C     MODE   THIS IS A SUCCESS-FAILURE FLAG WITH THE FOLLOWING
C            MEANINGS
C            1   THE SOLUTION HAS BEEN COMPUTED SUCCESSFULLY
C            2   THE DIMENSIONS OF THE PROBLEM ARE BAD.
C                EITHER M.LE.0 OR N.LE.0
C            3   ITERATION COUNT EXCEEDED. MORE THAN 3*N ITERATIONS
C
      IMPLICIT REAL*8(A-H,O-Z)
      IMPLICIT INTEGER(I-N)
C
      DIMENSION A(MDA,N),B(M),X(N),W(N),ZZ(M),INDEX(N)
      ZERO = 0.0
      ONE = 1.0
      TWO = 2.0
      FACTOR = 0.01
C
      MODE = 1
      IF(M.GT.0.AND.N.GT.0) GO TO 10
      MODE = 2
      RETURN
   10 ITER = 0
      ITMAX = 3*N
C
C     INITIALIZE THE ARRAYS INDEX() AND X()
C
      DO 20 I=1,N
      X(I) = ZERO
   20 INDEX(I) = I
C
      IZ2 = N
      IZ1 = 1
      NSETP = 0
      NPP1 = 1
C
C     *****MAIN LOOP BEGINS HERE*****
C
   30 CONTINUE
C
C     QUIT IF ALL COEFFICIENTS ARE ALREADY IN THE SOLUTION OF IF M
C     COLUMNS HAVE BEEN TRIANGULARIZED
C
      IF(IZ1.GT.IZ2.OR.NSETP.GE.M) GO TO 350
C
C     COMPUTE COMPONENTS OF THE DUAL (NEGATIVE GRADIENT) VECTOR W()
C
      DO 50 IZ=IZ1,IZ2
      J = INDEX(IZ)
      SM = ZERO
      DO 40 L=NPP1,M
   40 SM = SM + A(L,J)*B(L)
   50 W(J) = SM
C
C     FIND LARGEST POSITIVE W(J)
C
   60 WMAX = ZERO
      DO 70 IZ=IZ1,IZ2
      J = INDEX(IZ)
      IF(W(J).LE.WMAX) GO TO 70
      WMAX = W(J)
      IZMAX = IZ
   70 CONTINUE
C
C     IF WMAX .LE. 0 GO TO TERMINATION
C     THIS INDICATES SATISFACTION OF THE KUHN-TUCKER CONDITIONS
C
      IF(WMAX) 350,350,80
   80 IZ = IZMAX
      J = INDEX(IZ)
C
C     THE SIGN OF W(J) IS OK FOR J TO BE MOVED TO SET P
C     BEGIN THE TRANSFORMATION AND CHECK NEW DIAGONAL ELEMENT TO AVOID
C     NEAR LINEAR DEPENDENCE
C
      ASAVE = A(NPP1,J)
      CALL H12(1,NPP1,NPP1+1,M,A(1,J),1,UP,DUMMY,1,1,0)
      UNORM = ZERO
      IF(NSETP.EQ.0) GO TO 100
      DO 90 L=1,NSETP
   90 UNORM = UNORM + A(L,J)**2
  100 UNORM = SQRT(UNORM)
      IF(DIFF(UNORM+ABS(A(NPP1,J))*FACTOR,UNORM)) 130,130,110
C
C     SOL J IS SUFFICIENTLY INDEPENDENT. COPY B INTO ZZ, UPDATE ZZ AND
C     SOLVE FOR ZTEST ( = PROPOSED NEW VALUE FOR X(J)).
C
  110 DO 120 L=1,M
  120 ZZ(L) = B(L)
      CALL H12(2,NPP1,NPP1+1,M,A(1,J),1,UP,ZZ,1,1,1)
      ZTEST = ZZ(NPP1)/A(NPP1,J)
C
C     SEE IF ZTEST IS POSITIVE
C
      IF(ZTEST) 130,130,140
C
C     REJECT J AS A CANDIDATE TO BE MOVED FROM SET Z TO SET P
C     RESTORE A(NPP1,J), SET W(J) = 0., AND LOOP BACK TO TEST DUAL
C     COEFFS AGAIN.
C
  130 A(NPP1,J) = ASAVE
      W(J) = ZERO
      GO TO 60
C
C     THE INDEX J = INDEX(IZ) HAS BEEN SELECTED TO BE MOVED FROM THE
C     SET Z TO SET P. UPDATE B. UPDATE INDICES. APPLY HOUSEHOLDER
C     TRANSFORMATIONS TO COLS IN NEW SET Z. ZERO SUBDIAGONAL ELTS IN
C     COL J. SET W(J) =0.
C
  140 DO 150 L=1,M
  150 B(L) = ZZ(L)
C
      INDEX(IZ) = INDEX(IZ1)
      INDEX(IZ1) = J
      IZ1 = IZ1 + 1
      NSETP = NPP1
      NPP1 = NPP1 + 1
C
      IF(IZ1.GT.IZ2) GO TO 170
      DO 160 JZ=IZ1,IZ2
      JJ = INDEX(JZ)
  160 CALL H12(2,NSETP,NPP1,M,A(1,J),1,UP,A(1,JJ),1,MDA,1)
  170 CONTINUE
C
      IF(NSETP.EQ.M) GO TO 190
      DO 180 L=NPP1,M
  180 A(L,J) = ZERO
  190 CONTINUE
C
      W(J) = ZERO
C
C     SOLVE THE TRIANGULAR SYSTEM
C     STORE THE SOLUTION TEMPORARILY IN ZZ()
C
      ASSIGN 200 TO NEXT
      GO TO 400
  200 CONTINUE
C
C     *****SECONDARY LOOP BEGINS HERE*****
C
C     ITERATION COUNTER
C
  210 ITER = ITER + 1
      IF(ITER.LE.ITMAX) GO TO 220
      MODE = 3
      WRITE(6,500)
      GO TO 350
  220 CONTINUE
C
C     SEE IF ALL NEW CONSTRAINED COEFFS ARE FEASIBLE
C     IF NOT COMPUTE ALPHA
C
      ALPHA = TWO
      DO 240 IP=1,NSETP
      L = INDEX(IP)
      IF(ZZ(IP)) 230,230,240
C
  230 T = -X(L)/(ZZ(IP)-X(L))
      IF(ALPHA.LE.T) GO TO 240
      ALPHA = T
      JJ = IP
  240 CONTINUE
C
C     IF ALL NEW CONSTRAINED COEFFS ARE FEASIBLE THEN ALPHA WILL
C     STILL = 2. IF SO EXIT FROM SECONDARY LOOP TO MAIN LOOP
C
      IF(ALPHA.EQ.TWO) GO TO 330
C
C     OTHERWISE USE ALPHA WHICH WILL BE BETWEEN 0. AND 1. TO
C     INTERPOLATE BETWEEN THE OLD X AND THE NEW ZZ.
C
      DO 250 IP=1,NSETP
      L = INDEX(IP)
  250 X(L) = X(L) + ALPHA*(ZZ(IP)-X(L))
C
C     MODIFY A AND B AND THE INDEX ARRAYS TO MOVE COEFFICIENT I
C     FROM SET P TO SET Z
C
      I = INDEX(JJ)
  260 X(I) = ZERO
C
      IF(JJ.EQ.NSETP) GO TO 290
      JJ = JJ + 1
      DO 280 J=JJ,NSETP
      II = INDEX(J)
      INDEX(J-1) = II
      CALL G1(A(J-1,II),A(J,II),CC,SS,A(J-1,II))
      A(J,II) = ZERO
      DO 270 L=1,N
      IF(L.NE.II) CALL G2(CC,SS,A(J-1,L),A(J,L))
  270 CONTINUE
  280 CALL G2(CC,SS,B(J-1),B(J))
  290 NPP1 = NSETP
      NSETP = NSETP - 1
      IZ1 = IZ1 - 1
      INDEX(IZ1) = I
C
C     SEE IF THE REMAINING COEFFS IN SET P ARE FEASIBLE. THEY SHOULD
C     BE BECAUSE OF THE WAY ALPHA WAS DETERMINED.
C     IF ANY ARE INFEASIBLE IT IS DUE TO ROUND-OFF ERROR. ANY
C     THAT ARE NON-POSITIVE WILL BE SET TO ZERO
C     AND MOVED FROM SET P TO SET Z
C
      DO 300 JJ=1,NSETP
      I = INDEX(JJ)
      IF(X(I)) 260,260,300
  300 CONTINUE
C
C     COPY B() INTO ZZ(). THEN SOLVE AGAIN AND LOOP BACK
C
      DO 310 I=1,M
  310 ZZ(I) = B(I)
      ASSIGN 320 TO NEXT
      GO TO 400
  320 CONTINUE
      GO TO 210
C
C     *****END OF SECONDARY LOOP*****
C
  330 DO 340 IP=1,NSETP
      I = INDEX(IP)
  340 X(I) = ZZ(IP)
C
C     ALL NEW COEFFICIENTS ARE POSITIVE. LOOP BACK TO BEGINNING.
C
      GO TO 30
C
C     *****END OF MAIN LOOP*****
C
C     COME TO HERE FOR TERMINATION
C     COMPUTE THE NORM OF THE FINAL RESIDUAL VECTOR
C
  350 SM = ZERO
      IF(NPP1.GT.M) GO TO 370
      DO 360 I=NPP1,M
  360 SM = SM + B(I)**2
      GO TO 390
  370 DO 380 J=1,N
  380 W(J) = ZERO
  390 RNORM = SQRT(SM)
      RETURN
C
C
C     THE FOLLOWING BLOCK OF CODE IS USED AS AN INTERNAL SUBROUTINE
C     TO SOLVE THE TRIANGULAR SYSTEM. PUTTING THE SOLUTION IN ZZ()
C
  400 DO 430 L=1,NSETP
      IP = NSETP + 1 - L
      IF(L.EQ.1) GO TO 420
      DO 410 II=1,IP
  410 ZZ(II) = ZZ(II) - A(II,JJ)*ZZ(IP+1)
  420 JJ = INDEX(IP)
  430 ZZ(IP) = ZZ(IP)/A(IP,JJ)
      GO TO NEXT,(200,320)
  500 FORMAT(35H0 NNLS QUITTING ON ITERATION COUNT )
      END

      FUNCTION DIFF(X,Y)
C
C     C.L.LAWSON AND R.J.HANSON, JET PROPULSION LABORATORY, 1973 JUNE 7
C     TO APPEAR IN ^SOLVING LEAST SQUARES PROBLEMS^, PRENTICE-HALL, 1974
C
      IMPLICIT REAL*8(A-H,O-Z)
      IMPLICIT INTEGER(I-N)
C
      DIFF = X - Y
      RETURN
      END

      SUBROUTINE G1(A,B,COS,SIN,SIG)
C
C     C.L.LAWSON AND R.J.HANSON, JET PROPULSION LABORATORY, 1973 JUN 12
C     TO APPEAR IN ^SOLVING LEAST SQUARES PROBLEMS^, PRENTICE HALL, 1974
C
C     COMPUTE ORTHOGONAL ROTAION MATRIX
C     COMPUTE   MATRIX  (C,S) SO THAT (C,S)(A) = (SQRT(A**2+B**2)
C                      (-S,C) SO THAT (-S,C)(B) = (0)
C     COMPUTE SIG = SQRT(A**2+B**2)
C             SIG IS COMPUTED TO ALLOW FOR THE POSSIBILITY THAT
C             SIG MAY BE IN THE SAME LOCATION AS A OR B
C
      IMPLICIT REAL*8(A-H,O-Z)
      IMPLICIT INTEGER(I-N)
C
      ZERO = 0.0
      ONE = 1.0
      IF(ABS(A).LE.ABS(B)) GO TO 10
      XR = B/A
      YR = SQRT(ONE+XR**2)
      COS = SIGN(ONE/YR,A)
      SIN = COS*XR
      SIG = ABS(A)*YR
      RETURN
   10 IF(B) 20,30,20
   20 XR = A/B
      YR = SQRT(ONE+XR**2)
      SIN = SIGN(ONE/YR,B)
      COS = SIN*XR
      SIG = ABS(B)*YR
      RETURN
   30 SIG = ZERO
      COS = ZERO
      SIN = ONE
      RETURN
      END

      SUBROUTINE H12(MODE,LPIVOT,L1,M,U,IUE,UP,C,ICE,ICV,NCV)
C
C     C.L.LAWSON AND R.J.HANSON, JET PROPULSION LABORATORY, 1973 JUNE 12
C     TO APPEAR IN ^SOLVING LEAST SQUARES PROBLEMS^, PRENTICE-HALL, 1974
C
C     CONSTRUCTION AND/OR APPLICATION OF A SINGLE
C     HOUSEHOLDER TRANSFORMATION     Q = I + U*(U**T)/B
C
C     MODE  = 1 OR 2  TO SELECT ALGORITHM H1 OR H2
C     LPIVOT IS THE INDEX OF THE PIVOT ELEMENT
C     L1,M  IF L1.LE.M THE TRANSFORMATION WILL BE CONSTRUCTED TO
C           ZERO ELEMENTS INDEXED FROM L1 THROUGH M. IF L1.GT.M
C           THE SUBROUTINE DOES AN IDENTITY TRANSFORMATION
C     U(),IUE,UP  ON ENTRY TO H1 U() CONTAINS THE PIVOT VECTOR
C                 IUE IS THE STORAGE INCREMENT BETWEEN ELEMENTS.
C                 ON EXIT FROM H1 U() AND UP CONTAIN QUANTITIES
C                 DEFINING THE VECTOR U OF THE HOUSEHOLDER
C                 TRANSFORMATION. ON ENTRY TO H2 U() AND UP SHOULD
C                 CONTAIN QUANTITIES PREVIOUSLY COMPUTED BY H1.
C                 THESE WILL NOT BE MODIFIED BY H2.
C     C()  ON ENTRY TO H1 OR H2 C() CONTAINS A MATRIX WHICH WILL BE
C          REGARDED AS A SET OF VECTORS TO WHICH THE HOUSEHOLDER
C          TRANSFORMATION IS TO BE APPLIED. ON EXIT C() CONTAINS THE
C          SET OF TRANSFORMED VECTORS.
C     ICE  STORAGE INCREMENT BETWEEN ELEMENTS IN C ()
C     ICV  STORAGE INCREMENT BETWEEN VECTORS IN C ()
C     NCV  NUMBER OF VECTORS IN C() TO BE TRANSFORMED. IF NCV.LE.0
C          NO OPERATIONS WILL BE DONE ON C()
C
      IMPLICIT REAL*8(A-H,O-Z)
      IMPLICIT INTEGER(I-N)
C
      DIMENSION U(IUE,M),C(1)
      ONE = 1.
C
      IF(0.GE.LPIVOT.OR.LPIVOT.GE.L1.OR.L1.GT.M) RETURN
      CL = ABS(U(1,LPIVOT))
      IF(MODE.EQ.2) GO TO 60
C
C     CONSTRUCT THE TRANSFORMATION
C
      DO 10 J=L1,M
   10 CL = MAX(ABS(U(1,J)),CL)
      IF(CL) 130,130,20
   20 CONTINUE
C
C     ****** THIS IS A VAX TEST FOR D_FLOATING REAL*8 ARITHMETIC
C            IT PREVENTS AN OVERFLOW ON INVERSION
C
      IF(CL.LT.5.9D-39) GO TO 130
C
      CLINV = ONE/CL
      SM = (U(1,LPIVOT)*CLINV)**2
      DO 30 J=L1,M
   30 SM = SM + (U(1,J)*CLINV)**2
C
C     CONVERT DBLE PREC. SM TO SNGL PREC. SM1
C
      SM1 = SM
      CL = CL*SQRT(SM1)
      IF(U(1,LPIVOT)) 50,50,40
   40 CL = -CL
   50 UP = U(1,LPIVOT) - CL
      U(1,LPIVOT) = CL
      GO TO 70
C
C     APPLY THE TRANSFORMATION 1+U*U(T)/B TO C
C
   60 IF(CL) 130,130,70
   70 IF(NCV.LE.0) RETURN
      B = UP*U(1,LPIVOT)
C
C     B MUST BE NONPOSITIVE HERE. IF B=0 RETURN
C
      IF(B) 80,130,130
   80 CONTINUE
C
C     ****** THIS IS A VAX TEST FOR D_FLOATING REAL*8 ARITHMETIC
C            IT PREVENTS AN OVERFLOW ON INVERSION
C
      IF(-B.LT.5.9D-39) GO TO 130
C
      B = ONE/B
      I2 = 1 - ICV+ICE*(LPIVOT-1)
      INCR = ICE*(L1-LPIVOT)
      DO 120 J=1,NCV
      I2 = I2 + ICV
      I3 = I2 + INCR
      I4 = I3
      SM = C(I2)*UP
      DO 90 I=L1,M
      SM = SM + C(I3)*U(1,I)
   90 I3 = I3 + ICE
      IF(SM) 100,120,100
  100 SM = SM*B
      C(I2) = C(I2) + SM*UP
      DO 110 I=L1,M
      C(I4) = C(I4) + SM*U(1,I)
  110 I4 = I4 + ICE
  120 CONTINUE
  130 RETURN
      END
      SUBROUTINE G2(COS,SIN,X,Y)
C
C     C.L.LAWSON AND R.J.HANSON, JET PROPULSION  LABORATORY, 1972 DEC 15
C     TO APPEAR IN ^SOLVING LEAST SQUARES PROBLEMS^, PRENTICE-HALL, 1974
C               APPLY THE ROTATION COMPUTED BY G1 TO (X,Y)
C
      IMPLICIT REAL*8(A-H,O-Z)
      IMPLICIT INTEGER(I-N)
C
      XR = COS*X + SIN*Y
      Y = -SIN*X + COS*Y
      X = XR
      RETURN
      END

ccccccccccccccc

      subroutine ANNLS(a,maxdim,m,n,b,x)
      implicit real*8 (a-h,o-z)
      parameter (NCMAX=32,NSMAX=5)

      real*8 a(maxdim,maxdim)
      real*8 b(maxdim),x(maxdim)

      real*8 asave(maxdim,maxdim)
      real*8 bsave(maxdim)

      real*8 temps(maxdim)
      real*8 tempw(maxdim),tempv(maxdim,maxdim)

      integer ivaryobj(NSMAX)

      real*8 jy(NCMAX),ejy(NCMAX)
      integer nchan
      common /data1b/jy,ejy,nchan

      integer jyuse(NCMAX)
      common /data2/jyuse

c      real*8 vec(NSMAX)
c      real*8 jymod(NSMAX,NCMAX)
c      real*8 jymodtot(NCMAX)
c      common /models/jymod,jymodtot,vec

      real*8 jymodx(NSMAX,NCMAX)
      common /modelsx/jymodx
      real*8 jymodtot(NCMAX)

!$omp threadprivate(/data1b/,/data2/)      

c     Save the inputs
      do l1=1,n
         bsave(l1) = b(l1)
         do l2=1,m
            asave(l1,l2) = a(l1,l2)
         enddo
      enddo

c     Try to solve the system without constraints
      call svdcmp(a,m,n,maxdim,maxdim,tempw,tempv,its)
      call svbksb(a,tempw,tempv,m,n,maxdim,maxdim,b,x)     

c     If none of the components is negative, exit.
      ineg = 0
      do l=1,m
         if(x(l).lt.0.d0) ineg = ineg + 1
      enddo
      if(ineg.eq.0) goto 650
      
c     If not the case, try all possible combinations.
      imax    = 2**n
      count   = 0
      chi2min = 1.d32
      do i=1,imax-1
         call get_flags(ivaryobj,n,i)
         nm1 = 0
         do l1=1,n
            if(ivaryobj(l1).eq.1) then
               nm1    = nm1 + 1
               nm2    = 0
               b(nm1) = bsave(l1)
               do l2=1,m
                  if(ivaryobj(l2).eq.1) then
                     nm2 = nm2+1
                     a(nm1,nm2) = asave(l1,l2)
                  endif
               enddo
            endif
         enddo

         call svdcmp(a,nm1,nm1,maxdim,maxdim,tempw,tempv,its)
         call svbksb(a,tempw,tempv,nm1,nm1,maxdim,maxdim,b,temps)     

         do l=1,nm1
            if(temps(l).lt.0.d0) temps(l) = 0.d0
c            print*,'temps(',l,') = ',temps(l)
         enddo         
         
         chi2 = 0.d0
         do j=1,nchan
            jymodtot(j) = 0.d0
            nm1 = 0
            do l=1,n
               if(ivaryobj(l).eq.1) then
                  nm1 = nm1 + 1
                  jymodtot(j) = jymodtot(j) + temps(nm1)*jymodx(l,j)
               endif               
            enddo
            if(jyuse(j).ge.1) then
               diff = jy(j)-jymodtot(j)
               chi2 = chi2 + diff*diff/ejy(j)
            endif
c            print*,jymodtot(j),jymodx(1,j)
         enddo
c         print*,chi2
c         pause

         if(chi2.lt.chi2min) then
            chi2min = chi2
            nm1 = 0
            do l=1,n
               if(ivaryobj(l).ge.1) then
                  nm1 = nm1+1
                  x(l) = temps(nm1)
               else
                  x(l) = 0.d0
               endif
            enddo
         endif
               

      enddo
       

 650  continue

      return
      end

cccccc

      subroutine get_flags(vec,n,iflag)
      implicit real*8 (a-h,o-z)

      integer vec(n)
      integer flag

      flag = iflag

      do i=n-1,0,-1
         if(flag.ge.2**i) then
            flag    = flag - 2**i
            vec(i+1) = 1
         else
            vec(i+1) = 0
         endif
      enddo

      return
      end
     
