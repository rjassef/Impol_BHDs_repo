      program main
      implicit real*8(a-h,o-z)
      parameter(NCMAX=32,NSMAX=5,NWMAX=350)

      real*8 err_comp(NSMAX)
      common /comperr/err_comp

      real*8 jy(28),ejy(28)
      integer jyuse(28),jyusesave(28)

      real*8 mag(28),emag(28)
      integer maguse(28)

      real*8 jymod(28),jycorr(28)
      real*8 comp(5),ebv1x,ebv2x,igmx,ecomp(5)
      real*8 vec(5),evec(5)

      real*8 magabs(28),hcomp(5)

      real*8 alpha(NSMAX)
      common /alphanorm/alpha

      real*8 jyzero(NCMAX),sat(NCMAX),con(NCMAX),lbar(NCMAX)
      common /cal1/jyzero,sat,con,lbar

      real*8 spec(NSMAX,NWMAX),specuse(NSMAX,NWMAX)
      common /specmod1/spec,specuse,nspec
      common /specnorm/bminnorm,bmaxnorm

      real*8 bedge(NWMAX)
      real*8 bcen(NWMAX)
      common /wavegrid/bedge,bcen,nwave

      character*20 name

      real*8 tau(NWMAX),ebv1,ebv2,igm
      common /dust/tau,ebv1,ebv2,igm

      real*8 bcon(5)
      data bcon/1.4d-10,0.9d-10,1.2d-10,1.8d-10,7.4d-10/

      real*8 xlam(5),xnu(5),xLnu(5),dLnu(5)
      data xlam/6.d0,6.d0,6.d0,6.d0,6.d0/

      real*8 L6um_high, L6um_low

      z0 = 2.0d0
      nspec = 5
      open(unit=32, file='nchan.txt', status='old')
      read(32,*)nchan
      close(32)
c      nchan = 28


      call setfilt('bandmag.dat')
      call settemp(5)
      call set_red(1)
      call setdist

      open(unit=11,file='phot.dat',
     *     status='old')
c     '
      open(unit=12,file='kc_components.double.dat',status='unknown')

      open(unit=20,file='double.20',status='unknown')
      open(unit=21,file='double.21',status='unknown')
      open(unit=22,file='double.22',status='unknown')
      
      itarg = 0
 100  read(11,*,end=101)name,z,(mag(j),emag(j),j=1,nchan)

         itarg = itarg+1
         id = itarg
         i = itarg

c     Wright et al. suggests that W3 is too faint by 17% and W4 is too
c     bright by 9%. Lets make these corrections.
         mag(3) = mag(3) - 0.17d0
         mag(4) = mag(4) + 0.09d0

c     Convert mags to fluxes.
         do j=1,nchan
            if(j.le.(nchan-5)) then
               if(mag(j).lt.0.d0.and.emag(j).lt.0.d0) then
                  jyuse(j) = 0
                  jy(j)    = 0.d0
                  ejy(j)   = 0.d0
               else if(mag(j).lt.90.d0.and.emag(j).lt.90.d0) then
                  jyuse(j) = 1
                  jy(j)    = jyzero(j)*10.d0**(-0.4d0*mag(j))
                  ejy(j)   = 0.4d0*dlog(10.d0)*emag(j)*jy(j)
               else if(mag(j).lt.90.d0.and.emag(j).gt.90.d0) then
                  jyuse(j) = 2
                  jy(j)    = 0.d0
                  ejy(j)   = jyzero(j)*10.d0**(-0.4d0*mag(j))
c     WISE upper bounds are 2sigma, so match the others to be 2sigma.
                  if(j.gt.4) ejy(j) = ejy(j)*2.d0
               else
                  jyuse(j) = 0
                  jy(j)    = 0.d0
                  ejy(j)   = 0.d0
               endif
c     SDSS asinh mags
            else
               ib = j-(nchan-5)
               if(mag(j).gt.0.d0.and.mag(j).lt.90.d0) then
                  asinval = -0.4d0*dlog(10.d0)*mag(j)-dlog(bcon(ib))
                  jy(j)   = jyzero(j)*bcon(ib)*(dexp(asinval)-dexp(-asinval))
                  ejy(j)  = bcon(ib)*(dexp(asinval)+dexp(-asinval))
                  ejy(j)  = 0.4d0*dlog(10.d0)*jyzero(j)*ejy(j)*emag(j)
                  if(jy(j).gt.0.d0) then
                     jyuse(j) = 1
                  else
                     jyuse(j) = 2
                     jy(j)    = 0.d0
                     ejy(j)   = 3.d0*bcon(ib)*3631.d0
                  endif
               else
                  jyuse(j) = 0
                  jy(j)    = 0.d0
                  ejy(j)   = 0.d0
               endif
            endif
         enddo


c     Set an error floor of 5%
         do j=1,nchan
            if(ejy(j).lt.5d-2*jy(j)) ejy(j) = 5.d-2*jy(j)
         enddo

c     Do not use W1 or W2 if IRAC is available.
c         if(jyuse(5).eq.1) jyuse(1) = 0
c         if(jyuse(6).eq.1) jyuse(2) = 0

c     Get the number of usable bands.
         m = 0
         do j=1,nchan
            if(jyuse(j).gt.0) m = m + 1
         enddo
         write(0,*)name,m

c     Run the fits.
         call kca(jy,ejy,jyuse,nchan,z,z0,jymod,jycorr,comp,0,
     *        ebv1x,ebv2x,igmx,chi2)

         vecfac = DL(z)**2*1d10*3d-9/(1+z)
         do l=1,nspec
            vec(l) = comp(l)*alpha(l)/vecfac
            evec(l) = err_comp(l)*alpha(l)/vecfac
         enddo
         write(20,120)i,z,chi2,(vec(l),l=1,nspec)
 120     format(i10,7ES20.6)
         
         do j=1,nchan
            write(21,121)i,jy(j),jymod(j),ejy(j),jyuse(j),
     *           lbar(j)/(1.d0+z)
 121        format(i10,3ES20.6,i10,ES20.6)
         enddo
         
         write(22,122)i,ebv1x,ebv2x,igmx,name
 122     format(i10,3ES20.6,' ',a10)


c     Now get the 6um continuum luminosities.
         pi = 4.d0*datan(1.d0)
         do ll=1,nspec
            flux = 0.d0
            dflux = 0.d0
            ksave = -1
            do k=1,nwave
               if(bcen(k).ge.xlam(ll)) then
                  ksave = k
                  goto 70
               endif
            enddo
 70         continue
            suse = (spec(ll,ksave)-spec(ll,ksave-1))/(bcen(ksave)-bcen(ksave-1))
     *           * (xlam(ll)-bcen(ksave)) + spec(ll,ksave)
            flux  = suse*vec(ll)
            dflux = suse*evec(ll)

            xnu(ll) = 3d14/xlam(ll)

            flux = flux*1.d-23
            xLnu(ll) = 4.d0*pi*(DL(z)*3.086d24)**2 * flux/(1.d0+z)

            dflux = dflux*1.d-23
            dLnu(ll) = 4.d0*pi*(DL(z)*3.086d24)**2 * dflux/(1.d0+z)

         enddo
c         print*
c         print*,ebv1x,xLnu(1)*3d14/xlam(1),xLnu(1)*3d14/xlam(1)/3.839e33/1e14
c         print*,ebv2x,xLnu(5)*3d14/xlam(5),xLnu(5)*3d14/xlam(5)/3.839e33/1e14
         if(xLnu(1).gt.xLnu(5)) then
            L6um_high = xLnu(1)*3d14/xlam(1)/3.839e33/1e14
            L6um_low  = xLnu(5)*3d14/xlam(5)/3.839e33/1e14
            ebv_high  = ebv1x
            ebv_low   = ebv2x
         else
            L6um_low  = xLnu(1)*3d14/xlam(1)/3.839e33/1e14
            L6um_high = xLnu(5)*3d14/xlam(5)/3.839e33/1e14
            ebv_low   = ebv1x
            ebv_high  = ebv2x
         endif
         write(12,112)id,ebv_low,ebv_high,L6um_low,L6um_high
 112     format(i10,4ES20.6)


c     Now get the UV continuum luminosities.
c         uvlam = 0.5d0*(1500.d0+2800.d0)*1.d-4
c         ksave = -1
c         do k=1,nwave
c            if(bcen(k).ge.uvlam) then
c               ksave = k
c               goto 80
c            endif
c         enddo
c 80      continue
c         tauuse = (tau(ksave)-tau(ksave-1))/(bcen(ksave)-bcen(ksave-1))
c     *        * (uvlam-bcen(ksave)) + tau(ksave)
c         flux = 0.d0
c         dflux = 0.d0
c         do l=1,nspec
c            suse = (spec(l,ksave)-spec(l,ksave-1))/(bcen(ksave)-bcen(ksave-1))
c     *           * (uvlam-bcen(ksave)) + spec(l,ksave)
c            dust = 1.d0
c            if(l.eq.1) dust = 10.d0**(-0.4d0*tauuse*ebv1x)
c            if(l.eq.5) dust = 10.d0**(-0.4d0*tauuse*ebv2x)
c            suse = suse*dust
c            flux  = flux + suse*vec(l)
c            dflux = dflux + (suse*evec(l))**2
c         enddo
c         dflux = dsqrt(dflux)
c         uvnu = 3d14/uvlam
c         print*,flux,jy(nchan-3)
c         flux = flux*1.d-23
c         uvLnu = 4.d0*pi*(DL(z)*3.086d24)**2 * flux/(1.d0+z)
c
c         dflux = dflux*1.d-23
c         duvLnu = 4.d0*pi*(DL(z)*3.086d24)**2 * dflux/(1.d0+z)
c
c         print*
c         print*,"UV Lum =",uvLnu,duvLnu
c         print*,"UV SFR =",1.4e-28*uvLnu,1.4e-28*duvLnu

c     Now get the 4400A flux density
c         optlam = 0.44d0
c         ksave = -1
c         do k=1,nwave
c            if(bcen(k).ge.optlam) then
c               ksave = k
c               goto 90
c            endif
c         enddo
c 90      continue
c         tauuse = (tau(ksave)-tau(ksave-1))/(bcen(ksave)-bcen(ksave-1))
c     *        * (optlam-bcen(ksave)) + tau(ksave)
c         flux = 0.d0
c         dflux = 0.d0
c         do l=1,nspec
c            suse = (spec(l,ksave)-spec(l,ksave-1))/(bcen(ksave)-bcen(ksave-1))
c     *           * (optlam-bcen(ksave)) + spec(l,ksave)
c            dust = 1.d0
c            if(l.eq.1) dust = 10.d0**(-0.4d0*tauuse*ebv1x)
c            if(l.eq.5) dust = 10.d0**(-0.4d0*tauuse*ebv2x)
c            suse = suse*dust
c            flux  = flux + suse*vec(l)
c            dflux = dflux + (suse*evec(l))**2
c         enddo
c         dflux = dsqrt(dflux)
c         print*,'At 4400A: ',flux,dflux
c         print*,'J-band:',jy(7)

         goto 100
 101  continue

      close(11)
      close(12)
      close(13)

      stop
      end
