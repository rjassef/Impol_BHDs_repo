      program main
      implicit real*8(a-h,o-z)
      parameter(NCMAX=32,NSMAX=5,NWMAX=350)

      real*8 err_comp(NSMAX)
      common /comperr/err_comp

      real*8 jy(21),ejy(21)
      integer jyuse(21),jyusesave(21)

      real*8 mag(21),emag(21)
      integer maguse(21)

      real*8 jymod(21),jycorr(21)
      real*8 comp(5),ebv1,ebv2,igm,ecomp(5)
      real*8 vec(5),evec(5)

      real*8 magabs(21),hcomp(5)

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

      real*8 bcon(5)
      data bcon/1.4d-10,0.9d-10,1.2d-10,1.8d-10,7.4d-10/

      real*8 xlam(2),xnu(2),xLnu(2),dLnu(2)

      z0 = 2.0d0
      nspec = 5
      nchan = 16


      call setfilt('bandmag.dat')
      call settemp(4)
      call set_red(1)
      call setdist

      open(unit=11,file='phot.dat',
     *     status='old')
c     '

      open(unit=20,file='single.20',status='unknown')
      open(unit=21,file='single.21',status='unknown')
      open(unit=22,file='single.22',status='unknown')
      
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
c     WISE, Spitzer, and NIR
            if(j.le.11) then
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
               ib = j-11
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
         if(jyuse(5).eq.1) jyuse(1) = 0
         if(jyuse(6).eq.1) jyuse(2) = 0
         
c     Get the number of usable bands.
         m = 0
         do j=1,nchan
            if(jyuse(j).gt.0) m = m + 1
         enddo
         write(0,*)name,m

c     Run the fits.
         call kca(jy,ejy,jyuse,nchan,z,z0,jymod,jycorr,comp,0,
     *        ebv1,ebv2,igm,chi2)


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
         
         write(22,122)i,ebv1,ebv2,igm,name
 122     format(i10,3ES20.6,' ',a10)

         goto 100
 101  continue

      close(11)
      close(12)
      close(13)

      stop
      end
