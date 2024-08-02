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
      real*8 comp(5),ebv1x,ebv2x,igmx,ecomp(5)
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

      real*8 tau(NWMAX),ebv1,ebv2,igm
      common /dust/tau,ebv1,ebv2,igm

      real*8 bcon(5)
      data bcon/1.4d-10,0.9d-10,1.2d-10,1.8d-10,7.4d-10/

      real*8 xlam(5),xnu(5),xLnu(5),dLnu(5)
      data xlam/6.d0,6.d0,6.d0,6.d0,6.d0/

      real*8 L6um_high, L6um_low

      z0 = 2.0d0
      nspec = 5
      nchan = 16


      call setfilt('bandmag.dat')
      call settemp(5)
      call set_red(1)
      call setdist

      open(unit=11,file='BHDs.ran.phot',status='old')
      open(unit=12,file='kc_components.dat',status='unknown')

      open(unit=20,file='double.20',status='unknown')
      open(unit=21,file='double.21',status='unknown')
      open(unit=22,file='double.22',status='unknown')
      
      itarg = 0
 100  read(11,*,end=101)id,z,(jy(j),j=1,nchan),(ejy(j),j=1,nchan),
     *     (jyuse(j),j=1,nchan)

         itarg = itarg + 1

c     Get the number of usable bands.
         m = 0
         do j=1,nchan
            if(jyuse(j).gt.0) m = m + 1
         enddo

c     Run the fits.
         call kca(jy,ejy,jyuse,nchan,z,z0,jymod,jycorr,comp,0,
     *        ebv1x,ebv2x,igmx,chi2)

         vecfac = DL(z)**2*1d10*3d-9/(1+z)
         do l=1,nspec
            vec(l) = comp(l)*alpha(l)/vecfac
            evec(l) = err_comp(l)*alpha(l)/vecfac
         enddo
         write(20,120)id,z,chi2,(vec(l),l=1,nspec)
 120     format(i10,7ES20.6)
         
         do j=1,nchan
            write(21,121)id,jy(j),jymod(j),ejy(j),jyuse(j),
     *           lbar(j)/(1.d0+z)
 121        format(i10,3ES20.6,i10,ES20.6)
         enddo
         
         write(22,122)id,ebv1x,ebv2x,igmx,name
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
            xnu(ll) = 3d14/xlam(ll)
            flux = flux*1.d-23
            xLnu(ll) = 4.d0*pi*(DL(z)*3.086d24)**2 * flux/(1.d0+z)
         enddo
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

         ii = itarg/100
         if(itarg.eq.ii*100) print*,itarg

         write(12,112)id,ebv_low,ebv_high,L6um_low,L6um_high
 112     format(i10,4ES20.6)

         goto 100
 101  continue

      close(11)
      close(12)
      close(13)

      stop
      end
