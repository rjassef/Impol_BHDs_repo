#Compiler
FC = gfortran
CC = icc

#Compiler Flags
FCFLAGS = -ffixed-line-length-none -std=legacy #-fbounds-check
#Remover tool
RM = /bin/rm -f

#Open MP flag
OMP = #-fopenmp

#PZA or KCA Location
LIBPATH = /Users/rassef/OSU/AGN_IRAC/GR45_All/codes

#Libraries
PZALIB  = $(LIBPATH)/pza_agn_logred.f
#NNLSLIB = $(LIBPATH)/my_nnls_2.f
NNLSLIB = my_nnls_2.f
#KCALIB  = $(LIBPATH)/kca_agn_logred_galprior.f
#KCALIB  = $(LIBPATH)/kca_agn_logred.f
KCALIB  = kca_agn_logred.f

#top-level
single: single.f $(KCALIB) $(NNLSLIB)
	$(FC) $(FCFLAGS) -o single single.f $(KCALIB) $(NNLSLIB)

double: double.f $(KCALIB) $(NNLSLIB)
	$(FC) $(FCFLAGS) -o double double.f $(KCALIB) $(NNLSLIB)

clean: 
	$(RM) single double
