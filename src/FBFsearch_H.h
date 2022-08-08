#ifndef FBFsearch_H
#define FBFsearch_H

#define uli unsigned long int

#include <RcppArmadillo.h>

using namespace arma;

double log_sum(colvec v);
double lfactorial(uli n);
double lchoose(uli n, uli k);

mat sub_mat(mat M, vec vr, vec vc);
mat pow_vec(vec v, vec w);

field<mat> add_to_tree(vec M, double lM, uli nM, mat tree, double ltree);
vec mov_tree(mat tree, vec M, uli lM, vec vlM, uli max_lM);

double log_H_h_i(double mu, double sigma, uli h, uli i);
double log_FBF_Ga_Gb(vec G_a, vec G_b, uli edge, mat edges, mat YtY, uli add, double n, double h);
field<mat> FBF_heart(double nt, mat YtY, vec vG_base, double lcv, vec vlcv, mat edges, double n_tot_mod, double C, double maxne, double h);
mat G_fin_fill(mat G, vec vr, uli ic, vec x);


RcppExport SEXP FBF_LS(SEXP Corr_r, SEXP nobs_r, SEXP G_base_r, SEXP h_r, SEXP C_r, SEXP n_tot_mod_r);
RcppExport SEXP FBF_RS(SEXP Corr_r, SEXP nobs_r, SEXP G_base_r, SEXP h_r, SEXP C_r, SEXP n_tot_mod_r, SEXP n_hpp_r);
RcppExport SEXP FBF_GS(SEXP Corr_r, SEXP nobs_r, SEXP G_base_r, SEXP h_r, SEXP C_r, SEXP n_tot_mod_r, SEXP n_hpp_r);

#endif
