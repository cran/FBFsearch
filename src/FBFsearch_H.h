#ifndef FBFSEARCH_H
#define FBFSEARCH_H

#pragma once

#include <RcppArmadillo.h>

// Alias coerente con il .cpp
using uli = unsigned long int;
using namespace arma;

// --- Prototipi funzioni di utilità ---
double log_sum(colvec v);
double lfactorial(uli n);
double lchoose(uli n, uli k);

// --- Prototipi coerenti con le definizioni del .cpp ---
mat sub_mat(const mat& M, const vec& vr, const vec& vc);
vec pow_vec(const vec& v, const vec& w);

field<mat> add_to_tree(vec M, double lM, uli nM, mat tree, double ltree);
vec mov_tree(const mat& tree, const vec& M, uli lM, const vec& vlM, uli max_lM);

double log_H_h_i(double mu, double sigma, uli h, uli i);
double log_FBF_Ga_Gb(vec G_a, vec G_b, uli edge, mat edges, mat YtY, uli add, double n, double h);
field<mat> FBF_heart(double nt, mat YtY, vec vG_base, double lcv, vec vlcv, mat edges,
                     double n_tot_mod, double C, double maxne, double h);

mat G_fin_fill(mat G, const vec& vr, uli ic, const vec& x);

// --- Interfaccia esportata verso R ---
RcppExport SEXP FBF_LS(SEXP Corr, SEXP nobs, SEXP G_base, SEXP h, SEXP C, SEXP n_tot_mod);
RcppExport SEXP FBF_RS(SEXP Corr, SEXP nobs, SEXP G_base, SEXP h, SEXP C, SEXP n_tot_mod, SEXP n_hpp);
RcppExport SEXP FBF_GS(SEXP Corr, SEXP nobs, SEXP G_base, SEXP h, SEXP C, SEXP n_tot_mod, SEXP n_hpp);

#endif // FBFSEARCH_H
