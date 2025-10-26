#include <iostream>
#include <string>
#include <cmath>                // std::isfinite, std::log1p, std::exp
#include <RcppArmadillo.h>
#include "FBFsearch_H.h"

#define uli unsigned long int

using namespace std;
using namespace arma;

// -----------------------------------------------------------------------------
// Helper numerico per log-sum-exp (sostituisce la vecchia log_add deprecata)
// -----------------------------------------------------------------------------
inline double log_add(double a, double b) {
  if (std::isinf(a)) return b;
  if (std::isinf(b)) return a;
  if (a < b) std::swap(a, b);
  return a + std::log1p(std::exp(b - a));
}

//---------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------------------------------
// FUNZIONI MATEMATICHE GENERALI
//------------------------------------------------------------------------------------------------------------------------

// LOGARITMO FATTORIALE
double lfactorial(uli n) {
  return lgamma(n + 1);
}

// LOGARITMO COEFFICIENTE BINOMIALE
double lchoose(uli n, uli k) {
  return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
}

// LOGARITMO DELLA SOMMA (stabile)
double log_sum(colvec v) {
  double lsum;
  if (accu(v >= 0) > 0) {
    if (v.max() <= 700) lsum = std::log(accu(exp(v)));
    else lsum = v.max();
  } else {
    if (v.min() >= -700) lsum = std::log(accu(exp(v)));
    else lsum = v.max();
  }
  return lsum;
}

// RESTITUISCE UN SOTTOINSIEME DI UN VETTORE DATA UNA CONDIZIONE
vec sub_elem_eq(vec v, vec w, double x) {
  uvec iw = find(w == x);
  if (!iw.is_empty()) v = v.elem(iw);
  else v.fill(datum::nan);
  return v;
}

// RESTITUISCE UNA SOTTOMATRICE CON COLONNE E RIGHE INDICATE IN UN VETTORE
mat sub_mat(const mat& M, const vec& vr, const vec& vc) {
  uli lvr = vr.n_elem, lvc = vc.n_elem;
  mat Q(lvr, lvc);
  for (uli c = 0; c < lvc; c++) {
    for (uli r = 0; r < lvr; r++) {
      Q(r, c) = M( (uword) vr(r), (uword) vc(c) );
    }
  }
  return Q;
}

// ELEVAMENTO A POTENZA ELEMENTO-PER-ELEMENTO (ritorno: vec)
vec pow_vec(const vec& v, const vec& w) {
  uli lv = v.n_elem;
  vec vv(lv);
  for (uli k = 0; k < lv; k++) vv(k) = std::pow(v(k), w(k));
  return vv;
}

//------------------------------------------------------------------------------------------------------------------------
// BINARY TREE
//------------------------------------------------------------------------------------------------------------------------

// AGGIUNGE IL MODELLO M ALL'ALBERO
field<mat> add_to_tree(vec M, double lM, uli nM, mat tree, double ltree) {

  uli j, k;
  field<mat> Res(2, 1);

  if (nM == 0) {
    for (j = 0; j <= lM; j++) {
      if (M(j) == 1) tree(j, 0) = j + 1;
      else           tree(j, 1) = j + 1;
    }
    ltree = lM;
    tree.row((uword)ltree) = tree.row((uword)ltree) * 0 + nM;
  }

  uli z, h, iM = 0;

  if (nM > 0) {
    z = 0;
    h = (uli)ltree + 1;

    for (j = 0; j <= lM; j++) {
      iM = 1 - M(j);

      if (!std::isfinite(tree(z, iM)) && (j <= lM)) {
        tree(z, iM) = h;

        for (k = (j + 1); k <= lM; k++) {
          if (M(k) == 1) tree(h, 0) = h + 1;
          else           tree(h, 1) = h + 1;
          h = h + 1;
        }

        iM   = 1 - M(lM);
        ltree = h - 1;
        break;
      }

      if (j == lM) { tree(z, iM) = nM; ltree = ltree + 1; break; }
      if (tree(z, iM) >= 0) z = (uli)tree(z, iM);
    }

    tree((uword)ltree, iM) = tree((uword)ltree, iM) * 0 + nM;
  }

  Res(0, 0) = tree;
  Res(1, 0) = mat(1, 1, fill::value(ltree));
  return Res;
}

// MOVIMENTI POSSIBILI DA M DATI I MODELLI PRECEDENTEMENTE VISITATI
vec mov_tree(const mat& tree, const vec& M, uli lM, const vec& vlM, uli max_lM) {

  uli k, z, h, iM2, q = 0;
  double sumM = sum(M);
  vec mov(lM + 1, fill::value(-1));

  for (k = 0; k <= lM; k++) {
    vec M2 = M; M2(k) = 1 - M(k);
    z = 0;

    for (h = 0; h <= lM; h++) {
      iM2 = 1 - M2(h);
      if (!std::isfinite(tree(z, iM2))) { mov(q++) = k; break; }
      else z = (uli)tree(z, iM2);
    }
  }

  uvec imov = find(mov > -1);
  if (imov.is_empty()) { mov.fill(datum::nan); return mov; }

  mov = mov.elem(imov);
  uvec umov = conv_to<uvec>::from(mov);

  if (sumM >= max_lM) {
    vec mov2 = zeros<vec>(lM + 1);
    mov2.elem(umov).ones();
    mov = (mov2 % M) % vlM;
    imov = find(mov > 0);
    if (imov.is_empty()) { mov.fill(datum::nan); }
    else { mov = mov.elem(imov) - 1; }
  }
  return mov;
}

//------------------------------------------------------------------------------------------------------------------------
// LEGATE AL PROBLEMA
//------------------------------------------------------------------------------------------------------------------------

double log_H_h_i(double mu, double sigma, uli h, uli i) {
  return lfactorial(2 * h) + i * log(sigma) - lfactorial(i)
       + (2 * h - 2 * i) * log(std::abs(mu)) - lfactorial(2 * h - 2 * i);
}


double log_FBF_Ga_Gb(vec G_a, vec G_b, uli edge, mat edges, mat YtY, uli add, double n, double h) {

  uli e1, e2, i;
  double p, b, S2, mu, sigma, logS2, ilogS2, logHhi, ilog4, log_num1, log_den1, log_w_1, log_num0i0, log_den0i0, log_w_0, log_FBF_unpasso;
  vec V1, V2, G1, V11, pa1, pa0, betah, vv(1), z1;
  uvec iw;
  mat e, yty, XtX, Xty;

  e  = edges.row(edge);
  e1 = (uli)e(0);
  e2 = (uli)e(1);

  V1 = edges.col(0);
  V2 = edges.col(1);

  G1 = (add == 1) ? G_a : G_b;

  V11 = (V1 + 1) % G1;
  iw  = find(V2 == (double)e2); pa1 = V11.elem(iw);
  iw  = find(pa1 > 0);          pa1 = pa1.elem(iw); pa1 = pa1 - 1;

  iw = find(pa1 != (double)e1);
  if (!iw.is_empty()) pa0 = pa1.elem(iw);
  else                pa0.fill(datum::nan);

  p = pa1.n_elem;
  b = (p + 2 * h + 1) / n;

  yty = YtY(e2, e2);

  // calcolo w1
  vv(0) = e2; Xty = sub_mat(YtY, pa1, vv);
  XtX    = sub_mat(YtY, pa1, pa1);
  betah  = solve(XtX, Xty);
  S2     = as_scalar(yty - (trans(Xty) * betah));

  iw = find(pa1 == (double)e1);
  if (iw.n_elem > 0) {
    mu = as_scalar(betah.elem(iw));
    uword iwi_u = iw(0);

    z1 = zeros<vec>(pa1.n_elem);
    z1(iwi_u) = 1;
    z1 = solve(XtX, z1);
    sigma = as_scalar(z1.elem(iw));
  } else {
    mu = 0.0;
    sigma = std::numeric_limits<double>::infinity();
  }

  if (S2 > 0) {
    log_w_1 = (-(n * (1 - b) / 2.0)) * log(datum::pi * b * S2);
    logS2   = log(S2);
    log_num1 = -datum::inf;
    log_den1 = -datum::inf;

    for (i = 0; i <= h; i++) {
      ilogS2 = i * logS2;
      logHhi = log_H_h_i(mu, sigma, h, i);
      ilog4  = -i * log(4.0);

      log_num1 = log_add(log_num1, (ilog4 + logHhi + lgamma((n - p - 2 * i) / 2.0) + ilogS2));
      log_den1 = log_add(log_den1, (ilog4 + logHhi + lgamma((n * b - p - 2 * i) / 2.0) + ilogS2));
    }

    log_w_1 = log_w_1 + log_num1 - log_den1;
  } else {
    log_w_1 = datum::inf;
  }

  // calcolo w0
  if (!pa0.is_finite()) p = 0;
  else                  p = pa0.n_elem;

  log_num0i0 = lgamma((n - p) / 2.0);
  log_den0i0 = lgamma((n * b - p) / 2.0);

  if (p == 0) {
    S2 = as_scalar(yty);
  } else {
    vv(0) = e2; Xty = sub_mat(YtY, pa0, vv);
    XtX    = sub_mat(YtY, pa0, pa0);
    betah  = solve(XtX, Xty);
    S2     = as_scalar(yty - (trans(Xty) * betah));
  }

  if (S2 > 0) {
    log_w_0 = (-(n * (1 - b) / 2.0)) * log(datum::pi * b * S2) + log_num0i0 - log_den0i0;
  } else {
    log_w_0 = datum::inf;
  }

  // calcolo FBF
  log_FBF_unpasso = (add == 1) ? (log_w_1 - log_w_0) : (log_w_0 - log_w_1);
  if (!std::isfinite(log_FBF_unpasso)) log_FBF_unpasso = 0.0;

  return log_FBF_unpasso;
}


field<mat> FBF_heart(double nt, mat YtY, vec vG_base, double lcv, vec vlcv, mat edges, double n_tot_mod, double C, double maxne, double h) {

  uli t, add, edge, imq, limodR, s;
  double ltree, lM, sum_log_FBF, log_FBF_G, log_pi_G, log_num_MP_G, sum_log_RSMP, n_mod_r, log_FBF_t, log_FBF1;
  vec M_log_FBF, log_num_MP, log_sume, G, imod_R, M_log_RSMP, pRSMP, mov, vlM, qh, G_t, M_q, M_P;
  uvec iw;
  mat tree, M_G;
  field<mat> treeRes, Res(3, 1);
  uword i_n_mod_r, imaxe;

  M_G        = zeros<mat>(lcv, n_tot_mod);
  M_P        = zeros<vec>(n_tot_mod);
  M_log_FBF  = zeros<vec>(n_tot_mod);
  log_num_MP = zeros<vec>(n_tot_mod);
  M_q        = zeros<vec>(lcv);
  tree       = zeros<mat>(n_tot_mod * lcv, 2); tree.fill(datum::nan);
  ltree      = datum::nan;
  lM         = lcv - 1;

  sum_log_FBF = -datum::inf;
  log_sume    = zeros<vec>(lcv); log_sume.fill(-datum::inf);

  M_log_RSMP   = zeros<vec>(n_tot_mod);
  sum_log_RSMP = -datum::inf;
  imod_R       = zeros<vec>(n_tot_mod);

  // inizializzazione: esplora i vicini di vG_base
  for (t = 0; t < lcv; t++) {
    G = vG_base;
    G(t) = 1 - vG_base(t);
    add  = (uli)G(t);
    edge = t;

    log_FBF_G = log_FBF_Ga_Gb(G, vG_base, edge, edges, YtY, add, nt, h);

    M_G.col(t) = G;

    treeRes = add_to_tree(G, lM, t, tree, ltree);
    tree    = treeRes(0, 0);
    ltree   = as_scalar(treeRes(1, 0));

    M_log_FBF(t) = log_FBF_G;
    log_pi_G     = -log(lcv + 1.0) - lchoose((uli)lcv, (uli)sum(G));
    log_num_MP_G = log_FBF_G + log_pi_G;
    log_num_MP(t)= log_num_MP_G;

    sum_log_FBF  = log_add(sum_log_FBF, log_num_MP_G);

    for (imq = 0; imq < lcv; imq++) {
      if (G(imq) == 1) log_sume(imq) = log_add(log_sume(imq), log_num_MP_G);
    }

    M_q = exp(log_sume - sum_log_FBF);

    M_log_RSMP(t)   = log_num_MP_G;
    sum_log_RSMP    = log_add(sum_log_RSMP, log_num_MP_G);
    imod_R(t)       = t;
  }

  limodR = t - 1;
  s      = lcv;

  // espansione
  while (t < n_tot_mod) {

    pRSMP      = exp(M_log_RSMP.subvec(0, limodR) - sum_log_RSMP);
    i_n_mod_r  = pRSMP.index_max();
    n_mod_r    = imod_R(i_n_mod_r);

    G          = M_G.col((uword)n_mod_r);
    G_t        = G;
    log_FBF_t  = M_log_FBF((uword)n_mod_r);

    vlM = vlcv + 1;
    mov = mov_tree(tree, G, lM, vlM, (uli)maxne);

    // nessuna mossa valida
    if (!mov.is_finite()) {
      imod_R(i_n_mod_r)   = -1;
      iw                  = find(imod_R > -1);
      imod_R              = imod_R.elem(iw);
      M_log_RSMP          = M_log_RSMP.elem(iw);
      limodR              = limodR - 1;
      t                   = t - 1;
    } else {
      qh  = pow_vec((M_q + C) / (1 - M_q + C), (2 * (1 - G)) - 1);
      qh  = qh.elem(conv_to<uvec>::from(mov));

      if (mov.n_elem == 1) {
        imod_R(i_n_mod_r) = -1;
        iw                 = find(imod_R > -1);
        imod_R             = imod_R.elem(iw);
        M_log_RSMP         = M_log_RSMP.elem(iw);
        limodR             = limodR - 1;
        edge               = (uli)mov(0);
      } else {
        imaxe = qh.index_max();
        edge  = (uli)mov(imaxe);
      }

      G(edge) = 1 - G(edge);
      add     = (uli)G(edge);

      log_FBF1   = log_FBF_Ga_Gb(G, G_t, edge, edges, YtY, add, nt, h);
      log_FBF_G  = log_FBF1 + log_FBF_t;

      M_G.col(t) = G;

      treeRes = add_to_tree(G, lM, t, tree, ltree);
      tree    = treeRes(0, 0);
      ltree   = as_scalar(treeRes(1, 0));

      M_log_FBF(t)  = log_FBF_G;
      log_pi_G      = -log(lcv + 1.0) - lchoose((uli)lcv, (uli)sum(G));
      log_num_MP_G  = log_FBF_G + log_pi_G;
      log_num_MP(t) = log_num_MP_G;

      sum_log_FBF = log_add(sum_log_FBF, log_num_MP_G);

      for (imq = 0; imq < lcv; imq++) {
        if (G(imq) == 1) log_sume(imq) = log_add(log_sume(imq), log_num_MP_G);
      }
      M_q = exp(log_sume - sum_log_FBF);

      limodR            = limodR + 1;
      imod_R(limodR)    = t;
      M_log_RSMP(limodR)= log_num_MP_G;
    }

    t = t + 1;
    s = s + 1;
  }

  t = t - 1;
  s = s - 1;

  M_P.subvec(0, t) = exp(log_num_MP.subvec(0, t) - sum_log_FBF);
  if (M_P.subvec(0, t).max() > 0) {
    M_P.subvec(0, t) = M_P.subvec(0, t) / sum(M_P.subvec(0, t));
  } else {
    M_P.subvec(0, t) = zeros<vec>(t + 1);
  }

  M_G = M_G.submat(0, 0, (uword)lcv - 1, t);

  Res(0, 0) = M_q;
  Res(1, 0) = M_G;
  Res(2, 0) = M_P;
  return Res;
}


// RIEMPIE LA MATRICE G_fin con gli elementi di M_q
mat G_fin_fill(mat G, const vec& vr, uli ic, const vec& x) {
  uli lvr = vr.n_elem;
  for (uli k = 0; k < lvr; k++) {
    G((uword)vr(k), ic) = x(k);
  }
  return G;
}

//---------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------

// LOCAL SEARCH
RcppExport SEXP FBF_LS(SEXP Corr, SEXP nobs, SEXP G_base, SEXP h, SEXP C, SEXP n_tot_mod) {

  uli k, neq, rr;
  double maxne, Mlogbin_sum, lcv, rrmax, q;
  vec V1, V2, vlcv, vG_base, M_q;
  mat edges, G_fin;
  field<mat> heartRes;

  Rcpp::NumericMatrix Corr_c1(Corr);
  Rcpp::NumericVector nobs_c1(nobs);
  Rcpp::NumericMatrix G_base_c1(G_base);
  Rcpp::NumericVector h_c1(h);
  Rcpp::NumericVector C_c1(C);
  Rcpp::NumericVector n_tot_mod_c1(n_tot_mod);

  q = Corr_c1.ncol();

  mat Corr_c(Corr_c1.begin(), (uword)q, (uword)q, false);
  mat G_base_c(G_base_c1.begin(), (uword)q, (uword)q, false);
  double nobs_c      = nobs_c1[0];
  double h_c         = h_c1[0];
  double C_c         = C_c1[0];
  double n_tot_mod_c = n_tot_mod_c1[0];

  q = Corr_c.n_cols;

  G_fin = zeros<mat>(q, q);
  maxne = nobs_c - 2 * h_c - 2;

  for (k = 0; k < (q - 1); k++) {

    neq = k + 1;

    V1    = linspace<vec>(q - neq - 1, 0, q - neq);
    V2    = zeros<vec>(q - neq); V2.fill(q - neq);
    edges = join_rows(V1, V2);

    lcv   = V1.n_elem;
    vlcv  = linspace<vec>(0, lcv - 1, lcv);

    vG_base = flipud(G_base_c.submat(0, q - neq, q - neq - 1, q - neq));

    rrmax = std::min(maxne, lcv);
    Mlogbin_sum = 0;

    for (rr = 1; rr <= rrmax; rr++) {
      Mlogbin_sum = log_add(Mlogbin_sum, lchoose((uli)lcv, (uli)rr));
    }

    n_tot_mod_c = std::min(Mlogbin_sum, log(n_tot_mod_c));
    n_tot_mod_c = round(exp(n_tot_mod_c));

    heartRes = FBF_heart(nobs_c, Corr_c * nobs_c, vG_base, lcv, vlcv, edges, n_tot_mod_c, C_c, maxne, h_c);
    M_q      = heartRes(0, 0);

    G_fin = G_fin_fill(G_fin, V1, (uli)V2[0], M_q);
  }

  return Rcpp::wrap(G_fin);
}


// REGRESSION SEARCH
RcppExport SEXP FBF_RS(SEXP Corr, SEXP nobs, SEXP G_base, SEXP h, SEXP C, SEXP n_tot_mod, SEXP n_hpp) {

  uli neq, rr, j;
  double maxne, Mlogbin_sum, lcv, rrmax, q;
  vec V1, V2, vlcv, vG_base, M_q, M_P, iM_P, M_P2;
  mat edges, M_G, M_G2;
  field<mat> heartRes;

  Rcpp::NumericMatrix Corr_c1(Corr);
  Rcpp::NumericVector nobs_c1(nobs);
  Rcpp::NumericVector G_base_c1(G_base);
  Rcpp::NumericVector h_c1(h);
  Rcpp::NumericVector C_c1(C);
  Rcpp::NumericVector n_tot_mod_c1(n_tot_mod);
  Rcpp::NumericVector n_hpp_c1(n_hpp);

  q = Corr_c1.ncol();

  mat Corr_c(Corr_c1.begin(), (uword)q, (uword)q, false);
  vec G_base_c(G_base_c1.begin(), (uword)q - 1, false);
  double nobs_c      = nobs_c1[0];
  double h_c         = h_c1[0];
  double C_c         = C_c1[0];
  double n_tot_mod_c = n_tot_mod_c1[0];
  double n_hpp_c     = n_hpp_c1[0];

  q = Corr_c.n_cols;

  maxne = nobs_c - 2 * h_c - 2;

  neq = 1;

  V1    = linspace<vec>(q - neq - 1, 0, q - neq);
  V2    = zeros<vec>(q - neq); V2.fill(q - neq);
  edges = join_rows(V1, V2);

  lcv   = V1.n_elem;
  vlcv  = linspace<vec>(0, lcv - 1, lcv);

  vG_base = flipud(G_base_c);

  rrmax = std::min(maxne, lcv);
  Mlogbin_sum = 0;

  for (rr = 1; rr <= rrmax; rr++) {
    Mlogbin_sum = log_add(Mlogbin_sum, lchoose((uli)lcv, (uli)rr));
  }

  n_tot_mod_c = std::min(Mlogbin_sum, log(n_tot_mod_c));
  n_tot_mod_c = round(exp(n_tot_mod_c));

  heartRes = FBF_heart(nobs_c, Corr_c * nobs_c, vG_base, lcv, vlcv, edges, n_tot_mod_c, C_c, maxne, h_c);

  M_q = heartRes(0, 0);
  M_G = heartRes(1, 0);
  M_P = heartRes(2, 0);

  iM_P = conv_to<vec>::from(sort_index(conv_to<vec>::from(M_P), "descend"));

  M_P2 = zeros<vec>((uword)n_hpp_c);
  M_G2 = zeros<mat>(lcv, (uword)n_hpp_c);

  for (j = 0; j < (uli)n_hpp_c; j++) {
    M_P2(j)    = M_P((uword)iM_P(j));
    M_G2.col(j)= M_G.col((uword)iM_P(j));
  }

  return Rcpp::List::create(Rcpp::Named("M_q") = M_q,
                            Rcpp::Named("M_G") = M_G2,
                            Rcpp::Named("M_P") = M_P2);
}


// GLOBAL SEARCH
RcppExport SEXP FBF_GS(SEXP Corr, SEXP nobs, SEXP G_base, SEXP h, SEXP C, SEXP n_tot_mod, SEXP n_hpp) {

  uli j, k, neq, rr;
  double rrmax, Mlogbin_sum, maxne, lcv, nc_edges, nc_edges2, q;
  vec V1, V2, vlcv, vG_base, M_q, M_P, iM_P, M_P2, M_G_j;
  mat edges, M_q2, M_G, M_G2, M_G2_j;
  field<mat> heartRes;

  Rcpp::NumericMatrix Corr_c1(Corr);
  Rcpp::NumericVector nobs_c1(nobs);
  Rcpp::NumericMatrix G_base_c1(G_base);
  Rcpp::NumericVector h_c1(h);
  Rcpp::NumericVector C_c1(C);
  Rcpp::NumericVector n_tot_mod_c1(n_tot_mod);
  Rcpp::NumericVector n_hpp_c1(n_hpp);

  q = Corr_c1.ncol();

  mat Corr_c(Corr_c1.begin(), (uword)q, (uword)q, false);
  mat G_base_c(G_base_c1.begin(), (uword)q, (uword)q, false);
  double nobs_c      = nobs_c1[0];
  double h_c         = h_c1[0];
  double C_c         = C_c1[0];
  double n_tot_mod_c = n_tot_mod_c1[0];
  double n_hpp_c     = n_hpp_c1[0];

  q = Corr_c.n_cols;

  lcv  = q * (q - 1) / 2.0;
  vlcv = linspace<vec>(0, lcv - 1, lcv);

  maxne = nobs_c - 2 * h_c - 2;

  edges   = zeros<mat>(lcv, 2);
  vG_base = zeros<vec>(lcv);

  nc_edges = 0;

  for (k = 0; k < (q - 1); k++) {

    neq  = k + 1;
    V1   = linspace<vec>(q - neq - 1, 0, q - neq);
    V2   = zeros<vec>(q - neq); V2.fill(q - neq);

    nc_edges2 = nc_edges + (q - neq);

    edges.rows((uword)nc_edges, (uword)nc_edges2 - 1) =
      join_rows(V1, V2);
    vG_base.subvec((uword)nc_edges, (uword)nc_edges2 - 1) =
      flipud(G_base_c.submat(0, q - neq, q - neq - 1, q - neq));

    nc_edges = nc_edges2;
  }

  rrmax = std::min(maxne, lcv);
  Mlogbin_sum = 0;
  for (rr = 1; rr <= rrmax; rr++) {
    Mlogbin_sum = log_add(Mlogbin_sum, lchoose((uli)lcv, (uli)rr));
  }
  n_tot_mod_c = std::min(Mlogbin_sum, log(n_tot_mod_c));
  n_tot_mod_c = round(exp(n_tot_mod_c));

  heartRes = FBF_heart(nobs_c, Corr_c * nobs_c, vG_base, lcv, vlcv, edges, n_tot_mod_c, C_c, maxne, h_c);

  M_q = conv_to<vec>::from(heartRes(0, 0));
  M_G = heartRes(1, 0);
  M_P = conv_to<vec>::from(heartRes(2, 0));

  M_q2 = zeros<mat>(q, q);
  for (j = 0; j < lcv; j++) {
    M_q2((uword)edges(j, 0), (uword)edges(j, 1)) = M_q(j);
  }

  iM_P  = conv_to<vec>::from(sort_index(conv_to<vec>::from(M_P), "descend"));
  M_P2  = zeros<vec>((uword)n_hpp_c);
  M_G2_j= zeros<mat>(lcv, (uword)n_hpp_c);
  M_G2  = zeros<mat>(q * lcv, q);

  for (j = 0; j < (uli)n_hpp_c; j++) {
    M_P2(j) = M_P((uword)iM_P(j));

    M_G_j  = M_G.col((uword)iM_P(j));
    M_G2_j = zeros<mat>(q, q);

    for (k = 0; k < lcv; k++) {
      M_G2_j((uword)edges(k, 0), (uword)edges(k, 1)) = M_G_j(k);
    }

    M_G2.rows((uword)j * q, (uword)((j + 1) * q - 1)) = M_G2_j;
  }

  return Rcpp::List::create(Rcpp::Named("M_q2") = M_q2,
                            Rcpp::Named("M_G2") = M_G2,
                            Rcpp::Named("M_P2") = M_P2);
}
